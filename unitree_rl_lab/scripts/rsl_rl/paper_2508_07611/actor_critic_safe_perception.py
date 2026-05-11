from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation


@dataclass(frozen=True)
class ObsTermSpec:
    name: str
    dim: int
    history: int


class TemporalObservationEncoder(nn.Module):
    """Encodes proprioceptive histories and dense LiDAR scans separately.

    The scan branch uses a 1D spatial encoder followed by a GRU over time.
    This is not raw point-cloud processing, but it is much closer to the
    paper's spatio-temporal LiDAR idea than the previous flat MLP.
    """

    def __init__(
        self,
        term_specs: list[ObsTermSpec],
        activation_name: str = "elu",
        scan_term_name: str = "obstacle_scan",
        pointcloud_term_name: str = "lidar_points",
        proprio_hidden_dim: int = 128,
        scan_hidden_dim: int = 128,
        rnn_hidden_dim: int = 128,
    ):
        super().__init__()
        self.term_specs = term_specs
        self.scan_term_name = scan_term_name
        self.pointcloud_term_name = pointcloud_term_name
        activation = resolve_nn_activation(activation_name)

        history_lengths = {spec.history for spec in term_specs}
        if len(history_lengths) != 1:
            raise ValueError(f"Expected a shared history length, got {history_lengths}")
        self.history_length = history_lengths.pop()

        self.term_offsets: dict[str, tuple[int, int, int]] = {}
        cursor = 0
        proprio_frame_dim = 0
        lidar_dim = None
        self.lidar_mode = "scan"
        for spec in term_specs:
            total = spec.dim * spec.history
            self.term_offsets[spec.name] = (cursor, cursor + total, spec.dim)
            cursor += total
            if spec.name == scan_term_name:
                lidar_dim = spec.dim
                self.lidar_mode = "scan"
            elif spec.name == pointcloud_term_name:
                lidar_dim = spec.dim
                self.lidar_mode = "pointcloud"
            else:
                proprio_frame_dim += spec.dim
        if lidar_dim is None:
            raise ValueError(f"Missing LiDAR term in {term_specs}")

        self.input_dim = cursor
        self.lidar_dim = lidar_dim
        self.proprio_frame_dim = proprio_frame_dim

        self.proprio_frame_encoder = nn.Sequential(
            nn.Linear(proprio_frame_dim, proprio_hidden_dim),
            activation,
            nn.Linear(proprio_hidden_dim, proprio_hidden_dim),
            activation,
        )
        self.proprio_gru = nn.GRU(
            input_size=proprio_hidden_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        if self.lidar_mode == "scan":
            self.scan_spatial_encoder = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=5, padding=2),
                activation,
                nn.Conv1d(16, 32, kernel_size=5, padding=2),
                activation,
                nn.AdaptiveAvgPool1d(8),
            )
            scan_in_dim = 32 * 8
        else:
            self.num_points = self.lidar_dim // 3
            if self.lidar_dim % 3 != 0:
                raise ValueError(f"Point-cloud term dimension must be divisible by 3, got {self.lidar_dim}")
            self.point_mlp = nn.Sequential(
                nn.Linear(3, 32),
                activation,
                nn.Linear(32, 64),
                activation,
                nn.Linear(64, 64),
                activation,
            )
            scan_in_dim = 64
        self.scan_frame_encoder = nn.Sequential(
            nn.Linear(scan_in_dim, scan_hidden_dim),
            activation,
            nn.Linear(scan_hidden_dim, scan_hidden_dim),
            activation,
        )
        self.scan_gru = nn.GRU(
            input_size=scan_hidden_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.output_dim = 2 * rnn_hidden_dim + scan_hidden_dim

    def _split_terms(self, observations: torch.Tensor) -> dict[str, torch.Tensor]:
        if observations.shape[-1] != self.input_dim:
            raise ValueError(f"Expected obs dim {self.input_dim}, got {observations.shape[-1]}")

        terms: dict[str, torch.Tensor] = {}
        for spec in self.term_specs:
            start, end, term_dim = self.term_offsets[spec.name]
            term = observations[:, start:end].view(observations.shape[0], self.history_length, term_dim)
            terms[spec.name] = term
        return terms

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        terms = self._split_terms(observations)
        lidar_name = self.pointcloud_term_name if self.pointcloud_term_name in terms else self.scan_term_name
        scan_seq = terms.pop(lidar_name)
        proprio_names = [spec.name for spec in self.term_specs if spec.name not in {self.scan_term_name, self.pointcloud_term_name}]
        proprio_seq = torch.cat([terms[name] for name in proprio_names], dim=-1)

        proprio_feat = self.proprio_frame_encoder(proprio_seq)
        _, proprio_hidden = self.proprio_gru(proprio_feat)
        proprio_hidden = proprio_hidden[-1]

        batch_size = observations.shape[0]
        if self.lidar_mode == "scan":
            scan_seq = scan_seq.reshape(batch_size * self.history_length, 1, self.lidar_dim)
            scan_feat = self.scan_spatial_encoder(scan_seq).flatten(1)
        else:
            scan_seq = scan_seq.reshape(batch_size * self.history_length, self.num_points, 3)
            point_feat = self.point_mlp(scan_seq)
            scan_feat = point_feat.max(dim=1).values
        scan_feat = self.scan_frame_encoder(scan_feat)
        scan_feat = scan_feat.view(batch_size, self.history_length, -1)
        _, scan_hidden = self.scan_gru(scan_feat)
        scan_hidden = scan_hidden[-1]
        scan_latest = scan_feat[:, -1, :]

        return torch.cat((proprio_hidden, scan_hidden, scan_latest), dim=-1)


class ActorCriticSafePerception(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        actor_term_specs: list[ObsTermSpec],
        critic_term_specs: list[ObsTermSpec],
        num_actions: int,
        actor_hidden_dims: list[int] | None = None,
        critic_hidden_dims: list[int] | None = None,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        proprio_hidden_dim: int = 128,
        scan_hidden_dim: int = 128,
        rnn_hidden_dim: int = 128,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticSafePerception.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        actor_hidden_dims = actor_hidden_dims or [256, 128]
        critic_hidden_dims = critic_hidden_dims or [256, 128]
        activation_layer = resolve_nn_activation(activation)

        self.actor_encoder = TemporalObservationEncoder(
            actor_term_specs,
            activation_name=activation,
            proprio_hidden_dim=proprio_hidden_dim,
            scan_hidden_dim=scan_hidden_dim,
            rnn_hidden_dim=rnn_hidden_dim,
        )
        self.critic_encoder = TemporalObservationEncoder(
            critic_term_specs,
            activation_name=activation,
            proprio_hidden_dim=proprio_hidden_dim,
            scan_hidden_dim=scan_hidden_dim,
            rnn_hidden_dim=rnn_hidden_dim,
        )

        def build_mlp(input_dim: int, hidden_dims: list[int], output_dim: int):
            layers: list[nn.Module] = []
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(activation_layer)
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, output_dim))
            return nn.Sequential(*layers)

        self.actor = build_mlp(self.actor_encoder.output_dim, actor_hidden_dims, num_actions)
        self.critic = build_mlp(self.critic_encoder.output_dim, critic_hidden_dims, 1)
        self.cost_critic = build_mlp(self.critic_encoder.output_dim, critic_hidden_dims, 1)

        print(f"Actor encoder: {self.actor_encoder}")
        print(f"Critic encoder: {self.critic_encoder}")
        print(f"Actor head: {self.actor}")
        print(f"Critic head: {self.critic}")
        print(f"Cost critic head: {self.cost_critic}")

        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")

        self.distribution = None
        Normal.set_default_validate_args(False)

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(self.actor_encoder(observations))
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        return self.actor(self.actor_encoder(observations))

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(self.critic_encoder(critic_observations))

    def cost_evaluate(self, critic_observations, **kwargs):
        return self.cost_critic(self.critic_encoder(critic_observations))

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True
