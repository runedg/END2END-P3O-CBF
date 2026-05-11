from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@configclass
class UniformTargetPositionCommandCfg:
    """Configuration for uniform target position command."""

    @configclass
    class Ranges:
        """Ranges for sampling target position."""

        x: tuple[float, float] = MISSING
        y: tuple[float, float] = MISSING

    class_type: type = MISSING
    resampling_time_range: tuple[float, float] = (20.0, 20.0)
    debug_vis: bool = True
    ranges: Ranges = MISSING

    def __post_init__(self):
        self.class_type = UniformTargetPositionCommand


class UniformTargetPositionCommand:
    """Command generator for target position navigation."""

    cfg: UniformTargetPositionCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: UniformTargetPositionCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command generator."""
        self.cfg = cfg
        self.env = env

        # create buffers for the command
        self.target_pos = torch.zeros(env.num_envs, 2, device=env.device)
        self.resampling_time = torch.zeros(env.num_envs, device=env.device)
        self.time = torch.zeros(env.num_envs, device=env.device)

        # initialize the command
        self.resample()

    def __str__(self) -> str:
        """Return a string representation of the class."""
        msg = "Target Position Command:\n"
        msg += f"\tTarget position: {self.target_pos}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """Return the current command."""
        return self.target_pos

    def resample(self, env_ids: torch.Tensor | None = None):
        """Resample the target position."""
        if env_ids is None:
            env_ids = slice(None)

        # sample target position
        self.target_pos[env_ids, 0] = torch.rand_like(self.target_pos[env_ids, 0]) * (
            self.cfg.ranges.x[1] - self.cfg.ranges.x[0]
        ) + self.cfg.ranges.x[0]
        self.target_pos[env_ids, 1] = torch.rand_like(self.target_pos[env_ids, 1]) * (
            self.cfg.ranges.y[1] - self.cfg.ranges.y[0]
        ) + self.cfg.ranges.y[0]

        # sample resampling time
        r = self.cfg.resampling_time_range
        self.resampling_time[env_ids] = torch.rand_like(self.resampling_time[env_ids]) * (r[1] - r[0]) + r[0]

    def update(self, dt: float):
        """Update the command generator."""
        # update time
        self.time += dt

        # check if resampling is needed
        resample_ids = self.time >= self.resampling_time
        if resample_ids.any():
            self.resample(resample_ids.nonzero(as_tuple=False).squeeze(-1))
            self.time[resample_ids] = 0.0

        # visualize
        if self.cfg.debug_vis:
            self._debug_vis()

    def _debug_vis(self):
        """Visualize the target position."""
        pass  # Can be implemented with Isaac Sim debug visualization

    def get_command(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Return the current command."""
        if env_ids is None:
            return self.target_pos
        return self.target_pos[env_ids]
