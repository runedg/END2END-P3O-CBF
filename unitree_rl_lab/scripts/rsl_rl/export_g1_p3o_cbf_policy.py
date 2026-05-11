import argparse
from pathlib import Path

import torch
import yaml

from rsl_rl.modules.actor_critic_safe import ActorCriticSafe


def build_deploy_cfg(template_cfg: dict, stage: str) -> dict:
    cfg = yaml.safe_load(yaml.safe_dump(template_cfg))
    observations = {
        "base_ang_vel": cfg["observations"]["base_ang_vel"],
        "projected_gravity": cfg["observations"]["projected_gravity"],
        "velocity_commands": cfg["observations"]["velocity_commands"],
        "joint_pos_rel": cfg["observations"]["joint_pos_rel"],
        "joint_vel_rel": cfg["observations"]["joint_vel_rel"],
        "last_action": cfg["observations"]["last_action"],
    }
    if stage == "walk":
        cfg["commands"]["base_velocity"]["ranges"]["lin_vel_x"] = [0.2, 0.6]
        cfg["commands"]["base_velocity"]["ranges"]["lin_vel_y"] = [-0.08, 0.08]
        cfg["commands"]["base_velocity"]["ranges"]["ang_vel_z"] = [-0.15, 0.15]
    else:
        cfg["commands"]["base_velocity"]["ranges"]["lin_vel_x"] = [0.15, 0.8]
        cfg["commands"]["base_velocity"]["ranges"]["lin_vel_y"] = [-0.15, 0.15]
        cfg["commands"]["base_velocity"]["ranges"]["ang_vel_z"] = [-0.25, 0.25]
        observations["obstacle_scan"] = {
            "params": {"num_rays": 9, "max_distance": 6.0, "fov_deg": 180.0},
            "clip": [0.0, 6.0],
            "scale": [1.0] * 9,
            "history_length": 5,
        }
    cfg["observations"] = observations
    return cfg


def compute_obs_dim(deploy_cfg: dict) -> int:
    total = 0
    for term_cfg in deploy_cfg["observations"].values():
        term_dim = len(term_cfg["scale"])
        total += term_dim * int(term_cfg["history_length"])
    return total


class PolicyExportWrapper(torch.nn.Module):
    def __init__(self, policy: ActorCriticSafe):
        super().__init__()
        self.policy = policy

    def forward(self, obs):
        return self.policy.act_inference(obs)


def main():
    parser = argparse.ArgumentParser(description="Export G1 P3O-CBF checkpoint to ONNX for MuJoCo sim2sim.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--stage", type=str, default="avoid", choices=["walk", "avoid"])
    parser.add_argument(
        "--template_deploy_yaml",
        type=str,
        default="/home/ubuntu/P3O-CBF/unitree_rl_lab/deploy/robots/g1_29dof/config/policy/velocity/v0/params/deploy.yaml",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    (output_dir / "params").mkdir(parents=True, exist_ok=True)
    (output_dir / "exported").mkdir(parents=True, exist_ok=True)

    template_cfg = yaml.safe_load(Path(args.template_deploy_yaml).read_text(encoding="utf-8"))
    deploy_cfg = build_deploy_cfg(template_cfg, args.stage)
    obs_dim = compute_obs_dim(deploy_cfg)
    action_dim = len(deploy_cfg["actions"]["JointPositionAction"]["scale"])
    critic_obs_dim = obs_dim + 3 * 5

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint["policy_state_dict"]

    policy = ActorCriticSafe(
        num_actor_obs=obs_dim,
        num_critic_obs=critic_obs_dim,
        num_actions=action_dim,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
    )
    policy.load_state_dict(state_dict, strict=True)
    policy.eval()

    wrapper = PolicyExportWrapper(policy)
    dummy = torch.zeros(1, obs_dim, dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        dummy,
        output_dir / "exported" / "policy.onnx",
        input_names=["obs"],
        output_names=["actions"],
        opset_version=13,
    )

    with open(output_dir / "params" / "deploy.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(deploy_cfg, f, sort_keys=False)
    with open(output_dir / "params" / "checkpoint.txt", "w", encoding="utf-8") as f:
        f.write(str(Path(args.checkpoint).resolve()) + "\n")


if __name__ == "__main__":
    main()
