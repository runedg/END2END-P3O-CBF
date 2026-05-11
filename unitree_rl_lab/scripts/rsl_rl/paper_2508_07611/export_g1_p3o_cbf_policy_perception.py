import argparse
from pathlib import Path

import torch
import yaml

from actor_critic_safe_perception import ActorCriticSafePerception, ObsTermSpec


def build_actor_term_specs(history_length: int, lidar_dim: int, lidar_term_name: str) -> list[ObsTermSpec]:
    return [
        ObsTermSpec("base_ang_vel", 3, history_length),
        ObsTermSpec("projected_gravity", 3, history_length),
        ObsTermSpec("velocity_commands", 3, history_length),
        ObsTermSpec(lidar_term_name, lidar_dim, history_length),
        ObsTermSpec("joint_pos_rel", 29, history_length),
        ObsTermSpec("joint_vel_rel", 29, history_length),
        ObsTermSpec("last_action", 29, history_length),
    ]


def build_critic_term_specs(history_length: int, lidar_dim: int, lidar_term_name: str) -> list[ObsTermSpec]:
    return [
        ObsTermSpec("base_lin_vel", 3, history_length),
        ObsTermSpec("base_ang_vel", 3, history_length),
        ObsTermSpec("projected_gravity", 3, history_length),
        ObsTermSpec("velocity_commands", 3, history_length),
        ObsTermSpec(lidar_term_name, lidar_dim, history_length),
        ObsTermSpec("joint_pos_rel", 29, history_length),
        ObsTermSpec("joint_vel_rel", 29, history_length),
        ObsTermSpec("last_action", 29, history_length),
    ]


def build_deploy_cfg(template_cfg: dict, stage: str, history_length: int, num_rays: int, max_distance: float, fov_deg: float) -> dict:
    cfg = yaml.safe_load(yaml.safe_dump(template_cfg))
    observations = {
        "base_ang_vel": cfg["observations"]["base_ang_vel"],
        "projected_gravity": cfg["observations"]["projected_gravity"],
        "velocity_commands": cfg["observations"]["velocity_commands"],
        "obstacle_scan": {
            "params": {"num_rays": num_rays, "max_distance": max_distance, "fov_deg": fov_deg},
            "clip": [0.0, max_distance],
            "scale": [1.0] * num_rays,
            "history_length": history_length,
        },
        "joint_pos_rel": cfg["observations"]["joint_pos_rel"],
        "joint_vel_rel": cfg["observations"]["joint_vel_rel"],
        "last_action": cfg["observations"]["last_action"],
    }
    for term_name in observations:
        if term_name != "obstacle_scan":
            observations[term_name]["history_length"] = history_length

    if stage == "walk":
        cfg["commands"]["base_velocity"]["ranges"]["lin_vel_x"] = [0.2, 0.6]
        cfg["commands"]["base_velocity"]["ranges"]["lin_vel_y"] = [-0.08, 0.08]
        cfg["commands"]["base_velocity"]["ranges"]["ang_vel_z"] = [-0.15, 0.15]
    elif stage == "clutter":
        cfg["commands"]["base_velocity"]["ranges"]["lin_vel_x"] = [0.0, 0.6]
        cfg["commands"]["base_velocity"]["ranges"]["lin_vel_y"] = [-0.18, 0.18]
        cfg["commands"]["base_velocity"]["ranges"]["ang_vel_z"] = [-0.65, 0.65]
    else:
        cfg["commands"]["base_velocity"]["ranges"]["lin_vel_x"] = [0.15, 0.8]
        cfg["commands"]["base_velocity"]["ranges"]["lin_vel_y"] = [-0.15, 0.15]
        cfg["commands"]["base_velocity"]["ranges"]["ang_vel_z"] = [-0.25, 0.25]
    cfg["observations"] = observations
    return cfg


def compute_obs_dim(deploy_cfg: dict) -> int:
    total = 0
    for term_cfg in deploy_cfg["observations"].values():
        term_dim = len(term_cfg["scale"])
        total += term_dim * int(term_cfg["history_length"])
    return total


class PolicyExportWrapper(torch.nn.Module):
    def __init__(self, policy: ActorCriticSafePerception):
        super().__init__()
        self.policy = policy

    def forward(self, obs):
        return self.policy.act_inference(obs)


def infer_arch_from_state_dict(state_dict: dict) -> dict[str, int]:
    proprio_hidden_dim = int(state_dict["actor_encoder.proprio_frame_encoder.0.weight"].shape[0])
    scan_hidden_dim = int(state_dict["actor_encoder.scan_frame_encoder.0.weight"].shape[0])
    rnn_hidden_dim = int(state_dict["actor_encoder.proprio_gru.weight_hh_l0"].shape[1])
    lidar_mode = "pointcloud" if "actor_encoder.point_mlp.0.weight" in state_dict else "scan"
    return {
        "proprio_hidden_dim": proprio_hidden_dim,
        "scan_hidden_dim": scan_hidden_dim,
        "rnn_hidden_dim": rnn_hidden_dim,
        "lidar_mode": lidar_mode,
    }


def main():
    parser = argparse.ArgumentParser(description="Export the perception-enhanced G1 P3O-CBF checkpoint to ONNX.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--stage", type=str, default="avoid", choices=["walk", "avoid", "clutter"])
    parser.add_argument("--history_length", type=int, default=5)
    parser.add_argument("--lidar_num_rays", type=int, default=64)
    parser.add_argument("--lidar_fov_deg", type=float, default=180.0)
    parser.add_argument("--lidar_max_distance", type=float, default=6.0)
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
    deploy_cfg = build_deploy_cfg(
        template_cfg,
        args.stage,
        args.history_length,
        args.lidar_num_rays,
        args.lidar_max_distance,
        args.lidar_fov_deg,
    )
    obs_dim = compute_obs_dim(deploy_cfg)
    action_dim = len(deploy_cfg["actions"]["JointPositionAction"]["scale"])
    critic_obs_dim = obs_dim + 3 * args.history_length

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint["policy_state_dict"]

    arch_cfg = infer_arch_from_state_dict(state_dict)
    lidar_term_name = "lidar_points" if arch_cfg["lidar_mode"] == "pointcloud" else "obstacle_scan"

    policy = ActorCriticSafePerception(
        actor_term_specs=build_actor_term_specs(args.history_length, args.lidar_num_rays, lidar_term_name),
        critic_term_specs=build_critic_term_specs(args.history_length, args.lidar_num_rays, lidar_term_name),
        num_actions=action_dim,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
        activation="elu",
        init_noise_std=1.0,
        proprio_hidden_dim=arch_cfg["proprio_hidden_dim"],
        scan_hidden_dim=arch_cfg["scan_hidden_dim"],
        rnn_hidden_dim=arch_cfg["rnn_hidden_dim"],
    )
    policy.load_state_dict(state_dict, strict=True)
    policy.eval()

    wrapper = PolicyExportWrapper(policy)
    wrapper.eval()
    dummy = torch.zeros(1, obs_dim, dtype=torch.float32)
    traced = torch.jit.trace(wrapper, dummy)
    traced.save(str(output_dir / "policy.pt"))
    torch.onnx.export(
        wrapper,
        dummy,
        output_dir / "exported" / "policy.onnx",
        input_names=["obs"],
        output_names=["actions"],
        opset_version=18,
    )

    with open(output_dir / "params" / "deploy.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(deploy_cfg, f, sort_keys=False)
    with open(output_dir / "params" / "checkpoint.txt", "w", encoding="utf-8") as f:
        f.write(str(Path(args.checkpoint).resolve()) + "\n")


if __name__ == "__main__":
    main()
