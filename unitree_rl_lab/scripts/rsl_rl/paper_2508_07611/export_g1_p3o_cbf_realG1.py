import argparse
from pathlib import Path

import torch
import yaml

from actor_critic_safe_perception import ActorCriticSafePerception, ObsTermSpec


def build_actor_term_specs(history_length: int, num_features: int) -> list[ObsTermSpec]:
    return [
        ObsTermSpec("base_ang_vel", 3, history_length),
        ObsTermSpec("projected_gravity", 3, history_length),
        ObsTermSpec("velocity_commands", 3, history_length),
        ObsTermSpec("obstacle_scan", num_features, history_length),
        ObsTermSpec("joint_pos_rel", 29, history_length),
        ObsTermSpec("joint_vel_rel", 29, history_length),
        ObsTermSpec("last_action", 29, history_length),
    ]


def build_critic_term_specs(history_length: int, num_features: int) -> list[ObsTermSpec]:
    return [
        ObsTermSpec("base_lin_vel", 3, history_length),
        ObsTermSpec("base_ang_vel", 3, history_length),
        ObsTermSpec("projected_gravity", 3, history_length),
        ObsTermSpec("velocity_commands", 3, history_length),
        ObsTermSpec("obstacle_scan", num_features, history_length),
        ObsTermSpec("joint_pos_rel", 29, history_length),
        ObsTermSpec("joint_vel_rel", 29, history_length),
        ObsTermSpec("last_action", 29, history_length),
    ]


def build_deploy_cfg(template_cfg: dict, args) -> dict:
    cfg = yaml.safe_load(yaml.safe_dump(template_cfg))
    observations = {
        "base_ang_vel": cfg["observations"]["base_ang_vel"],
        "projected_gravity": cfg["observations"]["projected_gravity"],
        "velocity_commands": cfg["observations"]["velocity_commands"],
        "obstacle_scan": {
            "params": {
                "compression_mode": "realg1_mid360_range64",
                "feature_dim": args.lidar_feature_dim,
                "pattern_file": args.omni_pattern_file,
                "samples": args.omni_point_samples,
                "max_distance": args.lidar_max_distance,
                "horizontal_fov_deg": args.compression_fov_deg,
                "sensor_offset": [args.sensor_offset_x, args.sensor_offset_y, args.sensor_offset_z],
                "roi_x_min": args.roi_x_min,
                "roi_x_max": args.roi_x_max,
                "roi_abs_y_max": args.roi_abs_y_max,
                "roi_z_min": args.roi_z_min,
                "roi_z_max": args.roi_z_max,
                "min_planar_distance": args.min_planar_distance,
            },
            "clip": [0.0, args.lidar_max_distance],
            "scale": [1.0] * args.lidar_feature_dim,
            "history_length": args.history_length,
        },
        "joint_pos_rel": cfg["observations"]["joint_pos_rel"],
        "joint_vel_rel": cfg["observations"]["joint_vel_rel"],
        "last_action": cfg["observations"]["last_action"],
    }
    for term_name in observations:
        if term_name != "obstacle_scan":
            observations[term_name]["history_length"] = args.history_length

    if args.stage == "walk":
        cfg["commands"]["base_velocity"]["ranges"]["lin_vel_x"] = [0.2, 0.6]
        cfg["commands"]["base_velocity"]["ranges"]["lin_vel_y"] = [-0.08, 0.08]
        cfg["commands"]["base_velocity"]["ranges"]["ang_vel_z"] = [-0.15, 0.15]
    elif args.stage == "clutter":
        cfg["commands"]["base_velocity"]["ranges"]["lin_vel_x"] = [0.0, 3.0]
        cfg["commands"]["base_velocity"]["ranges"]["lin_vel_y"] = [-0.35, 0.35]
        cfg["commands"]["base_velocity"]["ranges"]["ang_vel_z"] = [-0.8, 0.8]
    else:
        cfg["commands"]["base_velocity"]["ranges"]["lin_vel_x"] = [0.15, 0.8]
        cfg["commands"]["base_velocity"]["ranges"]["lin_vel_y"] = [-0.15, 0.15]
        cfg["commands"]["base_velocity"]["ranges"]["ang_vel_z"] = [-0.25, 0.25]

    cfg["observations"] = observations
    return cfg


def compute_obs_dim(deploy_cfg: dict) -> int:
    total = 0
    for term_cfg in deploy_cfg["observations"].values():
        total += len(term_cfg["scale"]) * int(term_cfg["history_length"])
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
    return {
        "proprio_hidden_dim": proprio_hidden_dim,
        "scan_hidden_dim": scan_hidden_dim,
        "rnn_hidden_dim": rnn_hidden_dim,
    }


def main():
    parser = argparse.ArgumentParser(description="Export the deployment-oriented realG1 G1 P3O-CBF checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--stage", type=str, default="clutter", choices=["walk", "avoid", "clutter"])
    parser.add_argument("--history_length", type=int, default=5)
    parser.add_argument("--lidar_feature_dim", type=int, default=64)
    parser.add_argument("--lidar_max_distance", type=float, default=6.0)
    parser.add_argument("--compression_fov_deg", type=float, default=180.0)
    parser.add_argument(
        "--omni_pattern_file",
        type=str,
        default="/home/ubuntu/P3O-CBF/OmniPerception/LidarSensor/LidarSensor/sensor_pattern/sensor_lidar/scan_mode/mid360.npy",
    )
    parser.add_argument("--omni_point_samples", type=int, default=1024)
    parser.add_argument("--roi_x_min", type=float, default=-0.5)
    parser.add_argument("--roi_x_max", type=float, default=6.0)
    parser.add_argument("--roi_abs_y_max", type=float, default=3.0)
    parser.add_argument("--roi_z_min", type=float, default=-1.0)
    parser.add_argument("--roi_z_max", type=float, default=0.8)
    parser.add_argument("--min_planar_distance", type=float, default=0.2)
    parser.add_argument("--sensor_offset_x", type=float, default=0.10)
    parser.add_argument("--sensor_offset_y", type=float, default=0.0)
    parser.add_argument("--sensor_offset_z", type=float, default=0.63)
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
    deploy_cfg = build_deploy_cfg(template_cfg, args)
    obs_dim = compute_obs_dim(deploy_cfg)
    action_dim = len(deploy_cfg["actions"]["JointPositionAction"]["scale"])

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint["policy_state_dict"]
    arch_cfg = infer_arch_from_state_dict(state_dict)

    policy = ActorCriticSafePerception(
        actor_term_specs=build_actor_term_specs(args.history_length, args.lidar_feature_dim),
        critic_term_specs=build_critic_term_specs(args.history_length, args.lidar_feature_dim),
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
