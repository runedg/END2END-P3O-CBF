"""Standalone video recorder for P3O_V2_Tuned checkpoint.

Loads the v2-tuned checkpoint into ActorCriticSafePerception, runs the
ObstacleAvoidance-realG1 environment with Mid360 point-cloud observations,
and writes an mp4 video.

Does NOT depend on existing eval scripts.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import math
import os

import cv2
import torch

from isaaclab.app import AppLauncher
from actor_critic_safe_perception import ActorCriticSafePerception, ObsTermSpec
from cuda_reduce_workaround import apply_cuda_reduce_workaround

apply_cuda_reduce_workaround()

# ── CLI ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Record P3O_V2_Tuned eval video.")
parser.add_argument("--checkpoint", type=str, required=True)
# NOTE: --device is added automatically by AppLauncher.add_app_launcher_args()
# Do NOT define it here to avoid ArgParser conflict.
parser.add_argument("--steps", type=int, default=1800)
parser.add_argument("--num_envs", type=int, default=1)
# NOTE: --headless is added automatically by AppLauncher.add_app_launcher_args()
parser.add_argument("--topdown_follow", action="store_true", default=False)
parser.add_argument("--follow_robot", action="store_true", default=False)
parser.add_argument("--name", type=str, default=None)
parser.add_argument("--video_fps", type=float, default=30.0)

# Perception params (defaults match checkpoint training args)
parser.add_argument("--history_length", type=int, default=5)
parser.add_argument("--omni_point_samples", type=int, default=1024)
parser.add_argument("--num_fps_points", type=int, default=128)
parser.add_argument(
    "--omni_pattern_file",
    type=str,
    default="/home/ubuntu/P3O-CBF/OmniPerception/LidarSensor/LidarSensor/sensor_pattern/sensor_lidar/scan_mode/mid360.npy",
)
parser.add_argument("--lidar_max_distance", type=float, default=6.0)
parser.add_argument("--compression_fov_deg", type=float, default=180.0)
parser.add_argument("--roi_x_min", type=float, default=-0.5)
parser.add_argument("--roi_x_max", type=float, default=6.0)
parser.add_argument("--roi_abs_y_max", type=float, default=3.0)
parser.add_argument("--roi_z_min", type=float, default=-1.0)
parser.add_argument("--roi_z_max", type=float, default=0.8)
parser.add_argument("--min_planar_distance", type=float, default=0.2)
parser.add_argument("--sensor_offset_x", type=float, default=0.10)
parser.add_argument("--sensor_offset_y", type=float, default=0.0)
parser.add_argument("--sensor_offset_z", type=float, default=0.63)
parser.add_argument("--enable_sensor_noise", action="store_true", default=True)
parser.add_argument("--random_distance_noise", type=float, default=0.02)
parser.add_argument("--pixel_dropout_prob", type=float, default=0.01)
parser.add_argument("--sector_dropout_prob", type=float, default=0.10)
parser.add_argument("--sector_dropout_width_deg", type=float, default=8.0)
parser.add_argument("--translation_noise_x", type=float, default=0.015)
parser.add_argument("--translation_noise_y", type=float, default=0.015)
parser.add_argument("--translation_noise_z", type=float, default=0.015)
parser.add_argument("--rotation_noise_roll_deg", type=float, default=2.0)
parser.add_argument("--rotation_noise_pitch_deg", type=float, default=2.0)
parser.add_argument("--rotation_noise_yaw_deg", type=float, default=2.0)

# Safety / CBF (from training defaults)
parser.add_argument("--safety_margin", type=float, default=0.55)
parser.add_argument("--cbf_gamma", type=float, default=0.35)
parser.add_argument("--collision_distance", type=float, default=0.18)
parser.add_argument("--robot_body_radius", type=float, default=0.4)

# Command
parser.add_argument("--cmd_vx", type=float, default=0.30)
parser.add_argument("--cmd_vy", type=float, default=0.0)
parser.add_argument("--cmd_wz", type=float, default=0.0)

# Terrain / obstacle
parser.add_argument("--terrain_clutter", action="store_true", default=False)
parser.add_argument("--terrain_obstacles", type=int, default=4)
parser.add_argument(
    "--terrain_layout", type=str, default="continuous_avoidance",
    choices=[
        "lanes", "surround_wide", "surround_hybrid", "surround_nonconvex",
        "front_dense_hybrid", "u_wall", "u_wall_gap", "continuous_avoidance",
        "surrounded_front_open", "mixed_obstacles_no_wall",
    ],
)
parser.add_argument("--terrain_size_x", type=float, default=30.0)
parser.add_argument("--terrain_size_y", type=float, default=24.0)
parser.add_argument("--terrain_x_min", type=float, default=-10.5)
parser.add_argument("--terrain_x_max", type=float, default=10.5)
parser.add_argument("--terrain_y_span", type=float, default=8.5)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: E401, E402
import unitree_rl_lab.tasks  # noqa: F401, E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from unitree_rl_lab.tasks.locomotion.obstacle_manager import ObstacleManager  # noqa: E402
from unitree_rl_lab.tasks.locomotion import mdp  # noqa: E402
from isaaclab.managers import ObservationTermCfg as ObsTerm  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402

TASK_NAME = "Unitree-G1-29dof-ObstacleAvoidance-realG1"


# ── obs specs (match training) ─────────────────────────────────────────
def build_actor_specs(h: int, n_pts: int) -> list[ObsTermSpec]:
    return [
        ObsTermSpec("base_ang_vel", 3, h),
        ObsTermSpec("projected_gravity", 3, h),
        ObsTermSpec("velocity_commands", 3, h),
        ObsTermSpec("lidar_points", n_pts * 3, h),
        ObsTermSpec("joint_pos_rel", 29, h),
        ObsTermSpec("joint_vel_rel", 29, h),
        ObsTermSpec("last_action", 29, h),
    ]


def build_critic_specs(h: int, n_pts: int) -> list[ObsTermSpec]:
    return [
        ObsTermSpec("base_lin_vel", 3, h),
        ObsTermSpec("base_ang_vel", 3, h),
        ObsTermSpec("projected_gravity", 3, h),
        ObsTermSpec("velocity_commands", 3, h),
        ObsTermSpec("lidar_points", n_pts * 3, h),
        ObsTermSpec("joint_pos_rel", 29, h),
        ObsTermSpec("joint_vel_rel", 29, h),
        ObsTermSpec("last_action", 29, h),
    ]


# ── env configuration ──────────────────────────────────────────────────
def configure_env(env_cfg):
    import isaaclab.terrains as terrain_gen

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.safety_margin = args_cli.safety_margin
    env_cfg.cbf_gamma = args_cli.cbf_gamma
    env_cfg.collision_distance = args_cli.collision_distance
    env_cfg.lidar_max_distance = args_cli.lidar_max_distance
    env_cfg.compression_fov_deg = args_cli.compression_fov_deg
    env_cfg.roi_x_min = args_cli.roi_x_min
    env_cfg.roi_x_max = args_cli.roi_x_max
    env_cfg.roi_abs_y_max = args_cli.roi_abs_y_max
    env_cfg.roi_z_min = args_cli.roi_z_min
    env_cfg.roi_z_max = args_cli.roi_z_max
    env_cfg.min_planar_distance = args_cli.min_planar_distance
    env_cfg.robot_body_radius = args_cli.robot_body_radius

    env_cfg.num_obstacles = args_cli.terrain_obstacles

    if args_cli.terrain_clutter:
        obstacle_env_cfg_mod = importlib.import_module(
            "unitree_rl_lab.tasks.locomotion.robots.g1.29dof.obstacle_env_cfg"
        )
        MeshClutterPillarsTerrainCfg = obstacle_env_cfg_mod.MeshClutterPillarsTerrainCfg
        terrain_rows = int(math.ceil(math.sqrt(args_cli.num_envs)))
        terrain_cols = int(math.ceil(args_cli.num_envs / terrain_rows))
        env_cfg.scene.env_spacing = max(env_cfg.scene.env_spacing, 20.0)
        env_cfg.scene.terrain.max_init_terrain_level = terrain_rows - 1
        env_cfg.scene.terrain.terrain_generator = terrain_gen.TerrainGeneratorCfg(
            size=(args_cli.terrain_size_x, args_cli.terrain_size_y),
            border_width=0.0,
            num_rows=terrain_rows,
            num_cols=terrain_cols,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            difficulty_range=(0.0, 0.0),
            use_cache=False,
            sub_terrains={
                "clutter_pillars": MeshClutterPillarsTerrainCfg(
                    proportion=1.0,
                    num_obstacles=env_cfg.num_obstacles,
                    radius=env_cfg.obstacle_radius,
                    height=env_cfg.obstacle_height,
                    x_range=(args_cli.terrain_x_min, args_cli.terrain_x_max),
                    y_range=(-args_cli.terrain_y_span, args_cli.terrain_y_span),
                    layout_variant=args_cli.terrain_layout,
                ),
            },
        )
        env_cfg.obstacle_layout_mode = "clutter"
        env_cfg.obstacle_layout_variant = args_cli.terrain_layout
        env_cfg.obstacle_x_range = (args_cli.terrain_x_min, args_cli.terrain_x_max)
        env_cfg.obstacle_y_range = (-args_cli.terrain_y_span, args_cli.terrain_y_span)
        env_cfg.obstacle_collision_mode = "terrain_mesh"

    # Point-cloud observations
    params = {
        "pattern_file": args_cli.omni_pattern_file,
        "samples": args_cli.omni_point_samples,
        "num_fps_points": args_cli.num_fps_points,
        "max_distance": args_cli.lidar_max_distance,
        "sensor_offset": (args_cli.sensor_offset_x, args_cli.sensor_offset_y, args_cli.sensor_offset_z),
        "horizontal_fov_deg": args_cli.compression_fov_deg,
        "roi_x_min": args_cli.roi_x_min,
        "roi_x_max": args_cli.roi_x_max,
        "roi_abs_y_max": args_cli.roi_abs_y_max,
        "roi_z_min": args_cli.roi_z_min,
        "roi_z_max": args_cli.roi_z_max,
        "min_planar_distance": args_cli.min_planar_distance,
        "enable_sensor_noise": args_cli.enable_sensor_noise,
        "random_distance_noise": args_cli.random_distance_noise,
        "pixel_dropout_prob": args_cli.pixel_dropout_prob,
        "sector_dropout_prob": args_cli.sector_dropout_prob,
        "sector_dropout_width_deg": args_cli.sector_dropout_width_deg,
        "random_translation_range": (
            args_cli.translation_noise_x,
            args_cli.translation_noise_y,
            args_cli.translation_noise_z,
        ),
        "random_rotation_deg_range": (
            args_cli.rotation_noise_roll_deg,
            args_cli.rotation_noise_pitch_deg,
            args_cli.rotation_noise_yaw_deg,
        ),
    }
    point_dim = args_cli.num_fps_points * 3
    scale = tuple([1.0 / args_cli.lidar_max_distance] * point_dim)

    env_cfg.observations.policy.obstacle_scan = ObsTerm(
        func=mdp.omni_lidar_realg1_pointcloud_fps,
        params=params,
        clip=(-args_cli.lidar_max_distance, args_cli.lidar_max_distance),
        scale=scale,
    )
    env_cfg.observations.policy.history_length = args_cli.history_length
    env_cfg.observations.critic.obstacle_scan = ObsTerm(
        func=mdp.omni_lidar_realg1_pointcloud_fps,
        params=params,
        clip=(-args_cli.lidar_max_distance, args_cli.lidar_max_distance),
        scale=scale,
    )
    env_cfg.observations.critic.history_length = args_cli.history_length

    # Episode length
    env_cfg.episode_length_s = max(120.0, args_cli.steps * env_cfg.decimation * env_cfg.sim.dt + 5.0)

    # Fix initial pose
    if hasattr(env_cfg, "events") and hasattr(env_cfg.events, "reset_base"):
        env_cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}

    # Fixed command
    if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "base_velocity"):
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (args_cli.cmd_vx, args_cli.cmd_vx)
        env_cfg.commands.base_velocity.ranges.lin_vel_y = (args_cli.cmd_vy, args_cli.cmd_vy)
        env_cfg.commands.base_velocity.ranges.ang_vel_z = (args_cli.cmd_wz, args_cli.cmd_wz)
        env_cfg.commands.base_velocity.resampling_time_range = (120.0, 120.0)
        env_cfg.commands.base_velocity.rel_standing_envs = 0.0

    # Relax terminations for long eval
    if hasattr(env_cfg, "terminations"):
        if hasattr(env_cfg.terminations, "base_height"):
            env_cfg.terminations.base_height.params["minimum_height"] = -1.0
        if hasattr(env_cfg.terminations, "bad_orientation"):
            env_cfg.terminations.bad_orientation.params["limit_angle"] = 3.14


# ── camera follow ──────────────────────────────────────────────────────
def set_camera(env):
    robot = env.unwrapped.scene["robot"]
    root_pos = robot.data.root_pos_w[0]
    if args_cli.topdown_follow:
        env.unwrapped.sim.set_camera_view(
            eye=[float(root_pos[0] + 0.2), float(root_pos[1]), float(root_pos[2] + 16.0)],
            target=[float(root_pos[0] + 1.5), float(root_pos[1]), float(root_pos[2] + 0.2)],
        )
        return
    root_quat = robot.data.root_quat_w[0]
    forward = quat_apply(
        root_quat.unsqueeze(0),
        torch.tensor([[[1.0, 0.0, 0.0]]], device=root_pos.device),
    )[0, 0]
    if args_cli.follow_robot:
        env.unwrapped.sim.set_camera_view(
            eye=[float(root_pos[0] - 3.2 * forward[0]), float(root_pos[1] - 3.2 * forward[1] - 1.2), float(root_pos[2] + 2.0)],
            target=[float(root_pos[0] + 2.8 * forward[0]), float(root_pos[1] + 2.8 * forward[1]), float(root_pos[2] + 0.8)],
        )
        return
    env.unwrapped.sim.set_camera_view(
        eye=[float(root_pos[0] - 3.2 * forward[0]), float(root_pos[1] - 3.2 * forward[1] - 1.2), float(root_pos[2] + 2.0)],
        target=[float(root_pos[0] + 2.8 * forward[0]), float(root_pos[1] + 2.8 * forward[1]), float(root_pos[2] + 0.8)],
    )


# ── capture ────────────────────────────────────────────────────────────
def capture_frame(env):
    set_camera(env)
    env.unwrapped.sim.render()
    frame = env.render()
    if frame is None:
        raise RuntimeError("env.render() returned None")
    if frame.dtype != "uint8":
        frame = frame.clip(0, 255).astype("uint8")
    if float(frame.mean()) < 2.0:
        env.unwrapped.sim.render()
        retry = env.render()
        if retry is not None:
            if retry.dtype != "uint8":
                retry = retry.clip(0, 255).astype("uint8")
            if float(retry.mean()) >= float(frame.mean()):
                frame = retry
    return frame


# ── metrics ────────────────────────────────────────────────────────────
def compute_metrics(env) -> dict:
    robot = env.unwrapped.scene["robot"]
    command = mdp.generated_commands(env.unwrapped, command_name="base_velocity")
    cmd_vx = float(command[0, 0])
    cmd_vy = float(command[0, 1])
    cmd_wz = float(command[0, 2])

    act_vx = float(robot.data.root_lin_vel_b[0, 0])
    act_vy = float(robot.data.root_lin_vel_b[0, 1])
    act_wz = float(robot.data.root_ang_vel_b[0, 2])
    root_height = float(robot.data.root_pos_w[0, 2])
    gravity_xy = float(torch.linalg.norm(robot.data.projected_gravity_b[0, :2]))

    return {
        "cmd_vx": cmd_vx,
        "cmd_vy": cmd_vy,
        "cmd_wz": cmd_wz,
        "act_vx": act_vx,
        "act_vy": act_vy,
        "act_wz": act_wz,
        "root_height": root_height,
        "gravity_xy_norm": gravity_xy,
        "fallen": float((root_height < 0.45) or (gravity_xy > 0.8)),
    }


# ── main ───────────────────────────────────────────────────────────────
def main():
    env_cfg = load_cfg_from_registry(TASK_NAME, "env_cfg_entry_point")
    configure_env(env_cfg)

    env = gym.make(TASK_NAME, cfg=env_cfg, render_mode="rgb_array")
    env.unwrapped.obstacle_manager = ObstacleManager(env.unwrapped, env.unwrapped.cfg)
    env.unwrapped.obstacle_manager.spawn_obstacles()

    num_obs = env.unwrapped.observation_space["policy"].shape[-1]
    num_priv_obs = env.unwrapped.observation_space["critic"].shape[-1]
    num_actions = env.unwrapped.action_space.shape[-1]

    h = args_cli.history_length
    n_pts = args_cli.num_fps_points
    policy = ActorCriticSafePerception(
        actor_term_specs=build_actor_specs(h, n_pts),
        critic_term_specs=build_critic_specs(h, n_pts),
        num_actions=num_actions,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
        activation="elu",
        init_noise_std=1.0,
        proprio_hidden_dim=128,
        scan_hidden_dim=64,
        rnn_hidden_dim=64,
    ).to(args_cli.device)

    # Sanity checks
    if policy.actor_encoder.input_dim != num_obs:
        raise RuntimeError(
            f"Actor obs dim mismatch: encoder={policy.actor_encoder.input_dim}, env={num_obs}"
        )
    if policy.critic_encoder.input_dim != num_priv_obs:
        raise RuntimeError(
            f"Critic obs dim mismatch: encoder={policy.critic_encoder.input_dim}, env={num_priv_obs}"
        )

    # Load checkpoint
    ckpt = torch.load(args_cli.checkpoint, map_location=args_cli.device, weights_only=False)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()

    print(f"[INFO] Loaded checkpoint: {args_cli.checkpoint}")
    print(f"[INFO] Iteration: {ckpt.get('iteration', 'unknown')}")
    print(f"[INFO] Command: vx={args_cli.cmd_vx} vy={args_cli.cmd_vy} wz={args_cli.cmd_wz}")
    print(f"[INFO] Terrain: obstacles={args_cli.terrain_obstacles} layout={args_cli.terrain_layout}")

    # Reset
    reset_result = env.reset()
    obs_dict = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    obs = obs_dict["policy"].to(args_cli.device)

    # Video output
    video_dir = os.path.join(os.path.dirname(args_cli.checkpoint), "videos")
    os.makedirs(video_dir, exist_ok=True)
    video_name = args_cli.name or "v2_tuned_eval.mp4"
    if not (video_name.endswith(".mp4") or video_name.endswith(".avi")):
        video_name = f"{video_name}.mp4"
    video_path = os.path.join(video_dir, video_name)
    metrics_path = os.path.join(video_dir, video_name.rsplit(".", 1)[0] + "_metrics.csv")

    # Capture first frame for dimensions
    frame = capture_frame(env)
    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, float(args_cli.video_fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {video_path}")

    print(f"[INFO] Recording to {video_path}")
    print(f"[INFO] Metrics to {metrics_path}")

    with open(metrics_path, "w", newline="", encoding="utf-8") as mf:
        mw = csv.DictWriter(mf, fieldnames=["step"] + list(compute_metrics(env).keys()))
        mw.writeheader()

        for step in range(args_cli.steps):
            with torch.no_grad():
                actions = policy.act_inference(obs)
            step_result = env.step(actions)
            next_obs_dict = step_result[0]
            obs = next_obs_dict["policy"].to(args_cli.device)

            frame = capture_frame(env)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            metrics = compute_metrics(env)
            metrics["step"] = step
            mw.writerow(metrics)

            if step % 100 == 0:
                fallen_str = "FALLEN" if metrics["fallen"] else "OK"
                print(
                    f"[Step {step:04d}/{args_cli.steps}] "
                    f"act_vx={metrics['act_vx']:.3f} act_wz={metrics['act_wz']:.3f} "
                    f"height={metrics['root_height']:.3f} {fallen_str}"
                )

    writer.release()
    env.close()
    print(f"[INFO] Video saved: {video_path}")
    print(f"[INFO] Metrics saved: {metrics_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
