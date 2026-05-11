#!/usr/bin/env python3

"""Paper-aligned realG1 Mid360 P3O-CBF training with structured curriculum."""

from __future__ import annotations

import argparse
import math
import os
from datetime import datetime

import importlib

import torch

from cuda_reduce_workaround import apply_cuda_reduce_workaround
from isaaclab.app import AppLauncher

from actor_critic_safe_perception import ActorCriticSafePerception, ObsTermSpec


apply_cuda_reduce_workaround()

parser = argparse.ArgumentParser(description="Train a paper-aligned realG1 Mid360 P3O-CBF policy with curriculum.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments.")
parser.add_argument("--max_iterations", type=int, default=10000, help="Training iterations.")
parser.add_argument("--save_interval", type=int, default=1000, help="Save checkpoint every N iterations.")
parser.add_argument("--experiment_name", type=str, default="End2EndP3OPaperCurriculum", help="Top-level log folder name.")
parser.add_argument("--headless", action="store_true", default=False, help="Run in headless mode.")
parser.add_argument("--device", type=str, default="cuda:0", help="Device to use.")
parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint to resume from.")
parser.add_argument("--num_steps_per_env", type=int, default=24, help="Rollout horizon per environment.")
parser.add_argument("--num_learning_epochs", type=int, default=5, help="Number of PPO/P3O epochs per update.")
parser.add_argument("--num_mini_batches", type=int, default=24, help="Number of mini-batches per update.")
parser.add_argument("--force_stage", type=int, default=None, help="Optional fixed curriculum stage index for fine-tuning.")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="Actor-critic optimizer learning rate.")
parser.add_argument("--cost_critic_learning_rate", type=float, default=1e-4, help="Cost critic optimizer learning rate.")
parser.add_argument("--entropy_coef", type=float, default=0.01, help="Policy entropy coefficient.")
parser.add_argument("--reset_optimizers", action="store_true", default=False, help="Resume policy weights but reinitialize optimizers.")
parser.add_argument("--cost_limit", type=float, default=0.22, help="Rollout cost limit for the paper-style P3O term.")
parser.add_argument("--kappa", type=float, default=1.0, help="Penalty coefficient.")
parser.add_argument("--cost_gamma", type=float, default=0.99, help="Cost discount factor.")
parser.add_argument("--cost_lam", type=float, default=0.95, help="Cost GAE lambda.")
parser.add_argument("--safety_margin", type=float, default=0.8, help="Paper-aligned safe distance to obstacles.")
parser.add_argument("--cbf_gamma", type=float, default=0.5, help="Discrete CBF coefficient.")
parser.add_argument("--collision_distance", type=float, default=0.2, help="Distance counted as collision.")
parser.add_argument("--unsafe_cost_weight", type=float, default=1.0, help="Weight for entering the unsafe set.")
parser.add_argument("--cbf_cost_weight", type=float, default=1.0, help="Weight for discrete CBF violation.")
parser.add_argument("--collision_cost_weight", type=float, default=2.0, help="Extra weight for near-collision states.")
parser.add_argument(
    "--omni_pattern_file",
    type=str,
    default="/home/ubuntu/P3O-CBF/OmniPerception/LidarSensor/LidarSensor/sensor_pattern/sensor_lidar/scan_mode/mid360.npy",
    help="Path to OmniPerception Mid360 pattern file.",
)
parser.add_argument("--history_length", type=int, default=5, help="Observation history length.")
parser.add_argument("--omni_point_samples", type=int, default=1024, help="Number of Mid360 points sampled per frame.")
parser.add_argument("--num_fps_points", type=int, default=128, help="Number of FPS points retained per frame.")
parser.add_argument("--lidar_max_distance", type=float, default=6.0, help="Maximum LiDAR range.")
parser.add_argument("--compression_fov_deg", type=float, default=180.0, help="Horizontal FOV retained after filtering.")
parser.add_argument("--roi_x_min", type=float, default=-0.5, help="Min x in base frame kept for compression.")
parser.add_argument("--roi_x_max", type=float, default=6.0, help="Max x in base frame kept for compression.")
parser.add_argument("--roi_abs_y_max", type=float, default=3.0, help="Absolute y limit in base frame kept for compression.")
parser.add_argument("--roi_z_min", type=float, default=-1.0, help="Min z in base frame kept for compression.")
parser.add_argument("--roi_z_max", type=float, default=0.8, help="Max z in base frame kept for compression.")
parser.add_argument("--min_planar_distance", type=float, default=0.2, help="Minimum planar range retained.")
parser.add_argument("--robot_body_radius", type=float, default=0.4, help="Conservative planar body radius subtracted from LiDAR obstacle distance.")
parser.add_argument("--contact_cost_weight", type=float, default=4.0, help="Extra P3O cost for non-foot body contacts.")
parser.add_argument("--contact_force_threshold", type=float, default=1.0, help="Net contact force threshold for link-level collision cost.")
parser.add_argument("--sensor_offset_x", type=float, default=0.10, help="Nominal sensor x offset in base frame.")
parser.add_argument("--sensor_offset_y", type=float, default=0.0, help="Nominal sensor y offset in base frame.")
parser.add_argument("--sensor_offset_z", type=float, default=0.63, help="Nominal sensor z offset in base frame.")
parser.add_argument("--enable_sensor_noise", action="store_true", default=True, help="Enable deployment-oriented Mid360 corruption.")
parser.add_argument("--random_distance_noise", type=float, default=0.02, help="Gaussian distance noise std in meters.")
parser.add_argument("--pixel_dropout_prob", type=float, default=0.01, help="Independent point dropout probability.")
parser.add_argument("--sector_dropout_prob", type=float, default=0.10, help="Probability of dropping one angular sector per frame.")
parser.add_argument("--sector_dropout_width_deg", type=float, default=8.0, help="Angular width of dropped sectors.")
parser.add_argument("--translation_noise_x", type=float, default=0.015, help="Sensor x translation jitter range.")
parser.add_argument("--translation_noise_y", type=float, default=0.015, help="Sensor y translation jitter range.")
parser.add_argument("--translation_noise_z", type=float, default=0.015, help="Sensor z translation jitter range.")
parser.add_argument("--rotation_noise_roll_deg", type=float, default=2.0, help="Sensor roll jitter range in degrees.")
parser.add_argument("--rotation_noise_pitch_deg", type=float, default=2.0, help="Sensor pitch jitter range in degrees.")
parser.add_argument("--rotation_noise_yaw_deg", type=float, default=2.0, help="Sensor yaw jitter range in degrees.")

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab.managers import ObservationTermCfg as ObsTerm
from torch.utils.tensorboard import SummaryWriter

from p3o_cbf_paper import P3OCBFPaper


CURRICULUM_STAGES = (
    {
        "name": "stage0_single_front",
        "progress": 0.0,
        "num_obstacles": 1,
        "layout": "continuous_avoidance",
        "lin_vel_x": (0.06, 0.18),
        "lin_vel_y": 0.04,
        "ang_vel_z": 0.16,
        "standing": 0.0,
        "track_lin": 1.20,
        "track_ang": 0.55,
        "proxemic": -0.75,
        "safe_vel": -0.55,
        "safe_acc": -0.08,
        "tangential": 1.20,
        "arm_dev": -1.00,
        "alive": 0.25,
    },
    {
        "name": "stage1_two_front_choices",
        "progress": 0.18,
        "num_obstacles": 2,
        "layout": "continuous_avoidance",
        "lin_vel_x": (0.06, 0.20),
        "lin_vel_y": 0.05,
        "ang_vel_z": 0.18,
        "standing": 0.0,
        "track_lin": 1.25,
        "track_ang": 0.60,
        "proxemic": -0.70,
        "safe_vel": -0.50,
        "safe_acc": -0.08,
        "tangential": 1.15,
        "arm_dev": -1.00,
        "alive": 0.25,
    },
    {
        "name": "stage2_four_open",
        "progress": 0.34,
        "num_obstacles": 4,
        "layout": "continuous_avoidance",
        "lin_vel_x": (0.08, 0.22),
        "lin_vel_y": 0.06,
        "ang_vel_z": 0.20,
        "standing": 0.0,
        "track_lin": 1.30,
        "track_ang": 0.65,
        "proxemic": -0.65,
        "safe_vel": -0.45,
        "safe_acc": -0.07,
        "tangential": 1.05,
        "arm_dev": -1.05,
        "alive": 0.25,
    },
    {
        "name": "stage3_eight_open",
        "progress": 0.50,
        "num_obstacles": 8,
        "layout": "continuous_avoidance",
        "lin_vel_x": (0.08, 0.25),
        "lin_vel_y": 0.08,
        "ang_vel_z": 0.26,
        "standing": 0.0,
        "track_lin": 1.35,
        "track_ang": 0.72,
        "proxemic": -0.58,
        "safe_vel": -0.40,
        "safe_acc": -0.05,
        "tangential": 0.95,
        "arm_dev": -1.10,
        "alive": 0.25,
    },
    {
        "name": "stage4_twelve_open",
        "progress": 0.66,
        "num_obstacles": 12,
        "layout": "continuous_avoidance",
        "lin_vel_x": (0.08, 0.28),
        "lin_vel_y": 0.10,
        "ang_vel_z": 0.32,
        "standing": 0.0,
        "track_lin": 1.40,
        "track_ang": 0.80,
        "proxemic": -0.48,
        "safe_vel": -0.34,
        "safe_acc": -0.05,
        "tangential": 0.85,
        "arm_dev": -1.10,
        "alive": 0.25,
    },
    {
        "name": "stage5_twenty_survival",
        "progress": 0.80,
        "num_obstacles": 20,
        "layout": "surrounded_front_open",
        "lin_vel_x": (0.06, 0.30),
        "lin_vel_y": 0.12,
        "ang_vel_z": 0.42,
        "standing": 0.0,
        "track_lin": 1.45,
        "track_ang": 0.86,
        "proxemic": -0.35,
        "safe_vel": -0.30,
        "safe_acc": -0.05,
        "tangential": 0.95,
        "arm_dev": -1.10,
        "alive": 0.25,
    },
    {
        "name": "stage6_mixed_final",
        "progress": 0.90,
        "num_obstacles": 28,
        "layout": "mixed_obstacles_no_wall",
        "lin_vel_x": (0.06, 0.32),
        "lin_vel_y": 0.16,
        "ang_vel_z": 0.50,
        "standing": 0.0,
        "track_lin": 1.45,
        "track_ang": 0.90,
        "proxemic": -0.22,
        "safe_vel": -0.28,
        "safe_acc": -0.05,
        "tangential": 1.15,
        "arm_dev": -1.10,
        "alive": 0.25,
    },
)


def _closest_obstacle_geometry(env):
    from unitree_rl_lab.tasks.locomotion import mdp

    return mdp.closest_mid360_obstacle(
        env,
        sensor_name="mid360_lidar",
        max_distance=args_cli.lidar_max_distance,
        horizontal_fov_deg=args_cli.compression_fov_deg,
        roi_x_min=args_cli.roi_x_min,
        roi_x_max=args_cli.roi_x_max,
        roi_abs_y_max=args_cli.roi_abs_y_max,
        roi_z_min=args_cli.roi_z_min,
        roi_z_max=args_cli.roi_z_max,
        min_planar_distance=args_cli.min_planar_distance,
        robot_body_radius=args_cli.robot_body_radius,
    )


def _non_foot_contact_cost(env, threshold: float) -> torch.Tensor:
    contact_sensor = env.scene.sensors["contact_forces"]
    force_norm = torch.linalg.norm(contact_sensor.data.net_forces_w, dim=-1)
    body_names = contact_sensor.body_names
    non_foot_ids = [idx for idx, name in enumerate(body_names) if "ankle" not in name]
    if not non_foot_ids:
        return torch.zeros(env.num_envs, device=env.device)
    body_ids = torch.tensor(non_foot_ids, device=env.device, dtype=torch.long)
    return torch.any(force_norm[:, body_ids] > threshold, dim=1).float()


def compute_paper_cost(env, cfg):
    robot = env.scene["robot"]
    robot_vel_xy = robot.data.root_lin_vel_w[:, :2]
    distances, directions = _closest_obstacle_geometry(env)
    h = distances - cfg.safety_margin
    h_dot = -torch.sum(robot_vel_xy * directions, dim=-1)
    dt = float(env.cfg.decimation * env.cfg.sim.dt)
    h_next_est = h + dt * h_dot
    cbf_margin = h_next_est - (1.0 - cfg.cbf_gamma) * h
    cbf_violation = torch.relu(-cbf_margin)
    unsafe_cost = (distances < cfg.safety_margin).float()
    collision_cost = (distances < cfg.collision_distance).float()
    contact_cost = _non_foot_contact_cost(env, cfg.contact_force_threshold)
    total_cost = (
        cfg.unsafe_cost_weight * unsafe_cost
        + cfg.cbf_cost_weight * cbf_violation
        + cfg.collision_cost_weight * collision_cost
        + cfg.contact_cost_weight * contact_cost
    )
    info = {
        "paper/distance_mean": distances.mean().item(),
        "paper/distance_min": distances.min().item(),
        "paper/cbf_violation_rate": (cbf_violation > 0).float().mean().item(),
        "paper/unsafe_rate": unsafe_cost.mean().item(),
        "paper/collision_rate": collision_cost.mean().item(),
        "paper/contact_rate": contact_cost.mean().item(),
    }
    return total_cost, info


def compute_tracking_metrics(env):
    from unitree_rl_lab.tasks.locomotion import mdp

    robot = env.scene["robot"]
    command = mdp.generated_commands(env, command_name="base_velocity")
    actual_lin = robot.data.root_lin_vel_b[:, :2]
    cmd_lin = command[:, :2]
    lin_error = torch.linalg.norm(actual_lin - cmd_lin, dim=-1)
    actual_speed = torch.linalg.norm(actual_lin, dim=-1)
    cmd_speed = torch.linalg.norm(cmd_lin, dim=-1)
    return {
        "track/lin_error": lin_error.mean().item(),
        "track/actual_speed": actual_speed.mean().item(),
        "track/command_speed": cmd_speed.mean().item(),
    }


def build_actor_term_specs(history_length: int, num_points: int) -> list[ObsTermSpec]:
    return [
        ObsTermSpec("base_ang_vel", 3, history_length),
        ObsTermSpec("projected_gravity", 3, history_length),
        ObsTermSpec("velocity_commands", 3, history_length),
        ObsTermSpec("lidar_points", num_points * 3, history_length),
        ObsTermSpec("joint_pos_rel", 29, history_length),
        ObsTermSpec("joint_vel_rel", 29, history_length),
        ObsTermSpec("last_action", 29, history_length),
    ]


def build_critic_term_specs(history_length: int, num_points: int) -> list[ObsTermSpec]:
    return [
        ObsTermSpec("base_lin_vel", 3, history_length),
        ObsTermSpec("base_ang_vel", 3, history_length),
        ObsTermSpec("projected_gravity", 3, history_length),
        ObsTermSpec("velocity_commands", 3, history_length),
        ObsTermSpec("lidar_points", num_points * 3, history_length),
        ObsTermSpec("joint_pos_rel", 29, history_length),
        ObsTermSpec("joint_vel_rel", 29, history_length),
        ObsTermSpec("last_action", 29, history_length),
    ]


def _configure_base_env(env_cfg):
    import isaaclab.terrains as terrain_gen

    obstacle_env_cfg_mod = __import__(
        "unitree_rl_lab.tasks.locomotion.robots.g1.29dof.obstacle_env_cfg",
        fromlist=["MeshClutterPillarsTerrainCfg"],
    )
    MeshClutterPillarsTerrainCfg = obstacle_env_cfg_mod.MeshClutterPillarsTerrainCfg

    terrain_rows = int(math.ceil(math.sqrt(args_cli.num_envs)))
    terrain_cols = int(math.ceil(args_cli.num_envs / terrain_rows))
    max_stage = max(CURRICULUM_STAGES, key=lambda stage: int(stage["num_obstacles"]))
    if args_cli.force_stage is None:
        terrain_stage = max_stage
    else:
        forced_index = int(max(0, min(args_cli.force_stage, len(CURRICULUM_STAGES) - 1)))
        terrain_stage = CURRICULUM_STAGES[forced_index]
    env_cfg.scene.env_spacing = max(env_cfg.scene.env_spacing, 20.0)
    env_cfg.scene.terrain.max_init_terrain_level = terrain_rows - 1
    env_cfg.scene.terrain.terrain_generator = terrain_gen.TerrainGeneratorCfg(
        size=(30.0, 24.0),
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
                num_obstacles=int(terrain_stage["num_obstacles"]),
                radius=env_cfg.obstacle_radius,
                height=env_cfg.obstacle_height,
                x_range=(-10.5, 10.5),
                y_range=(-8.5, 8.5),
                layout_variant=terrain_stage["layout"],
            ),
        },
    )
    env_cfg.min_obstacle_distance = 1.7
    env_cfg.num_obstacles = int(terrain_stage["num_obstacles"])
    env_cfg.obstacle_layout_mode = "clutter"
    env_cfg.obstacle_layout_variant = terrain_stage["layout"]
    env_cfg.obstacle_x_range = (-10.5, 10.5)
    env_cfg.obstacle_y_range = (-8.5, 8.5)
    env_cfg.obstacle_min_gap = 1.35
    env_cfg.obstacle_robot_clearance = 2.4
    env_cfg.obstacle_collision_mode = "terrain_mesh"
    env_cfg.lidar_max_distance = args_cli.lidar_max_distance
    env_cfg.compression_fov_deg = args_cli.compression_fov_deg
    env_cfg.roi_x_min = args_cli.roi_x_min
    env_cfg.roi_x_max = args_cli.roi_x_max
    env_cfg.roi_abs_y_max = args_cli.roi_abs_y_max
    env_cfg.roi_z_min = args_cli.roi_z_min
    env_cfg.roi_z_max = args_cli.roi_z_max
    env_cfg.min_planar_distance = args_cli.min_planar_distance
    env_cfg.robot_body_radius = args_cli.robot_body_radius

    env_cfg.rewards.alive.weight = 0.15
    env_cfg.rewards.base_height.weight = -8.0
    env_cfg.rewards.flat_orientation_l2.weight = -3.5
    env_cfg.rewards.action_rate.weight = -0.03
    env_cfg.rewards.joint_vel.weight = -0.0008
    env_cfg.rewards.energy.weight = -1.5e-5
    env_cfg.rewards.joint_deviation_waists.weight = -1.2
    env_cfg.rewards.joint_deviation_legs.weight = -1.0
    env_cfg.rewards.gait.weight = 0.40
    env_cfg.rewards.feet_slide.weight = -0.08
    env_cfg.rewards.feet_clearance.weight = 1.0
    env_cfg.rewards.undesired_contacts.weight = -0.5


def _set_realg1_pointcloud_observations(env_cfg):
    from unitree_rl_lab.tasks.locomotion import mdp

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


def _apply_stage(env, stage_index: int) -> dict[str, float]:
    stage = CURRICULUM_STAGES[stage_index]
    cfg = env.unwrapped.cfg

    cfg.num_obstacles = int(stage["num_obstacles"])
    cfg.obstacle_layout_variant = stage["layout"]
    cfg.commands.base_velocity.rel_standing_envs = float(stage["standing"])
    cfg.commands.base_velocity.ranges.lin_vel_x = tuple(stage["lin_vel_x"])
    cfg.commands.base_velocity.ranges.lin_vel_y = (-float(stage["lin_vel_y"]), float(stage["lin_vel_y"]))
    cfg.commands.base_velocity.ranges.ang_vel_z = (-float(stage["ang_vel_z"]), float(stage["ang_vel_z"]))

    cfg.rewards.track_lin_vel_xy.weight = float(stage["track_lin"])
    cfg.rewards.track_ang_vel_z.weight = float(stage["track_ang"])
    cfg.rewards.proxemic_comfort.weight = float(stage["proxemic"])
    cfg.rewards.safe_approach_velocity.weight = float(stage["safe_vel"])
    cfg.rewards.safe_approach_acceleration.weight = float(stage["safe_acc"])
    cfg.rewards.tangential_avoidance.weight = float(stage["tangential"])
    cfg.rewards.joint_deviation_arms.weight = float(stage["arm_dev"])
    if "alive" in stage:
        cfg.rewards.alive.weight = float(stage["alive"])

    env.unwrapped.obstacle_manager.set_active_obstacles(int(stage["num_obstacles"]))
    return stage


def _stage_index_for_iteration(iteration: int, final_iteration: int) -> int:
    if final_iteration <= 1:
        return len(CURRICULUM_STAGES) - 1
    progress = float(iteration) / float(final_iteration - 1)
    stage_index = 0
    for idx, stage in enumerate(CURRICULUM_STAGES):
        if progress >= float(stage["progress"]):
            stage_index = idx
    return stage_index


def main():
    import unitree_rl_lab.tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from unitree_rl_lab.tasks.locomotion.obstacle_manager import ObstacleManager

    task_name = "Unitree-G1-29dof-ObstacleAvoidance-realG1"
    env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.safety_margin = args_cli.safety_margin
    env_cfg.cbf_gamma = args_cli.cbf_gamma
    env_cfg.collision_distance = args_cli.collision_distance

    _configure_base_env(env_cfg)
    _set_realg1_pointcloud_observations(env_cfg)

    env = gym.make(task_name, cfg=env_cfg)
    env.unwrapped.obstacle_manager = ObstacleManager(env.unwrapped, env.unwrapped.cfg)
    env.unwrapped.obstacle_manager.spawn_obstacles()
    _apply_stage(env, 0)

    device = args_cli.device
    num_envs = env.unwrapped.num_envs
    num_obs = env.unwrapped.observation_space["policy"].shape[-1]
    num_privileged_obs = env.unwrapped.observation_space["critic"].shape[-1]
    num_actions = env.unwrapped.action_space.shape[-1]

    actor_term_specs = build_actor_term_specs(args_cli.history_length, args_cli.num_fps_points)
    critic_term_specs = build_critic_term_specs(args_cli.history_length, args_cli.num_fps_points)

    policy = ActorCriticSafePerception(
        actor_term_specs=actor_term_specs,
        critic_term_specs=critic_term_specs,
        num_actions=num_actions,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
        activation="elu",
        init_noise_std=1.0,
        proprio_hidden_dim=128,
        scan_hidden_dim=64,
        rnn_hidden_dim=64,
    ).to(device)

    alg = P3OCBFPaper(
        actor_critic=policy,
        num_learning_epochs=args_cli.num_learning_epochs,
        num_mini_batches=args_cli.num_mini_batches,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=args_cli.entropy_coef,
        learning_rate=args_cli.learning_rate,
        cost_critic_learning_rate=args_cli.cost_critic_learning_rate,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="adaptive",
        desired_kl=0.01,
        device=device,
        cost_gamma=args_cli.cost_gamma,
        cost_lam=args_cli.cost_lam,
        cost_limit=args_cli.cost_limit,
        kappa=args_cli.kappa,
    )

    start_iteration = 0
    if args_cli.resume:
        checkpoint = torch.load(args_cli.resume, map_location=device, weights_only=False)
        policy.load_state_dict(checkpoint["policy_state_dict"], strict=True)
        if not args_cli.reset_optimizers and "optimizer_state_dict" in checkpoint:
            alg.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if not args_cli.reset_optimizers and "cost_optimizer_state_dict" in checkpoint:
            alg.cost_value_optimizer.load_state_dict(checkpoint["cost_optimizer_state_dict"])
        for group in alg.optimizer.param_groups:
            group["lr"] = args_cli.learning_rate
        for group in alg.cost_value_optimizer.param_groups:
            group["lr"] = args_cli.cost_critic_learning_rate
        start_iteration = int(checkpoint.get("iteration", 0))

    log_dir = os.path.join("logs", args_cli.experiment_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir, flush_secs=10)

    num_steps_per_env = args_cli.num_steps_per_env
    alg.init_storage(
        num_envs=num_envs,
        num_transitions_per_env=num_steps_per_env,
        actor_obs_shape=[num_obs],
        critic_obs_shape=[num_privileged_obs],
        action_shape=[num_actions],
    )

    reset_result = env.reset()
    obs_dict = reset_result[0] if isinstance(reset_result, tuple) else reset_result

    final_iteration = args_cli.max_iterations if args_cli.resume else start_iteration + args_cli.max_iterations
    last_stage_index = None
    for iteration in range(start_iteration, final_iteration):
        if args_cli.force_stage is None:
            stage_index = _stage_index_for_iteration(iteration, final_iteration)
        else:
            stage_index = int(max(0, min(args_cli.force_stage, len(CURRICULUM_STAGES) - 1)))
        if stage_index != last_stage_index:
            stage = _apply_stage(env, stage_index)
            last_stage_index = stage_index
        else:
            stage = CURRICULUM_STAGES[stage_index]

        episode_reward = torch.zeros(num_envs, device=device)
        episode_cost = torch.zeros(num_envs, device=device)
        paper_info_acc = {}
        tracking_info_acc = {}

        for _ in range(num_steps_per_env):
            obs = obs_dict["policy"].to(device)
            critic_obs = obs_dict["critic"].to(device)
            with torch.no_grad():
                actions = alg.act(obs, critic_obs)

            step_result = env.step(actions)
            if len(step_result) == 5:
                obs_dict, rewards, terminated, truncated, infos = step_result
                dones = torch.logical_or(terminated, truncated).to(device)
            else:
                obs_dict, rewards, dones, infos = step_result
                dones = dones.to(device)

            costs, paper_info = compute_paper_cost(env.unwrapped, args_cli)
            track_info = compute_tracking_metrics(env.unwrapped)

            episode_reward += rewards.to(device)
            episode_cost += costs
            for key, value in paper_info.items():
                paper_info_acc.setdefault(key, 0.0)
                paper_info_acc[key] += value
            for key, value in track_info.items():
                tracking_info_acc.setdefault(key, 0.0)
                tracking_info_acc[key] += value
            alg.process_env_step(rewards.to(device), costs, dones, infos)

        with torch.no_grad():
            last_critic_obs = obs_dict["critic"].to(device)
            alg.compute_returns(last_critic_obs)

        losses = alg.update()
        reward_mean = episode_reward.mean().item() / num_steps_per_env
        cost_mean = episode_cost.mean().item() / num_steps_per_env

        writer.add_scalar("train/reward", reward_mean, iteration)
        writer.add_scalar("train/cost", cost_mean, iteration)
        writer.add_scalar("curriculum/stage_index", stage_index, iteration)
        writer.add_scalar("curriculum/active_obstacles", stage["num_obstacles"], iteration)
        writer.add_scalar("curriculum/cmd_lin_vel_x_min", stage["lin_vel_x"][0], iteration)
        writer.add_scalar("curriculum/cmd_lin_vel_x_max", stage["lin_vel_x"][1], iteration)
        writer.add_scalar("curriculum/max_lateral_speed", stage["lin_vel_y"], iteration)
        writer.add_scalar("curriculum/max_yaw_rate", stage["ang_vel_z"], iteration)
        writer.add_scalar("curriculum/standing_ratio", stage["standing"], iteration)

        for key, value in losses.items():
            writer.add_scalar(f"loss/{key}", value, iteration)
        for key, value in paper_info_acc.items():
            writer.add_scalar(key, value / num_steps_per_env, iteration)
        for key, value in tracking_info_acc.items():
            writer.add_scalar(key, value / num_steps_per_env, iteration)

        if (iteration + 1) % 20 == 0 or iteration == 0:
            print(
                f"[ITER {iteration + 1:05d}] "
                f"stage={stage['name']} reward={reward_mean:.4f} cost={cost_mean:.4f} "
                f"unsafe={paper_info_acc.get('paper/unsafe_rate', 0.0) / num_steps_per_env:.4f} "
                f"collision={paper_info_acc.get('paper/collision_rate', 0.0) / num_steps_per_env:.4f} "
                f"contact={paper_info_acc.get('paper/contact_rate', 0.0) / num_steps_per_env:.4f}"
            )

        if (iteration + 1) % args_cli.save_interval == 0:
            checkpoint = {
                "iteration": iteration + 1,
                "policy_state_dict": policy.state_dict(),
                "optimizer_state_dict": alg.optimizer.state_dict(),
                "cost_optimizer_state_dict": alg.cost_value_optimizer.state_dict(),
                "args": vars(args_cli),
            }
            torch.save(checkpoint, os.path.join(log_dir, f"model_{iteration + 1}.pt"))

    final_checkpoint = {
        "iteration": max(start_iteration, final_iteration),
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": alg.optimizer.state_dict(),
        "cost_optimizer_state_dict": alg.cost_value_optimizer.state_dict(),
        "args": vars(args_cli),
    }
    torch.save(final_checkpoint, os.path.join(log_dir, "model_final.pt"))
    writer.close()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
