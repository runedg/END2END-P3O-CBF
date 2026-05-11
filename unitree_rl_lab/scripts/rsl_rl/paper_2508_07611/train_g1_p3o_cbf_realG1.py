# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deployment-oriented G1 P3O-CBF training with Mid360 pointcloud -> 64D compression."""

import argparse
import importlib
import math
import os
from datetime import datetime

import torch

from isaaclab.app import AppLauncher

from actor_critic_safe_perception import ActorCriticSafePerception, ObsTermSpec

parser = argparse.ArgumentParser(description="Train a deployment-oriented realG1 G1 P3O-CBF policy.")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments.")
parser.add_argument("--max_iterations", type=int, default=15000, help="Training iterations.")
parser.add_argument("--save_interval", type=int, default=2000, help="Save checkpoint every N iterations.")
parser.add_argument("--experiment_name", type=str, default="p3o_g1_cbf_realG1_widespeed", help="Top-level log folder name.")
parser.add_argument("--headless", action="store_true", default=False, help="Run in headless mode.")
parser.add_argument("--device", type=str, default="cuda:0", help="Device to use.")
parser.add_argument("--stage", type=str, default="clutter", choices=["walk", "avoid", "clutter"], help="Training stage.")
parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint to resume from.")
parser.add_argument("--cost_limit", type=float, default=0.22, help="Rollout cost limit for the paper-style P3O term.")
parser.add_argument("--kappa", type=float, default=1.2, help="Penalty coefficient.")
parser.add_argument("--cost_gamma", type=float, default=0.99, help="Cost discount factor.")
parser.add_argument("--cost_lam", type=float, default=0.95, help="Cost GAE lambda.")
parser.add_argument("--safety_margin", type=float, default=0.8, help="Safe distance to obstacles.")
parser.add_argument("--cbf_gamma", type=float, default=0.5, help="Discrete CBF coefficient.")
parser.add_argument("--collision_distance", type=float, default=0.2, help="Distance counted as collision.")
parser.add_argument("--unsafe_cost_weight", type=float, default=1.25, help="Weight for entering the unsafe set.")
parser.add_argument("--cbf_cost_weight", type=float, default=1.25, help="Weight for discrete CBF violation.")
parser.add_argument("--collision_cost_weight", type=float, default=3.0, help="Extra weight for near-collision states.")
parser.add_argument("--curriculum_max_obstacles", type=int, default=24, help="Maximum number of obstacles to reveal.")
parser.add_argument("--history_length", type=int, default=5, help="Observation history length.")
parser.add_argument("--max_forward_speed", type=float, default=3.0, help="Maximum commanded forward speed used by the wide-speed curriculum.")
parser.add_argument("--max_lateral_speed", type=float, default=0.35, help="Maximum commanded lateral speed used by the wide-speed curriculum.")
parser.add_argument("--max_yaw_rate", type=float, default=0.8, help="Maximum commanded yaw rate used by the wide-speed curriculum.")
parser.add_argument(
    "--clutter_layout_variant",
    type=str,
    default="front_dense_hybrid",
    choices=["front_dense_hybrid", "surround_hybrid", "dense_irregular", "end2end_distributed", "arena_walled"],
    help="Terrain layout used for the high-speed clutter stage.",
)
parser.add_argument("--lidar_feature_dim", type=int, default=64, help="Compressed Mid360 feature dimension.")
parser.add_argument("--lidar_max_distance", type=float, default=6.0, help="Maximum compressed LiDAR range.")
parser.add_argument("--compression_fov_deg", type=float, default=180.0, help="Horizontal FOV retained after compression.")
parser.add_argument(
    "--omni_pattern_file",
    type=str,
    default="/home/ubuntu/P3O-CBF/OmniPerception/LidarSensor/LidarSensor/sensor_pattern/sensor_lidar/scan_mode/mid360.npy",
    help="Path to OmniPerception Mid360 pattern file.",
)
parser.add_argument("--omni_point_samples", type=int, default=1024, help="Number of Mid360 points sampled per frame before compression.")
parser.add_argument("--roi_x_min", type=float, default=-0.5, help="Min x in base frame kept for compression.")
parser.add_argument("--roi_x_max", type=float, default=6.0, help="Max x in base frame kept for compression.")
parser.add_argument("--roi_abs_y_max", type=float, default=3.0, help="Absolute y limit in base frame kept for compression.")
parser.add_argument("--roi_z_min", type=float, default=-1.0, help="Min z in base frame kept for compression.")
parser.add_argument("--roi_z_max", type=float, default=0.8, help="Max z in base frame kept for compression.")
parser.add_argument("--min_planar_distance", type=float, default=0.2, help="Minimum planar range retained in compressed features.")
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


def _closest_obstacle_geometry(env):
    robot_xy = env.scene["robot"].data.root_pos_w[:, :2]
    return env.obstacle_manager.get_closest_geometry(robot_xy)


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
    total_cost = (
        cfg.unsafe_cost_weight * unsafe_cost
        + cfg.cbf_cost_weight * cbf_violation
        + cfg.collision_cost_weight * collision_cost
    )
    info = {
        "paper/distance_mean": distances.mean().item(),
        "paper/distance_min": distances.min().item(),
        "paper/h_mean": h.mean().item(),
        "paper/h_min": h.min().item(),
        "paper/h_dot_mean": h_dot.mean().item(),
        "paper/h_next_est_mean": h_next_est.mean().item(),
        "paper/cbf_margin_mean": cbf_margin.mean().item(),
        "paper/cbf_margin_min": cbf_margin.min().item(),
        "paper/cbf_violation_rate": (cbf_violation > 0).float().mean().item(),
        "paper/unsafe_rate": unsafe_cost.mean().item(),
        "paper/collision_rate": collision_cost.mean().item(),
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


def _set_gait_reward_profile(env_cfg, gait_weight: float, feet_slide_weight: float, undesired_contacts_weight: float):
    env_cfg.rewards.gait.weight = gait_weight
    env_cfg.rewards.feet_slide.weight = feet_slide_weight
    env_cfg.rewards.undesired_contacts.weight = undesired_contacts_weight


def configure_stage(env_cfg, stage: str):
    import isaaclab.terrains as terrain_gen

    obstacle_env_cfg_mod = importlib.import_module(
        "unitree_rl_lab.tasks.locomotion.robots.g1.29dof.obstacle_env_cfg"
    )
    MeshClutterPillarsTerrainCfg = obstacle_env_cfg_mod.MeshClutterPillarsTerrainCfg

    cmd_cfg = env_cfg.commands.base_velocity
    if stage == "walk":
        env_cfg.num_obstacles = 2
        env_cfg.min_obstacle_distance = 4.0
        env_cfg.obstacle_layout_mode = "radial"
        cmd_cfg.rel_standing_envs = 0.02
        cmd_cfg.ranges.lin_vel_x = (0.2, 0.6)
        cmd_cfg.ranges.lin_vel_y = (-0.08, 0.08)
        cmd_cfg.ranges.ang_vel_z = (-0.15, 0.15)
        env_cfg.rewards.track_lin_vel_xy.weight = 2.5
        env_cfg.rewards.track_ang_vel_z.weight = 1.2
        env_cfg.rewards.alive.weight = 0.15
        env_cfg.rewards.base_height.weight = -7.0
        env_cfg.rewards.flat_orientation_l2.weight = -3.0
        env_cfg.rewards.action_rate.weight = -0.03
        env_cfg.rewards.joint_vel.weight = -0.0005
        env_cfg.rewards.energy.weight = -1.0e-5
        env_cfg.rewards.feet_clearance.weight = 1.2
        _set_gait_reward_profile(env_cfg, gait_weight=0.50, feet_slide_weight=-0.10, undesired_contacts_weight=-0.5)
    elif stage == "avoid":
        env_cfg.num_obstacles = max(2, args_cli.curriculum_max_obstacles)
        env_cfg.min_obstacle_distance = 2.5
        env_cfg.obstacle_layout_mode = "radial"
        cmd_cfg.rel_standing_envs = 0.02
        cmd_cfg.ranges.lin_vel_x = (0.15, 0.8)
        cmd_cfg.ranges.lin_vel_y = (-0.15, 0.15)
        cmd_cfg.ranges.ang_vel_z = (-0.25, 0.25)
        env_cfg.rewards.track_lin_vel_xy.weight = 1.8
        env_cfg.rewards.track_ang_vel_z.weight = 0.8
        env_cfg.rewards.alive.weight = 0.15
        env_cfg.rewards.base_height.weight = -8.0
        env_cfg.rewards.flat_orientation_l2.weight = -4.0
        env_cfg.rewards.action_rate.weight = -0.04
        env_cfg.rewards.joint_vel.weight = -0.001
        env_cfg.rewards.energy.weight = -2.0e-5
        env_cfg.rewards.feet_clearance.weight = 1.0
        _set_gait_reward_profile(env_cfg, gait_weight=0.45, feet_slide_weight=-0.10, undesired_contacts_weight=-0.5)
    else:
        env_cfg.num_obstacles = max(24, args_cli.curriculum_max_obstacles)
        terrain_rows = int(math.ceil(math.sqrt(args_cli.num_envs)))
        terrain_cols = int(math.ceil(args_cli.num_envs / terrain_rows))
        env_cfg.scene.env_spacing = max(env_cfg.scene.env_spacing, 28.0)
        env_cfg.scene.terrain.max_init_terrain_level = terrain_rows - 1
        env_cfg.scene.terrain.terrain_generator = terrain_gen.TerrainGeneratorCfg(
            size=(52.0, 34.0),
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
                    x_range=(-20.0, 20.0) if args_cli.clutter_layout_variant == "arena_walled" else ((-3.8, 21.8) if args_cli.clutter_layout_variant == "end2end_distributed" else (-1.5, 18.0)),
                    y_range=(-12.0, 12.0) if args_cli.clutter_layout_variant == "arena_walled" else ((-7.8, 7.8) if args_cli.clutter_layout_variant == "end2end_distributed" else (-4.8, 4.8)),
                    layout_variant=args_cli.clutter_layout_variant,
                ),
            },
        )
        env_cfg.min_obstacle_distance = 1.75
        env_cfg.obstacle_layout_mode = "clutter"
        env_cfg.obstacle_layout_variant = args_cli.clutter_layout_variant
        if args_cli.clutter_layout_variant == "arena_walled":
            env_cfg.obstacle_x_range = (-20.0, 20.0)
            env_cfg.obstacle_y_range = (-12.0, 12.0)
            env_cfg.obstacle_min_gap = 1.20
            env_cfg.obstacle_robot_clearance = 2.20
        elif args_cli.clutter_layout_variant == "end2end_distributed":
            env_cfg.obstacle_x_range = (-3.8, 21.8)
            env_cfg.obstacle_y_range = (-7.8, 7.8)
            env_cfg.obstacle_min_gap = 1.10
            env_cfg.obstacle_robot_clearance = 1.65
        else:
            env_cfg.obstacle_x_range = (-1.5, 18.0)
            env_cfg.obstacle_y_range = (-4.8, 4.8)
            env_cfg.obstacle_min_gap = 0.90
            env_cfg.obstacle_robot_clearance = 1.80
        env_cfg.obstacle_collision_mode = "terrain_mesh"
        cmd_cfg.resampling_time_range = (4.0, 7.0)
        cmd_cfg.rel_standing_envs = 0.08
        cmd_cfg.ranges.lin_vel_x = (0.0, min(args_cli.max_forward_speed, 0.6))
        cmd_cfg.ranges.lin_vel_y = (-0.10, 0.10)
        cmd_cfg.ranges.ang_vel_z = (-0.35, 0.35)
        env_cfg.rewards.track_lin_vel_xy.weight = 2.40
        env_cfg.rewards.track_ang_vel_z.weight = 1.20
        env_cfg.rewards.alive.weight = 0.20
        env_cfg.rewards.base_height.weight = -7.5
        env_cfg.rewards.flat_orientation_l2.weight = -3.5
        env_cfg.rewards.action_rate.weight = -0.02
        env_cfg.rewards.joint_vel.weight = -0.0007
        env_cfg.rewards.energy.weight = -1.5e-5
        env_cfg.rewards.feet_clearance.weight = 1.1
        _set_gait_reward_profile(env_cfg, gait_weight=0.45, feet_slide_weight=-0.08, undesired_contacts_weight=-0.5)


def apply_command_curriculum(env, cfg, stage_progress: float):
    if cfg.stage != "clutter":
        return

    cmd_cfg = env.unwrapped.cfg.commands.base_velocity
    speed_caps = [
        min(args_cli.max_forward_speed, 0.8),
        min(args_cli.max_forward_speed, 1.5),
        min(args_cli.max_forward_speed, 2.2),
        args_cli.max_forward_speed,
    ]
    lateral_caps = [
        min(args_cli.max_lateral_speed, 0.10),
        min(args_cli.max_lateral_speed, 0.16),
        min(args_cli.max_lateral_speed, 0.24),
        args_cli.max_lateral_speed,
    ]
    yaw_caps = [
        min(args_cli.max_yaw_rate, 0.30),
        min(args_cli.max_yaw_rate, 0.45),
        min(args_cli.max_yaw_rate, 0.60),
        args_cli.max_yaw_rate,
    ]
    if stage_progress < 0.20:
        phase_idx = 0
        cmd_cfg.rel_standing_envs = 0.10
    elif stage_progress < 0.45:
        phase_idx = 1
        cmd_cfg.rel_standing_envs = 0.07
    elif stage_progress < 0.70:
        phase_idx = 2
        cmd_cfg.rel_standing_envs = 0.05
    else:
        phase_idx = 3
        cmd_cfg.rel_standing_envs = 0.03
    cmd_cfg.ranges.lin_vel_x = (0.0, max(0.3, speed_caps[phase_idx]))
    cmd_cfg.ranges.lin_vel_y = (-lateral_caps[phase_idx], lateral_caps[phase_idx])
    cmd_cfg.ranges.ang_vel_z = (-yaw_caps[phase_idx], yaw_caps[phase_idx])


def get_stage_progress(iteration: int, max_iterations: int, stage: str) -> float:
    if max_iterations <= 1:
        return 1.0
    raw_progress = iteration / float(max_iterations - 1)
    return min(1.0, raw_progress / 0.6) if stage == "walk" else raw_progress


def apply_p3o_schedule(alg: P3OCBFPaper, cfg, stage_progress: float):
    if cfg.stage == "walk":
        alg.kappa = max(0.2, cfg.kappa * (0.3 + 0.5 * stage_progress))
        alg.cost_limit = max(0.15, cfg.cost_limit * (1.8 - 0.8 * stage_progress))
    elif cfg.stage == "clutter":
        alg.kappa = max(0.8, cfg.kappa * (0.8 + 0.8 * stage_progress))
        alg.cost_limit = max(0.04, cfg.cost_limit * (1.05 - 0.55 * stage_progress))
    else:
        alg.kappa = max(0.5, cfg.kappa * (0.6 + 0.8 * stage_progress))
        alg.cost_limit = max(0.05, cfg.cost_limit * (1.2 - 0.7 * stage_progress))


def apply_obstacle_curriculum(obstacle_manager, cfg, stage_progress: float) -> int:
    max_active = min(cfg.curriculum_max_obstacles, obstacle_manager.max_num_obstacles)
    if cfg.stage == "walk":
        active = 1 if stage_progress < 0.55 else min(2, max_active)
    elif cfg.stage == "clutter":
        if max_active <= 8:
            active = max_active
        elif stage_progress < 0.20:
            active = min(8, max_active)
        elif stage_progress < 0.45:
            active = min(12, max_active)
        elif stage_progress < 0.70:
            active = min(18, max_active)
        else:
            active = max_active
    elif max_active <= 2:
        active = max_active
    elif stage_progress < 0.30:
        active = 2
    elif stage_progress < 0.65:
        active = min(3, max_active)
    else:
        active = max_active
    obstacle_manager.set_active_obstacles(active)
    return active


def _set_realg1_observations(env_cfg):
    from unitree_rl_lab.tasks.locomotion import mdp

    params = {
        "sensor_name": "mid360_lidar",
        "feature_dim": args_cli.lidar_feature_dim,
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
    scale = tuple([1.0 / args_cli.lidar_max_distance] * args_cli.lidar_feature_dim)

    env_cfg.scene.mid360_lidar.max_distance = args_cli.lidar_max_distance
    env_cfg.scene.mid360_lidar.offset.pos = (
        args_cli.sensor_offset_x,
        args_cli.sensor_offset_y,
        args_cli.sensor_offset_z,
    )
    env_cfg.scene.mid360_lidar.pattern_cfg.pattern_file = args_cli.omni_pattern_file
    env_cfg.scene.mid360_lidar.pattern_cfg.samples = args_cli.omni_point_samples

    env_cfg.observations.policy.obstacle_scan = ObsTerm(
        func=mdp.mid360_realg1_range_features_from_raycaster,
        params=params,
        clip=(0.0, args_cli.lidar_max_distance),
        scale=scale,
    )
    env_cfg.observations.policy.history_length = args_cli.history_length

    env_cfg.observations.critic.obstacle_scan = ObsTerm(
        func=mdp.mid360_realg1_range_features_from_raycaster,
        params=params,
        clip=(0.0, args_cli.lidar_max_distance),
        scale=scale,
    )
    env_cfg.observations.critic.history_length = args_cli.history_length


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
    configure_stage(env_cfg, args_cli.stage)
    _set_realg1_observations(env_cfg)

    env = gym.make(task_name, cfg=env_cfg)
    env.unwrapped.obstacle_manager = ObstacleManager(env.unwrapped, env.unwrapped.cfg)
    env.unwrapped.obstacle_manager.spawn_obstacles()

    device = args_cli.device
    num_envs = env.unwrapped.num_envs
    num_obs = env.unwrapped.observation_space["policy"].shape[-1]
    num_privileged_obs = env.unwrapped.observation_space["critic"].shape[-1]
    num_actions = env.unwrapped.action_space.shape[-1]

    actor_term_specs = build_actor_term_specs(args_cli.history_length, args_cli.lidar_feature_dim)
    critic_term_specs = build_critic_term_specs(args_cli.history_length, args_cli.lidar_feature_dim)

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

    if policy.actor_encoder.input_dim != num_obs:
        raise RuntimeError(f"Actor obs dim mismatch: encoder expects {policy.actor_encoder.input_dim}, env provides {num_obs}")
    if policy.critic_encoder.input_dim != num_privileged_obs:
        raise RuntimeError(
            f"Critic obs dim mismatch: encoder expects {policy.critic_encoder.input_dim}, env provides {num_privileged_obs}"
        )

    alg = P3OCBFPaper(
        actor_critic=policy,
        num_learning_epochs=5,
        num_mini_batches=4,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.01,
        learning_rate=1e-3,
        cost_critic_learning_rate=1e-4,
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
        if "optimizer_state_dict" in checkpoint:
            alg.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "cost_optimizer_state_dict" in checkpoint:
            alg.cost_value_optimizer.load_state_dict(checkpoint["cost_optimizer_state_dict"])
        start_iteration = int(checkpoint.get("iteration", 0))
        print(f"[INFO] Resumed checkpoint: {args_cli.resume} (iteration={start_iteration})")

    log_dir = os.path.join("logs", args_cli.experiment_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir, flush_secs=10)
    print(f"[INFO] Logging to: {log_dir}")

    num_steps_per_env = 24
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
    for iteration in range(start_iteration, final_iteration):
        local_iteration = iteration - start_iteration
        stage_progress = get_stage_progress(local_iteration, max(args_cli.max_iterations, 1), args_cli.stage)
        apply_p3o_schedule(alg, args_cli, stage_progress)
        apply_command_curriculum(env, args_cli, stage_progress)
        active_obstacles = apply_obstacle_curriculum(env.unwrapped.obstacle_manager, args_cli, stage_progress)

        episode_reward = torch.zeros(num_envs, device=device)
        episode_cost = torch.zeros(num_envs, device=device)
        done_count = 0.0
        base_height_done_count = 0.0
        bad_orientation_done_count = 0.0
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

            done_count += dones.float().mean().item()
            time_outs = infos.get("time_outs", torch.zeros_like(dones, dtype=torch.bool))
            reset_terminated = dones & ~time_outs.to(device).bool()
            base_height_done_count += reset_terminated.float().mean().item()
            robot = env.unwrapped.scene["robot"]
            gravity_xy_norm = torch.linalg.norm(robot.data.projected_gravity_b[:, :2], dim=-1)
            bad_orientation_done_count += ((gravity_xy_norm > 0.8) & reset_terminated).float().mean().item()

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
        writer.add_scalar("train/done_rate", done_count / num_steps_per_env, iteration)
        writer.add_scalar("train/fall_like_rate", base_height_done_count / num_steps_per_env, iteration)
        writer.add_scalar("train/bad_orientation_like_rate", bad_orientation_done_count / num_steps_per_env, iteration)
        writer.add_scalar("curriculum/active_obstacles", active_obstacles, iteration)
        writer.add_scalar("curriculum/stage_progress", stage_progress, iteration)
        writer.add_scalar("curriculum/cmd_rel_standing_envs", env.unwrapped.cfg.commands.base_velocity.rel_standing_envs, iteration)
        writer.add_scalar("curriculum/cmd_lin_vel_x_max", env.unwrapped.cfg.commands.base_velocity.ranges.lin_vel_x[1], iteration)
        writer.add_scalar("curriculum/cmd_lin_vel_y_abs_max", abs(env.unwrapped.cfg.commands.base_velocity.ranges.lin_vel_y[1]), iteration)
        writer.add_scalar("curriculum/cmd_ang_vel_z_abs_max", abs(env.unwrapped.cfg.commands.base_velocity.ranges.ang_vel_z[1]), iteration)
        writer.add_scalar("perception/omni_point_samples", args_cli.omni_point_samples, iteration)
        writer.add_scalar("perception/lidar_feature_dim", args_cli.lidar_feature_dim, iteration)
        writer.add_scalar("perception/history_length", args_cli.history_length, iteration)
        writer.add_scalar("perception/compression_fov_deg", args_cli.compression_fov_deg, iteration)
        writer.add_scalar("perception/pixel_dropout_prob", args_cli.pixel_dropout_prob, iteration)
        writer.add_scalar("perception/sector_dropout_prob", args_cli.sector_dropout_prob, iteration)

        for key, value in losses.items():
            writer.add_scalar(f"loss/{key}", value, iteration)
        for key, value in paper_info_acc.items():
            writer.add_scalar(key, value / num_steps_per_env, iteration)
        for key, value in tracking_info_acc.items():
            writer.add_scalar(key, value / num_steps_per_env, iteration)

        if (iteration + 1) % 20 == 0 or iteration == 0:
            print(
                f"[ITER {iteration + 1:05d}] "
                f"reward={reward_mean:.4f} cost={cost_mean:.4f} "
                f"unsafe={paper_info_acc.get('paper/unsafe_rate', 0.0) / num_steps_per_env:.4f} "
                f"cbf={paper_info_acc.get('paper/cbf_violation_rate', 0.0) / num_steps_per_env:.4f} "
                f"active_obs={active_obstacles}"
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
