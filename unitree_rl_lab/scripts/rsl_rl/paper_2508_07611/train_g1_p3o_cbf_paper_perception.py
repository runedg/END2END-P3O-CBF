# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Perception-enhanced paper-aligned G1 P3O-CBF training variant.

This keeps the paper-style CMDP/P3O-CBF training loop, but upgrades the
perception stack from a tiny flat obstacle scan to a denser spatio-temporal
LiDAR-like representation with an internal temporal encoder.
"""

import argparse
import importlib
import math
import os
from datetime import datetime

import torch

from isaaclab.app import AppLauncher

from actor_critic_safe_perception import ActorCriticSafePerception, ObsTermSpec

parser = argparse.ArgumentParser(description="Train a perception-enhanced G1 paper-style P3O-CBF policy.")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments.")
parser.add_argument("--max_iterations", type=int, default=15000, help="Training iterations.")
parser.add_argument("--save_interval", type=int, default=2000, help="Save checkpoint every N iterations.")
parser.add_argument("--headless", action="store_true", default=False, help="Run in headless mode.")
parser.add_argument("--device", type=str, default="cuda:0", help="Device to use.")
parser.add_argument("--stage", type=str, default="avoid", choices=["walk", "avoid", "clutter"], help="Training stage.")
parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint to resume from.")
parser.add_argument("--cost_limit", type=float, default=0.3, help="Rollout cost limit for the paper-style P3O term.")
parser.add_argument("--kappa", type=float, default=1.0, help="Penalty coefficient.")
parser.add_argument("--cost_gamma", type=float, default=0.99, help="Cost discount factor.")
parser.add_argument("--cost_lam", type=float, default=0.95, help="Cost GAE lambda.")
parser.add_argument("--safety_margin", type=float, default=0.8, help="Safe distance to obstacles.")
parser.add_argument("--cbf_gamma", type=float, default=0.5, help="Discrete CBF coefficient.")
parser.add_argument("--collision_distance", type=float, default=0.2, help="Distance counted as collision.")
parser.add_argument("--unsafe_cost_weight", type=float, default=1.0, help="Weight for entering the unsafe set.")
parser.add_argument("--cbf_cost_weight", type=float, default=1.0, help="Weight for discrete CBF violation.")
parser.add_argument("--collision_cost_weight", type=float, default=2.0, help="Extra weight for near-collision states.")
parser.add_argument("--curriculum_max_obstacles", type=int, default=4, help="Maximum number of obstacles to reveal.")
parser.add_argument("--lidar_num_rays", type=int, default=64, help="Number of LiDAR rays in the dense scan.")
parser.add_argument("--lidar_fov_deg", type=float, default=180.0, help="Horizontal field-of-view of the LiDAR scan.")
parser.add_argument("--lidar_max_distance", type=float, default=6.0, help="Maximum LiDAR range.")
parser.add_argument("--history_length", type=int, default=5, help="Observation history length.")
parser.add_argument(
    "--lidar_mode",
    type=str,
    default="latent64",
    choices=["latent64", "raw_pointcloud", "dense_scan"],
    help="LiDAR input mode. latent64 pools Mid360 rays into a compact paper-style 64-D feature.",
)
parser.add_argument("--use_omni_pattern", action="store_true", default=True, help="Deprecated alias for raw_pointcloud.")
parser.add_argument(
    "--omni_pattern_file",
    type=str,
    default="/home/ubuntu/P3O-CBF/OmniPerception/LidarSensor/LidarSensor/sensor_pattern/sensor_lidar/scan_mode/mid360.npy",
    help="Path to OmniPerception Mid360 pattern file.",
)
parser.add_argument("--omni_point_samples", type=int, default=512, help="Number of sampled LiDAR points per frame.")
parser.add_argument("--lidar_latent_dim", type=int, default=64, help="Compact LiDAR feature dimension.")

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
    else:
        env_cfg.num_obstacles = max(22, args_cli.curriculum_max_obstacles)
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
                    x_range=(-4.5, 19.0),
                    y_range=(-9.0, 9.0),
                    layout_variant="surround_hybrid",
                ),
            },
        )
        env_cfg.min_obstacle_distance = 1.45
        env_cfg.obstacle_layout_mode = "clutter"
        env_cfg.obstacle_layout_variant = "surround_hybrid"
        env_cfg.obstacle_x_range = (-4.5, 19.0)
        env_cfg.obstacle_y_range = (-9.0, 9.0)
        env_cfg.obstacle_min_gap = 0.85
        env_cfg.obstacle_robot_clearance = 1.6
        env_cfg.obstacle_collision_mode = "terrain_mesh"
        cmd_cfg.resampling_time_range = (4.0, 7.0)
        cmd_cfg.rel_standing_envs = 0.18
        cmd_cfg.ranges.lin_vel_x = (0.0, 0.35)
        cmd_cfg.ranges.lin_vel_y = (-0.08, 0.08)
        cmd_cfg.ranges.ang_vel_z = (-0.30, 0.30)
        env_cfg.rewards.track_lin_vel_xy.weight = 1.55
        env_cfg.rewards.track_ang_vel_z.weight = 1.0
        env_cfg.rewards.alive.weight = 0.15
        env_cfg.rewards.base_height.weight = -8.0
        env_cfg.rewards.flat_orientation_l2.weight = -4.0
        env_cfg.rewards.action_rate.weight = -0.04
        env_cfg.rewards.joint_vel.weight = -0.001
        env_cfg.rewards.energy.weight = -2.0e-5
        env_cfg.rewards.feet_clearance.weight = 1.0


def apply_command_curriculum(env, cfg, stage_progress: float):
    if cfg.stage != "clutter":
        return

    cmd_cfg = env.unwrapped.cfg.commands.base_velocity
    if stage_progress < 0.25:
        cmd_cfg.rel_standing_envs = 0.18
        cmd_cfg.ranges.lin_vel_x = (0.0, 0.35)
        cmd_cfg.ranges.lin_vel_y = (-0.08, 0.08)
        cmd_cfg.ranges.ang_vel_z = (-0.30, 0.30)
    elif stage_progress < 0.60:
        cmd_cfg.rel_standing_envs = 0.14
        cmd_cfg.ranges.lin_vel_x = (0.0, 0.45)
        cmd_cfg.ranges.lin_vel_y = (-0.12, 0.12)
        cmd_cfg.ranges.ang_vel_z = (-0.45, 0.45)
    else:
        cmd_cfg.rel_standing_envs = 0.10
        cmd_cfg.ranges.lin_vel_x = (0.0, 0.60)
        cmd_cfg.ranges.lin_vel_y = (-0.18, 0.18)
        cmd_cfg.ranges.ang_vel_z = (-0.65, 0.65)


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
        alg.kappa = max(0.5, cfg.kappa * (0.5 + 0.9 * stage_progress))
        alg.cost_limit = max(0.05, cfg.cost_limit * (1.25 - 0.75 * stage_progress))
    else:
        alg.kappa = max(0.5, cfg.kappa * (0.6 + 0.8 * stage_progress))
        alg.cost_limit = max(0.05, cfg.cost_limit * (1.2 - 0.7 * stage_progress))


def apply_obstacle_curriculum(obstacle_manager, cfg, stage_progress: float) -> int:
    max_active = min(cfg.curriculum_max_obstacles, obstacle_manager.max_num_obstacles)
    if cfg.stage == "walk":
        active = 1 if stage_progress < 0.55 else min(2, max_active)
    elif cfg.stage == "clutter":
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


def _set_dense_lidar_observations(env_cfg, num_rays: int, max_distance: float, fov_deg: float, history_length: int):
    params = {"num_rays": num_rays, "max_distance": max_distance, "fov_deg": fov_deg}

    env_cfg.observations.policy.obstacle_scan.params = params
    env_cfg.observations.policy.obstacle_scan.clip = (0.0, max_distance)
    env_cfg.observations.policy.history_length = history_length

    env_cfg.observations.critic.obstacle_scan.params = params
    env_cfg.observations.critic.obstacle_scan.clip = (0.0, max_distance)
    env_cfg.observations.critic.history_length = history_length


def _set_omni_pointcloud_observations(env_cfg, pattern_file: str, samples: int, max_distance: float, history_length: int):
    from unitree_rl_lab.tasks.locomotion import mdp

    point_dim = samples * 3
    params = {"pattern_file": pattern_file, "samples": samples, "max_distance": max_distance}
    scale = tuple([1.0 / max_distance] * point_dim)

    env_cfg.observations.policy.obstacle_scan = ObsTerm(
        func=mdp.omni_lidar_pointcloud,
        params=params,
        clip=(-max_distance, max_distance),
        scale=scale,
    )
    env_cfg.observations.policy.history_length = history_length

    env_cfg.observations.critic.obstacle_scan = ObsTerm(
        func=mdp.omni_lidar_pointcloud,
        params=params,
        clip=(-max_distance, max_distance),
        scale=scale,
    )
    env_cfg.observations.critic.history_length = history_length


def _set_omni_lidar_feature_observations(
    env_cfg,
    pattern_file: str,
    samples: int,
    feature_dim: int,
    max_distance: float,
    history_length: int,
):
    from unitree_rl_lab.tasks.locomotion import mdp

    params = {
        "pattern_file": pattern_file,
        "samples": samples,
        "feature_dim": feature_dim,
        "max_distance": max_distance,
    }
    scale = tuple([1.0 / max_distance] * feature_dim)

    env_cfg.observations.policy.obstacle_scan = ObsTerm(
        func=mdp.omni_lidar_range_features,
        params=params,
        clip=(0.0, max_distance),
        scale=scale,
    )
    env_cfg.observations.policy.history_length = history_length

    env_cfg.observations.critic.obstacle_scan = ObsTerm(
        func=mdp.omni_lidar_range_features,
        params=params,
        clip=(0.0, max_distance),
        scale=scale,
    )
    env_cfg.observations.critic.history_length = history_length


def build_actor_term_specs(history_length: int, num_rays: int) -> list[ObsTermSpec]:
    return [
        ObsTermSpec("base_ang_vel", 3, history_length),
        ObsTermSpec("projected_gravity", 3, history_length),
        ObsTermSpec("velocity_commands", 3, history_length),
        ObsTermSpec("obstacle_scan", num_rays, history_length),
        ObsTermSpec("joint_pos_rel", 29, history_length),
        ObsTermSpec("joint_vel_rel", 29, history_length),
        ObsTermSpec("last_action", 29, history_length),
    ]


def build_actor_pointcloud_term_specs(history_length: int, num_points: int) -> list[ObsTermSpec]:
    return [
        ObsTermSpec("base_ang_vel", 3, history_length),
        ObsTermSpec("projected_gravity", 3, history_length),
        ObsTermSpec("velocity_commands", 3, history_length),
        ObsTermSpec("lidar_points", num_points * 3, history_length),
        ObsTermSpec("joint_pos_rel", 29, history_length),
        ObsTermSpec("joint_vel_rel", 29, history_length),
        ObsTermSpec("last_action", 29, history_length),
    ]


def build_critic_term_specs(history_length: int, num_rays: int) -> list[ObsTermSpec]:
    return [
        ObsTermSpec("base_lin_vel", 3, history_length),
        ObsTermSpec("base_ang_vel", 3, history_length),
        ObsTermSpec("projected_gravity", 3, history_length),
        ObsTermSpec("velocity_commands", 3, history_length),
        ObsTermSpec("obstacle_scan", num_rays, history_length),
        ObsTermSpec("joint_pos_rel", 29, history_length),
        ObsTermSpec("joint_vel_rel", 29, history_length),
        ObsTermSpec("last_action", 29, history_length),
    ]


def build_critic_pointcloud_term_specs(history_length: int, num_points: int) -> list[ObsTermSpec]:
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


def main():
    import unitree_rl_lab.tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from unitree_rl_lab.tasks.locomotion.obstacle_manager import ObstacleManager

    task_name = "Unitree-G1-29dof-ObstacleAvoidance"
    env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.safety_margin = args_cli.safety_margin
    env_cfg.cbf_gamma = args_cli.cbf_gamma
    env_cfg.collision_distance = args_cli.collision_distance
    configure_stage(env_cfg, args_cli.stage)
    if args_cli.lidar_mode == "latent64":
        _set_omni_lidar_feature_observations(
            env_cfg,
            pattern_file=args_cli.omni_pattern_file,
            samples=args_cli.omni_point_samples,
            feature_dim=args_cli.lidar_latent_dim,
            max_distance=args_cli.lidar_max_distance,
            history_length=args_cli.history_length,
        )
    elif args_cli.lidar_mode == "raw_pointcloud":
        _set_omni_pointcloud_observations(
            env_cfg,
            pattern_file=args_cli.omni_pattern_file,
            samples=args_cli.omni_point_samples,
            max_distance=args_cli.lidar_max_distance,
            history_length=args_cli.history_length,
        )
    else:
        _set_dense_lidar_observations(
            env_cfg,
            num_rays=args_cli.lidar_num_rays,
            max_distance=args_cli.lidar_max_distance,
            fov_deg=args_cli.lidar_fov_deg,
            history_length=args_cli.history_length,
        )

    env = gym.make(task_name, cfg=env_cfg)
    env.unwrapped.obstacle_manager = ObstacleManager(env.unwrapped, env.unwrapped.cfg)
    env.unwrapped.obstacle_manager.spawn_obstacles()

    device = args_cli.device
    num_envs = env.unwrapped.num_envs
    num_obs = env.unwrapped.observation_space["policy"].shape[-1]
    num_privileged_obs = env.unwrapped.observation_space["critic"].shape[-1]
    num_actions = env.unwrapped.action_space.shape[-1]

    if args_cli.lidar_mode == "latent64":
        actor_term_specs = build_actor_term_specs(args_cli.history_length, args_cli.lidar_latent_dim)
        critic_term_specs = build_critic_term_specs(args_cli.history_length, args_cli.lidar_latent_dim)
    elif args_cli.lidar_mode == "raw_pointcloud":
        actor_term_specs = build_actor_pointcloud_term_specs(args_cli.history_length, args_cli.omni_point_samples)
        critic_term_specs = build_critic_pointcloud_term_specs(args_cli.history_length, args_cli.omni_point_samples)
    else:
        actor_term_specs = build_actor_term_specs(args_cli.history_length, args_cli.lidar_num_rays)
        critic_term_specs = build_critic_term_specs(args_cli.history_length, args_cli.lidar_num_rays)

    policy = ActorCriticSafePerception(
        actor_term_specs=actor_term_specs,
        critic_term_specs=critic_term_specs,
        num_actions=num_actions,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
        activation="elu",
        init_noise_std=1.0,
        proprio_hidden_dim=128,
        scan_hidden_dim=64 if args_cli.lidar_mode == "latent64" else 128,
        rnn_hidden_dim=64 if args_cli.lidar_mode == "latent64" else 128,
    ).to(device)

    if args_cli.resume:
        checkpoint = torch.load(args_cli.resume, map_location=device)
        policy.load_state_dict(checkpoint["policy_state_dict"], strict=True)
        print(f"[INFO] Loaded checkpoint: {args_cli.resume}")

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

    log_dir = os.path.join(
        "logs", f"p3o_g1_cbf_paper_perception_{args_cli.stage}", datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
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
    if isinstance(reset_result, tuple):
        obs_dict, _ = reset_result
    else:
        obs_dict = reset_result

    for iteration in range(args_cli.max_iterations):
        stage_progress = get_stage_progress(iteration, args_cli.max_iterations, args_cli.stage)
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
        writer.add_scalar("perception/lidar_num_rays", args_cli.lidar_num_rays, iteration)
        writer.add_scalar("perception/omni_point_samples", args_cli.omni_point_samples, iteration)
        writer.add_scalar("perception/lidar_latent_dim", args_cli.lidar_latent_dim, iteration)
        writer.add_scalar("perception/history_length", args_cli.history_length, iteration)

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
        "iteration": args_cli.max_iterations,
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
