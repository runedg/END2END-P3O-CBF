# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent G1 goal-navigation P3O-HOCBF experiment with 2D obstacle scan.

This script is intentionally isolated from the successful P3O-END2END-001
baseline.  It changes the task from velocity-command tracking to target-point
navigation and changes obstacle perception to a fixed 2D scan interface.
"""

import argparse
import importlib
import math
import os
from datetime import datetime

import torch

from isaaclab.app import AppLauncher

from actor_critic_safe_perception import ActorCriticSafePerception, ObsTermSpec

parser = argparse.ArgumentParser(description="Train an isolated G1 P3O-HOCBF goal-navigation policy with 2D scan.")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments.")
parser.add_argument("--max_iterations", type=int, default=15000, help="Training iterations.")
parser.add_argument("--save_interval", type=int, default=2000, help="Save checkpoint every N iterations.")
parser.add_argument("--experiment_name", type=str, default="P3O-GOAL-HOCBF-SCAN-001", help="Top-level log folder name.")
parser.add_argument("--headless", action="store_true", default=False, help="Run in headless mode.")
parser.add_argument("--device", type=str, default="cuda:0", help="Device to use.")
parser.add_argument("--force_stage", type=int, default=None, help="Optional fixed curriculum stage index.")
parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint to resume from.")
parser.add_argument("--cost_limit", type=float, default=0.22, help="Rollout cost limit for the paper-style P3O term.")
parser.add_argument("--kappa", type=float, default=1.2, help="Penalty coefficient.")
parser.add_argument("--cost_gamma", type=float, default=0.99, help="Cost discount factor.")
parser.add_argument("--cost_lam", type=float, default=0.95, help="Cost GAE lambda.")
parser.add_argument("--safety_margin", type=float, default=0.8, help="Safe distance to obstacles.")
parser.add_argument("--cbf_gamma", type=float, default=0.5, help="Discrete CBF coefficient.")
parser.add_argument("--hocbf_alpha1", type=float, default=2.0, help="First high-order CBF gain.")
parser.add_argument("--hocbf_alpha2", type=float, default=2.0, help="Second high-order CBF gain.")
parser.add_argument("--hocbf_accel_clip", type=float, default=8.0, help="Clamp for finite-difference h_ddot.")
parser.add_argument("--hocbf_warmup_steps", type=int, default=4, help="Initial cost calls using zero h_ddot.")
parser.add_argument("--collision_distance", type=float, default=0.2, help="Distance counted as collision.")
parser.add_argument("--unsafe_cost_weight", type=float, default=1.25, help="Weight for entering the unsafe set.")
parser.add_argument("--cbf_cost_weight", type=float, default=1.25, help="Weight for discrete CBF violation.")
parser.add_argument("--collision_cost_weight", type=float, default=3.0, help="Extra weight for near-collision states.")
parser.add_argument("--curriculum_max_obstacles", type=int, default=24, help="Maximum number of obstacles to reveal.")
parser.add_argument("--history_length", type=int, default=5, help="Observation history length.")
parser.add_argument("--num_scan_rays", type=int, default=240, help="Fixed 2D scan dimension exposed to the policy.")
parser.add_argument("--scan_fov_deg", type=float, default=240.0, help="2D scan field of view in degrees.")
parser.add_argument("--scan_max_distance", type=float, default=6.0, help="Maximum 2D scan range.")
parser.add_argument("--goal_max_distance", type=float, default=8.0, help="Maximum sampled target distance.")
parser.add_argument("--goal_reach_threshold", type=float, default=0.55, help="Distance threshold for target arrival.")
parser.add_argument("--goal_progress_weight", type=float, default=14.0, help="Reward weight for progress toward target.")
parser.add_argument("--goal_heading_weight", type=float, default=0.50, help="Reward weight for facing target.")
parser.add_argument("--goal_arrival_bonus", type=float, default=12.0, help="One-step bonus when reaching target.")
parser.add_argument("--base_env_reward_weight", type=float, default=0.15, help="Weight retained from gait/posture environment rewards.")

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab.managers import ObservationTermCfg as ObsTerm
from torch.utils.tensorboard import SummaryWriter

from p3o_cbf_paper import P3OCBFPaper


GOAL_CURRICULUM_STAGES = (
    {
        "name": "stage0_goal_no_obstacles",
        "progress": 0.0,
        "num_obstacles": 0,
        "goal_distance": (0.8, 1.8),
        "goal_angle_deg": (-20.0, 20.0),
        "layout": "radial",
        "cost_limit": 0.30,
        "kappa_scale": 0.6,
    },
    {
        "name": "stage1_goal_short_turns_no_obstacles",
        "progress": 0.28,
        "num_obstacles": 0,
        "goal_distance": (1.0, 2.5),
        "goal_angle_deg": (-45.0, 45.0),
        "layout": "radial",
        "cost_limit": 0.30,
        "kappa_scale": 0.7,
    },
    {
        "name": "stage2_goal_two_open",
        "progress": 0.48,
        "num_obstacles": 2,
        "goal_distance": (2.0, 4.0),
        "goal_angle_deg": (-45.0, 45.0),
        "layout": "continuous_avoidance",
        "cost_limit": 0.24,
        "kappa_scale": 0.8,
    },
    {
        "name": "stage3_goal_four_choices",
        "progress": 0.62,
        "num_obstacles": 4,
        "goal_distance": (2.5, 5.0),
        "goal_angle_deg": (-60.0, 60.0),
        "layout": "continuous_avoidance",
        "cost_limit": 0.18,
        "kappa_scale": 1.0,
    },
    {
        "name": "stage4_goal_eight_clutter",
        "progress": 0.74,
        "num_obstacles": 8,
        "goal_distance": (3.0, 6.0),
        "goal_angle_deg": (-75.0, 75.0),
        "layout": "front_dense_hybrid",
        "cost_limit": 0.13,
        "kappa_scale": 1.2,
    },
    {
        "name": "stage5_goal_twelve_mixed",
        "progress": 0.86,
        "num_obstacles": 12,
        "goal_distance": (3.0, 7.0),
        "goal_angle_deg": (-100.0, 100.0),
        "layout": "surrounded_front_open",
        "cost_limit": 0.10,
        "kappa_scale": 1.4,
    },
    {
        "name": "stage6_goal_twenty_final",
        "progress": 0.94,
        "num_obstacles": 20,
        "goal_distance": (4.0, 8.0),
        "goal_angle_deg": (-120.0, 120.0),
        "layout": "mixed_obstacles_no_wall",
        "cost_limit": 0.08,
        "kappa_scale": 1.6,
    },
)


def _root_yaw(root_quat_w: torch.Tensor) -> torch.Tensor:
    qw = root_quat_w[:, 0]
    qx = root_quat_w[:, 1]
    qy = root_quat_w[:, 2]
    qz = root_quat_w[:, 3]
    return torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


class GoalNavigationManager:
    def __init__(self, env, cfg):
        self.env = env
        self.cfg = cfg
        self.target_xy = torch.zeros(env.num_envs, 2, device=env.device)
        self.prev_distance = torch.zeros(env.num_envs, device=env.device)
        self.stage = GOAL_CURRICULUM_STAGES[0]
        self.resample()

    def set_stage(self, stage: dict):
        self.stage = stage

    def resample(self, env_ids: torch.Tensor | None = None):
        robot = self.env.scene["robot"]
        if env_ids is None:
            env_ids = torch.arange(self.env.num_envs, device=self.env.device)
        root_xy = robot.data.root_pos_w[env_ids, :2]
        yaw = _root_yaw(robot.data.root_quat_w[env_ids])
        d_min, d_max = self.stage["goal_distance"]
        a_min, a_max = [math.radians(v) for v in self.stage["goal_angle_deg"]]
        distance = torch.empty_like(yaw).uniform_(float(d_min), float(d_max))
        angle = torch.empty_like(yaw).uniform_(float(a_min), float(a_max))
        world_angle = yaw + angle
        offset = torch.stack((torch.cos(world_angle), torch.sin(world_angle)), dim=-1) * distance.unsqueeze(-1)
        self.target_xy[env_ids] = root_xy + offset
        self.prev_distance[env_ids] = distance

    def relative_goal(self) -> torch.Tensor:
        robot = self.env.scene["robot"]
        delta_w = self.target_xy - robot.data.root_pos_w[:, :2]
        yaw = _root_yaw(robot.data.root_quat_w)
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        dx_b = cos_yaw * delta_w[:, 0] + sin_yaw * delta_w[:, 1]
        dy_b = -sin_yaw * delta_w[:, 0] + cos_yaw * delta_w[:, 1]
        dist = torch.linalg.norm(delta_w, dim=-1).clamp_min(1.0e-6)
        angle = torch.atan2(dy_b, dx_b)
        max_dist = float(self.cfg.goal_max_distance)
        return torch.stack(
            (
                (dx_b / max_dist).clamp(-1.0, 1.0),
                (dy_b / max_dist).clamp(-1.0, 1.0),
                (dist / max_dist).clamp(0.0, 1.0),
                angle / math.pi,
            ),
            dim=-1,
        )

    def compute_reward(self) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
        robot = self.env.scene["robot"]
        distance = torch.linalg.norm(self.target_xy - robot.data.root_pos_w[:, :2], dim=-1)
        progress = self.prev_distance - distance
        rel = self.relative_goal()
        heading_reward = torch.cos(rel[:, 3] * math.pi)
        reached = distance < float(self.cfg.goal_reach_threshold)
        reward = (
            float(self.cfg.goal_progress_weight) * progress
            + float(self.cfg.goal_heading_weight) * heading_reward
            + float(self.cfg.goal_arrival_bonus) * reached.float()
        )
        self.prev_distance = distance.detach()
        info = {
            "goal/distance_mean": distance.mean().item(),
            "goal/distance_min": distance.min().item(),
            "goal/progress_mean": progress.mean().item(),
            "goal/reach_rate": reached.float().mean().item(),
            "goal/heading_reward": heading_reward.mean().item(),
        }
        return reward, info, reached


def relative_goal_observation(env, max_goal_distance: float = 8.0):
    if not hasattr(env, "goal_manager"):
        return torch.zeros(env.num_envs, 4, device=env.device)
    return env.goal_manager.relative_goal()


class HOCBFCostState:
    def __init__(self, num_envs: int, device: str, warmup_steps: int):
        self.prev_h_dot = torch.zeros(num_envs, device=device)
        self.initialized = False
        self.calls = 0
        self.warmup_steps = warmup_steps

    def h_ddot(self, h_dot: torch.Tensor, dt: float, accel_clip: float) -> torch.Tensor:
        if not self.initialized:
            self.prev_h_dot.copy_(h_dot.detach())
            self.initialized = True
            self.calls = 1
            return torch.zeros_like(h_dot)
        raw = (h_dot - self.prev_h_dot) / max(dt, 1.0e-6)
        self.prev_h_dot.copy_(h_dot.detach())
        self.calls += 1
        if self.calls <= self.warmup_steps:
            return torch.zeros_like(h_dot)
        return raw.clamp(-accel_clip, accel_clip)


def _closest_obstacle_geometry(env):
    robot_xy = env.scene["robot"].data.root_pos_w[:, :2]
    return env.obstacle_manager.get_closest_geometry(robot_xy)


def compute_paper_cost(env, cfg, hocbf_state: HOCBFCostState):
    robot = env.scene["robot"]
    robot_vel_xy = robot.data.root_lin_vel_w[:, :2]
    distances, directions = _closest_obstacle_geometry(env)
    h = distances - cfg.safety_margin
    h_dot = -torch.sum(robot_vel_xy * directions, dim=-1)
    dt = float(env.cfg.decimation * env.cfg.sim.dt)
    h_ddot = hocbf_state.h_ddot(h_dot, dt, cfg.hocbf_accel_clip)
    psi1 = h_dot + cfg.hocbf_alpha1 * h
    hocbf_margin = h_ddot + cfg.hocbf_alpha2 * psi1
    cbf_violation = torch.relu(-hocbf_margin)
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
        "paper/h_ddot_mean": h_ddot.mean().item(),
        "paper/psi1_mean": psi1.mean().item(),
        "paper/hocbf_margin_mean": hocbf_margin.mean().item(),
        "paper/hocbf_margin_min": hocbf_margin.min().item(),
        "paper/cbf_violation_rate": (cbf_violation > 0).float().mean().item(),
        "paper/hocbf_violation_rate": (cbf_violation > 0).float().mean().item(),
        "paper/unsafe_rate": unsafe_cost.mean().item(),
        "paper/collision_rate": collision_cost.mean().item(),
    }
    return total_cost, info


def _set_gait_reward_profile(env_cfg, gait_weight: float, feet_slide_weight: float, undesired_contacts_weight: float):
    env_cfg.rewards.gait.weight = gait_weight
    env_cfg.rewards.feet_slide.weight = feet_slide_weight
    env_cfg.rewards.undesired_contacts.weight = undesired_contacts_weight


def configure_stage(env_cfg, stage: dict):
    import isaaclab.terrains as terrain_gen

    obstacle_env_cfg_mod = importlib.import_module(
        "unitree_rl_lab.tasks.locomotion.robots.g1.29dof.obstacle_env_cfg"
    )
    MeshClutterPillarsTerrainCfg = obstacle_env_cfg_mod.MeshClutterPillarsTerrainCfg

    env_cfg.num_obstacles = max(1, max(int(curr_stage["num_obstacles"]) for curr_stage in GOAL_CURRICULUM_STAGES))
    env_cfg.min_obstacle_distance = 1.75
    env_cfg.obstacle_layout_mode = "clutter" if int(stage["num_obstacles"]) >= 8 else "radial"
    env_cfg.obstacle_layout_variant = str(stage["layout"])
    env_cfg.obstacle_x_range = (-2.0, 18.0)
    env_cfg.obstacle_y_range = (-6.0, 6.0)
    env_cfg.obstacle_min_gap = 1.05
    env_cfg.obstacle_robot_clearance = 1.8
    env_cfg.obstacle_collision_mode = "terrain_mesh"
    env_cfg.obstacle_num_rays = args_cli.num_scan_rays
    env_cfg.obstacle_ray_max_distance = args_cli.scan_max_distance
    env_cfg.obstacle_ray_fov_deg = args_cli.scan_fov_deg

    terrain_rows = int(math.ceil(math.sqrt(args_cli.num_envs)))
    terrain_cols = int(math.ceil(args_cli.num_envs / terrain_rows))
    env_cfg.scene.env_spacing = max(env_cfg.scene.env_spacing, 24.0)
    env_cfg.scene.terrain.max_init_terrain_level = terrain_rows - 1
    env_cfg.scene.terrain.terrain_generator = terrain_gen.TerrainGeneratorCfg(
        size=(44.0, 24.0),
        border_width=0.0,
        num_rows=terrain_rows,
        num_cols=terrain_cols,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        difficulty_range=(0.0, 0.0),
        use_cache=False,
        sub_terrains={
            "goal_clutter": MeshClutterPillarsTerrainCfg(
                proportion=1.0,
                num_obstacles=env_cfg.num_obstacles,
                radius=env_cfg.obstacle_radius,
                height=env_cfg.obstacle_height,
                x_range=env_cfg.obstacle_x_range,
                y_range=env_cfg.obstacle_y_range,
                layout_variant=env_cfg.obstacle_layout_variant,
            ),
        },
    )

    cmd_cfg = env_cfg.commands.base_velocity
    cmd_cfg.rel_standing_envs = 1.0
    cmd_cfg.rel_heading_envs = 0.0
    cmd_cfg.heading_command = False
    cmd_cfg.ranges.lin_vel_x = (0.0, 0.0)
    cmd_cfg.ranges.lin_vel_y = (0.0, 0.0)
    cmd_cfg.ranges.ang_vel_z = (0.0, 0.0)

    env_cfg.rewards.track_lin_vel_xy.weight = 0.0
    env_cfg.rewards.track_ang_vel_z.weight = 0.0
    env_cfg.rewards.alive.weight = 0.20
    env_cfg.rewards.base_height.weight = -7.5
    env_cfg.rewards.flat_orientation_l2.weight = -3.5
    env_cfg.rewards.action_rate.weight = -0.025
    env_cfg.rewards.joint_vel.weight = -0.0007
    env_cfg.rewards.energy.weight = -1.5e-5
    env_cfg.rewards.feet_clearance.weight = 0.8
    _set_gait_reward_profile(env_cfg, gait_weight=0.10, feet_slide_weight=-0.08, undesired_contacts_weight=-0.5)


def get_stage_progress(iteration: int, max_iterations: int) -> float:
    if max_iterations <= 1:
        return 1.0
    return iteration / float(max_iterations - 1)


def select_stage(stage_progress: float) -> tuple[int, dict]:
    if args_cli.force_stage is not None:
        index = max(0, min(int(args_cli.force_stage), len(GOAL_CURRICULUM_STAGES) - 1))
        return index, GOAL_CURRICULUM_STAGES[index]
    index = 0
    for candidate_idx, stage in enumerate(GOAL_CURRICULUM_STAGES):
        if stage_progress >= float(stage["progress"]):
            index = candidate_idx
    return index, GOAL_CURRICULUM_STAGES[index]


def set_active_obstacles_allow_zero(obstacle_manager, active_num_obstacles: int) -> int:
    if active_num_obstacles > 0:
        obstacle_manager.set_active_obstacles(active_num_obstacles)
        return int(obstacle_manager.active_num_obstacles)

    if obstacle_manager.max_num_obstacles <= 0:
        return 0

    obstacle_manager.set_active_obstacles(1)
    num_envs = obstacle_manager.env.num_envs
    max_num = obstacle_manager.max_num_obstacles
    env_origins = obstacle_manager.env.scene.env_origins
    device = obstacle_manager.env.device
    hidden = []
    world_positions_by_obs = [[] for _ in range(max_num)]
    for env_idx in range(num_envs):
        env_origin = env_origins[env_idx]
        env_positions = []
        for obs_idx in range(max_num):
            pos_x = env_origin[0].item() + 100.0 + 8.0 * obs_idx
            pos_y = env_origin[1].item() + 100.0 + 8.0 * env_idx
            env_positions.append([pos_x, pos_y, obstacle_manager.obstacle_radius])
            world_positions_by_obs[obs_idx].append([pos_x, pos_y, obstacle_manager.obstacle_height / 2.0])
        hidden.append(torch.tensor(env_positions, dtype=torch.float32))
    obstacle_manager.obstacle_positions = hidden
    obstacle_manager.obstacle_positions_tensor = torch.stack(hidden, dim=0)
    obstacle_manager.active_num_obstacles = 0
    obstacle_manager._refresh_geometry_proxy_tensor()
    if obstacle_manager._obstacle_views:
        orientations = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * num_envs, device=device)
        for obs_idx, view in enumerate(obstacle_manager._obstacle_views):
            positions = torch.tensor(world_positions_by_obs[obs_idx], dtype=torch.float32, device=device)
            view.set_world_poses(positions=positions, orientations=orientations)
    return 0


def apply_p3o_schedule(alg: P3OCBFPaper, cfg, stage_progress: float):
    _, stage = select_stage(stage_progress)
    alg.kappa = max(0.2, cfg.kappa * float(stage["kappa_scale"]))
    alg.cost_limit = min(cfg.cost_limit, float(stage["cost_limit"]))


def apply_obstacle_curriculum(obstacle_manager, cfg, stage_progress: float) -> int:
    _, stage = select_stage(stage_progress)
    max_active = min(int(stage["num_obstacles"]), obstacle_manager.max_num_obstacles)
    active = max(0, max_active)
    return set_active_obstacles_allow_zero(obstacle_manager, active)


def _set_goal_scan_observations(env_cfg):
    from unitree_rl_lab.tasks.locomotion import mdp

    scan_params = {"num_rays": args_cli.num_scan_rays, "max_distance": args_cli.scan_max_distance, "fov_deg": args_cli.scan_fov_deg}
    scan_scale = tuple([1.0 / args_cli.scan_max_distance] * args_cli.num_scan_rays)

    env_cfg.observations.policy.obstacle_scan = ObsTerm(
        func=mdp.obstacle_raycast_scan,
        params=scan_params,
        clip=(0.0, args_cli.scan_max_distance),
        scale=scan_scale,
    )
    env_cfg.observations.policy.velocity_commands = ObsTerm(
        func=relative_goal_observation,
        params={"max_goal_distance": args_cli.goal_max_distance},
        clip=(-1.0, 1.0),
    )
    env_cfg.observations.policy.history_length = args_cli.history_length

    env_cfg.observations.critic.obstacle_scan = ObsTerm(
        func=mdp.obstacle_raycast_scan,
        params=scan_params,
        clip=(0.0, args_cli.scan_max_distance),
        scale=scan_scale,
    )
    env_cfg.observations.critic.velocity_commands = ObsTerm(
        func=relative_goal_observation,
        params={"max_goal_distance": args_cli.goal_max_distance},
        clip=(-1.0, 1.0),
    )
    env_cfg.observations.critic.history_length = args_cli.history_length


def build_actor_term_specs(history_length: int, num_features: int) -> list[ObsTermSpec]:
    return [
        ObsTermSpec("base_ang_vel", 3, history_length),
        ObsTermSpec("projected_gravity", 3, history_length),
        ObsTermSpec("relative_goal", 4, history_length),
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
        ObsTermSpec("relative_goal", 4, history_length),
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
    initial_stage_index, initial_stage = select_stage(0.0)
    configure_stage(env_cfg, initial_stage)
    _set_goal_scan_observations(env_cfg)

    env = gym.make(task_name, cfg=env_cfg)
    env.unwrapped.obstacle_manager = ObstacleManager(env.unwrapped, env.unwrapped.cfg)
    env.unwrapped.obstacle_manager.spawn_obstacles()
    env.unwrapped.goal_manager = GoalNavigationManager(env.unwrapped, args_cli)
    env.unwrapped.goal_manager.set_stage(initial_stage)
    print(f"[INFO] Initial goal curriculum stage {initial_stage_index}: {initial_stage['name']}")

    device = args_cli.device
    num_envs = env.unwrapped.num_envs
    num_obs = env.unwrapped.observation_space["policy"].shape[-1]
    num_privileged_obs = env.unwrapped.observation_space["critic"].shape[-1]
    num_actions = env.unwrapped.action_space.shape[-1]

    actor_term_specs = build_actor_term_specs(args_cli.history_length, args_cli.num_scan_rays)
    critic_term_specs = build_critic_term_specs(args_cli.history_length, args_cli.num_scan_rays)

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
    hocbf_state = HOCBFCostState(num_envs, device, args_cli.hocbf_warmup_steps)
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
        stage_progress = get_stage_progress(local_iteration, max(args_cli.max_iterations, 1))
        stage_index, stage = select_stage(stage_progress)
        env.unwrapped.goal_manager.set_stage(stage)
        apply_p3o_schedule(alg, args_cli, stage_progress)
        active_obstacles = apply_obstacle_curriculum(env.unwrapped.obstacle_manager, args_cli, stage_progress)

        episode_reward = torch.zeros(num_envs, device=device)
        episode_cost = torch.zeros(num_envs, device=device)
        done_count = 0.0
        base_height_done_count = 0.0
        bad_orientation_done_count = 0.0
        paper_info_acc = {}
        goal_info_acc = {}

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

            costs, paper_info = compute_paper_cost(env.unwrapped, args_cli, hocbf_state)
            goal_rewards, goal_info, reached = env.unwrapped.goal_manager.compute_reward()
            shaped_rewards = args_cli.base_env_reward_weight * rewards.to(device) + goal_rewards

            reset_goal_ids = torch.nonzero(dones | reached, as_tuple=False).squeeze(-1)
            if reset_goal_ids.numel() > 0:
                env.unwrapped.goal_manager.resample(reset_goal_ids)

            episode_reward += shaped_rewards
            episode_cost += costs

            for key, value in paper_info.items():
                paper_info_acc.setdefault(key, 0.0)
                paper_info_acc[key] += value
            for key, value in goal_info.items():
                goal_info_acc.setdefault(key, 0.0)
                goal_info_acc[key] += value

            alg.process_env_step(shaped_rewards, costs, dones, infos)

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
        writer.add_scalar("curriculum/stage_index", stage_index, iteration)
        writer.add_text("curriculum/stage_name", stage["name"], iteration)
        writer.add_scalar("curriculum/goal_distance_max", stage["goal_distance"][1], iteration)
        writer.add_scalar("perception/scan_rays", args_cli.num_scan_rays, iteration)
        writer.add_scalar("perception/history_length", args_cli.history_length, iteration)
        writer.add_scalar("perception/scan_fov_deg", args_cli.scan_fov_deg, iteration)
        writer.add_scalar("perception/scan_max_distance", args_cli.scan_max_distance, iteration)

        for key, value in losses.items():
            writer.add_scalar(f"loss/{key}", value, iteration)
        for key, value in paper_info_acc.items():
            writer.add_scalar(key, value / num_steps_per_env, iteration)
        for key, value in goal_info_acc.items():
            writer.add_scalar(key, value / num_steps_per_env, iteration)

        if (iteration + 1) % 20 == 0 or iteration == 0:
            print(
                f"[ITER {iteration + 1:05d}] "
                f"reward={reward_mean:.4f} cost={cost_mean:.4f} "
                f"unsafe={paper_info_acc.get('paper/unsafe_rate', 0.0) / num_steps_per_env:.4f} "
                f"hocbf={paper_info_acc.get('paper/hocbf_violation_rate', 0.0) / num_steps_per_env:.4f} "
                f"goal={goal_info_acc.get('goal/distance_mean', 0.0) / num_steps_per_env:.2f} "
                f"stage={stage_index}:{stage['name']} "
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
