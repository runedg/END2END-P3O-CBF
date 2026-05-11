#!/usr/bin/env python3
"""Speed-extension fine-tuning for P3O_V2_Tuned.

Resumes from a P3O_V2_Tuned checkpoint and progressively extends the
forward-speed command range up to ~1.0 m/s while **keeping obstacle-avoidance
behaviour intact**.

Key design choices:
- All CBF / safety parameters are frozen to the v2-tuned stage-6 values.
- All obstacle-avoidance reward weights are frozen to stage-6 values.
- Only `lin_vel_x` range is gradually increased via a mini-curriculum.
- Low learning rate to avoid catastrophic forgetting of avoidance skills.
"""

from __future__ import annotations

import argparse
import math
import os
from datetime import datetime

import torch

from cuda_reduce_workaround import apply_cuda_reduce_workaround
from isaaclab.app import AppLauncher
from actor_critic_safe_perception import ActorCriticSafePerception, ObsTermSpec

apply_cuda_reduce_workaround()

# ── argparse ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Speed-extension fine-tune for P3O_V2_Tuned.")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--max_iterations", type=int, default=8000)
parser.add_argument("--save_interval", type=int, default=1000)
parser.add_argument("--experiment_name", type=str, default="P3O_V2_Tuned_SpeedExt")
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument(
    "--resume",
    type=str,
    default="/home/ubuntu/P3O-CBF/logs/P3O_V2_Tuned/2026-05-09_01-24-50/model_26000.pt",
    help="P3O_V2_Tuned checkpoint to resume from.",
)
parser.add_argument("--num_steps_per_env", type=int, default=24)
parser.add_argument("--num_learning_epochs", type=int, default=5)
parser.add_argument("--num_mini_batches", type=int, default=24)
parser.add_argument("--learning_rate", type=float, default=5e-5, help="Low LR to preserve avoidance.")
parser.add_argument("--cost_critic_learning_rate", type=float, default=1e-5)
parser.add_argument("--entropy_coef", type=float, default=0.005)
parser.add_argument("--cost_limit", type=float, default=0.22)
parser.add_argument("--kappa", type=float, default=1.0)
parser.add_argument("--cost_gamma", type=float, default=0.99)
parser.add_argument("--cost_lam", type=float, default=0.95)

# CBF / obstacle — FROZEN to v2-tuned stage-6 values
parser.add_argument("--safety_margin", type=float, default=0.55)
parser.add_argument("--cbf_gamma", type=float, default=0.35)
parser.add_argument("--collision_distance", type=float, default=0.18)
parser.add_argument("--unsafe_cost_weight", type=float, default=1.0)
parser.add_argument("--cbf_cost_weight", type=float, default=1.0)
parser.add_argument("--collision_cost_weight", type=float, default=2.0)
parser.add_argument("--contact_cost_weight", type=float, default=2.5)
parser.add_argument("--contact_force_threshold", type=float, default=1.0)
parser.add_argument("--num_obstacles", type=int, default=12,
                    help="Number of obstacles for fine-tuning (default 12). "
                         "Higher = stronger avoidance pressure. "
                         "Use 8-16 for a balanced speed/avoidance trade-off.")
parser.add_argument("--obstacle_layout", type=str, default="continuous_avoidance",
                    choices=["continuous_avoidance", "mixed_obstacles_no_wall",
                             "surrounded_front_open", "front_dense_hybrid"],
                    help="Obstacle layout variant.")

# Perception (same as v2-tuned)
parser.add_argument(
    "--omni_pattern_file",
    type=str,
    default="/home/ubuntu/P3O-CBF/OmniPerception/LidarSensor/LidarSensor/sensor_pattern/sensor_lidar/scan_mode/mid360.npy",
)
parser.add_argument("--history_length", type=int, default=5)
parser.add_argument("--omni_point_samples", type=int, default=1024)
parser.add_argument("--num_fps_points", type=int, default=128)
parser.add_argument("--lidar_max_distance", type=float, default=6.0)
parser.add_argument("--compression_fov_deg", type=float, default=180.0)
parser.add_argument("--roi_x_min", type=float, default=-0.5)
parser.add_argument("--roi_x_max", type=float, default=6.0)
parser.add_argument("--roi_abs_y_max", type=float, default=3.0)
parser.add_argument("--roi_z_min", type=float, default=-1.0)
parser.add_argument("--roi_z_max", type=float, default=0.8)
parser.add_argument("--min_planar_distance", type=float, default=0.2)
parser.add_argument("--robot_body_radius", type=float, default=0.4)
parser.add_argument("--sensor_offset_x", type=float, default=0.10)
parser.add_argument("--sensor_offset_y", type=float, default=0.0)
parser.add_argument("--sensor_offset_z", type=float, default=0.63)

# Sensor noise
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

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
from isaaclab.managers import ObservationTermCfg as ObsTerm  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402

from p3o_cbf_paper import P3OCBFPaper  # noqa: E402


# ════════════════════════════════════════════════════════════════════════
#  MINI-CURRICULUM: only lin_vel_x is increased.
#  Everything else is locked to the v2-tuned stage-6 values.
# ════════════════════════════════════════════════════════════════════════
SPEED_CURRICULUM = (
    {
        "name": "speed_ext_0",
        "progress": 0.0,
        "lin_vel_x": (0.20, 0.65),   # slight extension beyond 0.60
    },
    {
        "name": "speed_ext_1",
        "progress": 0.30,
        "lin_vel_x": (0.20, 0.75),   # approaching target
    },
    {
        "name": "speed_ext_2",
        "progress": 0.60,
        "lin_vel_x": (0.20, 0.80),   # target speed
    },
)

# Frozen stage-6 reward config (do NOT change these)
# Obstacle count/layout are user-configurable so they can be tuned
# without touching safety parameters or reward weights.
STAGE6_REWARDS = {
    "track_lin": 2.00,
    "track_ang": 0.95,
    "proxemic": -0.08,
    "safe_vel": -0.12,
    "safe_acc": -0.04,
    "tangential": 1.10,
    "arm_dev": -0.50,
    "alive": 0.25,
}
STAGE6_VEL_RANGES = {
    "lin_vel_y": 0.16,
    "ang_vel_z": 0.55,
    "standing": 0.0,
}


# ── helpers (identical to v2-tuned trainer) ────────────────────────────

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
    return {
        "track/lin_error": torch.linalg.norm(actual_lin - cmd_lin, dim=-1).mean().item(),
        "track/actual_speed": torch.linalg.norm(actual_lin, dim=-1).mean().item(),
        "track/command_speed": torch.linalg.norm(cmd_lin, dim=-1).mean().item(),
    }


def build_actor_term_specs(h: int, n_pts: int) -> list[ObsTermSpec]:
    return [
        ObsTermSpec("base_ang_vel", 3, h),
        ObsTermSpec("projected_gravity", 3, h),
        ObsTermSpec("velocity_commands", 3, h),
        ObsTermSpec("lidar_points", n_pts * 3, h),
        ObsTermSpec("joint_pos_rel", 29, h),
        ObsTermSpec("joint_vel_rel", 29, h),
        ObsTermSpec("last_action", 29, h),
    ]


def build_critic_term_specs(h: int, n_pts: int) -> list[ObsTermSpec]:
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

def _configure_env(env_cfg):
    import isaaclab.terrains as terrain_gen

    obstacle_env_cfg_mod = __import__(
        "unitree_rl_lab.tasks.locomotion.robots.g1.29dof.obstacle_env_cfg",
        fromlist=["MeshClutterPillarsTerrainCfg"],
    )
    MeshClutterPillarsTerrainCfg = obstacle_env_cfg_mod.MeshClutterPillarsTerrainCfg

    terrain_rows = int(math.ceil(math.sqrt(args_cli.num_envs)))
    terrain_cols = int(math.ceil(args_cli.num_envs / terrain_rows))

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
                num_obstacles=args_cli.num_obstacles,
                radius=env_cfg.obstacle_radius,
                height=env_cfg.obstacle_height,
                x_range=(-10.5, 10.5),
                y_range=(-8.5, 8.5),
                layout_variant=args_cli.obstacle_layout,
            ),
        },
    )
    env_cfg.min_obstacle_distance = 1.7
    env_cfg.num_obstacles = args_cli.num_obstacles
    env_cfg.obstacle_layout_mode = "clutter"
    env_cfg.obstacle_layout_variant = args_cli.obstacle_layout
    env_cfg.obstacle_x_range = (-10.5, 10.5)
    env_cfg.obstacle_y_range = (-8.5, 8.5)
    env_cfg.obstacle_min_gap = 1.35
    env_cfg.obstacle_robot_clearance = 2.4
    env_cfg.obstacle_collision_mode = "terrain_mesh"

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

    # v2-tuned stage-6 reward base (FROZEN)
    env_cfg.rewards.alive.weight = 0.20
    env_cfg.rewards.base_height.weight = -6.0
    env_cfg.rewards.flat_orientation_l2.weight = -2.0
    env_cfg.rewards.action_rate.weight = -0.01
    env_cfg.rewards.joint_vel.weight = -0.0006
    env_cfg.rewards.energy.weight = -1.0e-5
    env_cfg.rewards.joint_deviation_waists.weight = -0.50
    env_cfg.rewards.joint_deviation_legs.weight = -0.50
    env_cfg.rewards.gait.weight = 0.80
    env_cfg.rewards.feet_slide.weight = -0.12
    env_cfg.rewards.feet_clearance.weight = 1.50
    env_cfg.rewards.undesired_contacts.weight = -0.40

    # Frozen avoidance / tracking weights from stage-6
    env_cfg.rewards.track_lin_vel_xy.weight = STAGE6_REWARDS["track_lin"]
    env_cfg.rewards.track_ang_vel_z.weight = STAGE6_REWARDS["track_ang"]
    env_cfg.rewards.proxemic_comfort.weight = STAGE6_REWARDS["proxemic"]
    env_cfg.rewards.safe_approach_velocity.weight = STAGE6_REWARDS["safe_vel"]
    env_cfg.rewards.safe_approach_acceleration.weight = STAGE6_REWARDS["safe_acc"]
    env_cfg.rewards.tangential_avoidance.weight = STAGE6_REWARDS["tangential"]
    env_cfg.rewards.joint_deviation_arms.weight = STAGE6_REWARDS["arm_dev"]
    env_cfg.rewards.alive.weight = STAGE6_REWARDS["alive"]


def _set_pointcloud_observations(env_cfg):
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


def _apply_speed_stage(env, stage_index: int) -> dict:
    stage = SPEED_CURRICULUM[stage_index]
    cfg = env.unwrapped.cfg

    # ONLY change speed command range
    cfg.commands.base_velocity.ranges.lin_vel_x = tuple(stage["lin_vel_x"])
    cfg.commands.base_velocity.ranges.lin_vel_y = (
        -STAGE6_VEL_RANGES["lin_vel_y"],
        STAGE6_VEL_RANGES["lin_vel_y"],
    )
    cfg.commands.base_velocity.ranges.ang_vel_z = (
        -STAGE6_VEL_RANGES["ang_vel_z"],
        STAGE6_VEL_RANGES["ang_vel_z"],
    )
    cfg.commands.base_velocity.rel_standing_envs = STAGE6_VEL_RANGES["standing"]

    return stage


def _stage_index_for_iteration(iteration: int, final_iteration: int) -> int:
    if final_iteration <= 1:
        return len(SPEED_CURRICULUM) - 1
    progress = float(iteration) / float(final_iteration - 1)
    stage_index = 0
    for idx, stage in enumerate(SPEED_CURRICULUM):
        if progress >= float(stage["progress"]):
            stage_index = idx
    return stage_index


# ── main ───────────────────────────────────────────────────────────────

def main():
    import unitree_rl_lab.tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from unitree_rl_lab.tasks.locomotion.obstacle_manager import ObstacleManager

    task_name = "Unitree-G1-29dof-ObstacleAvoidance-realG1"
    env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    _configure_env(env_cfg)
    _set_pointcloud_observations(env_cfg)

    env = gym.make(task_name, cfg=env_cfg)
    env = env.unwrapped
    env.obstacle_manager = ObstacleManager(env, env.cfg)
    env.obstacle_manager.spawn_obstacles()
    _apply_speed_stage(env, 0)

    device = args_cli.device
    num_envs = env.num_envs
    num_steps_per_env = args_cli.num_steps_per_env

    obs_dict = env.reset()
    if isinstance(obs_dict, tuple):
        obs_dict = obs_dict[0]

    num_obs = obs_dict["policy"].shape[-1]
    num_privileged_obs = obs_dict["critic"].shape[-1]
    num_actions = env.action_space.shape[-1]

    policy = ActorCriticSafePerception(
        actor_term_specs=build_actor_term_specs(args_cli.history_length, args_cli.num_fps_points),
        critic_term_specs=build_critic_term_specs(args_cli.history_length, args_cli.num_fps_points),
        num_actions=num_actions,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
        activation="elu",
        init_noise_std=1.0,
        proprio_hidden_dim=128,
        scan_hidden_dim=64,
        rnn_hidden_dim=64,
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(args_cli.resume, map_location=device, weights_only=False)
    policy.load_state_dict(checkpoint["policy_state_dict"], strict=True)
    print(f"[INFO] Loaded checkpoint: {args_cli.resume}")
    start_iteration = int(checkpoint.get("iteration", 0))
    print(f"[INFO] Resuming from iteration {start_iteration}")

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

    # Restore optimizers if possible
    if "optimizer_state_dict" in checkpoint:
        try:
            alg.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("[INFO] Restored actor-critic optimizer")
        except Exception as e:
            print(f"[WARN] Could not restore optimizer: {e}")
    if "cost_optimizer_state_dict" in checkpoint:
        try:
            alg.cost_value_optimizer.load_state_dict(checkpoint["cost_optimizer_state_dict"])
            print("[INFO] Restored cost critic optimizer")
        except Exception as e:
            print(f"[WARN] Could not restore cost optimizer: {e}")

    # Reset LR to the (lower) fine-tuning value
    for group in alg.optimizer.param_groups:
        group["lr"] = args_cli.learning_rate
    for group in alg.cost_value_optimizer.param_groups:
        group["lr"] = args_cli.cost_critic_learning_rate

    alg.init_storage(
        num_envs=num_envs,
        num_transitions_per_env=num_steps_per_env,
        actor_obs_shape=[num_obs],
        critic_obs_shape=[num_privileged_obs],
        action_shape=[num_actions],
    )

    log_root = os.path.join("logs", args_cli.experiment_name)
    log_dir = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"[INFO] Logging to {log_dir}")
    print(f"[INFO] Speed-extension fine-tune: lr={args_cli.learning_rate}")
    print(f"[INFO] Avoidance params FROZEN: safety_margin={args_cli.safety_margin}, "
          f"cbf_gamma={args_cli.cbf_gamma}, collision_dist={args_cli.collision_distance}")

    final_iteration = start_iteration + args_cli.max_iterations
    last_stage_index = None

    for iteration in range(start_iteration, final_iteration):
        stage_index = _stage_index_for_iteration(iteration - start_iteration, args_cli.max_iterations)
        if stage_index != last_stage_index:
            stage = _apply_speed_stage(env, stage_index)
            last_stage_index = stage_index
            print(f"[STAGE CHANGE] {stage['name']} | vx={stage['lin_vel_x']} "
                  f"(iter {iteration}/{final_iteration})")
        else:
            stage = SPEED_CURRICULUM[stage_index]

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

            costs, paper_info = compute_paper_cost(env, args_cli)
            track_info = compute_tracking_metrics(env)

            episode_reward += rewards.to(device)
            episode_cost += costs
            for k, v in paper_info.items():
                paper_info_acc.setdefault(k, 0.0)
                paper_info_acc[k] += v
            for k, v in track_info.items():
                tracking_info_acc.setdefault(k, 0.0)
                tracking_info_acc[k] += v
            alg.process_env_step(rewards.to(device), costs, dones, infos)

        with torch.no_grad():
            alg.compute_returns(obs_dict["critic"].to(device))

        losses = alg.update()
        reward_mean = episode_reward.mean().item() / num_steps_per_env
        cost_mean = episode_cost.mean().item() / num_steps_per_env

        writer.add_scalar("train/reward", reward_mean, iteration)
        writer.add_scalar("train/cost", cost_mean, iteration)
        writer.add_scalar("curriculum/stage_index", stage_index, iteration)
        writer.add_scalar("curriculum/cmd_lin_vel_x_min", stage["lin_vel_x"][0], iteration)
        writer.add_scalar("curriculum/cmd_lin_vel_x_max", stage["lin_vel_x"][1], iteration)

        for k, v in losses.items():
            writer.add_scalar(f"loss/{k}", v, iteration)
        for k, v in paper_info_acc.items():
            writer.add_scalar(k, v / num_steps_per_env, iteration)
        for k, v in tracking_info_acc.items():
            writer.add_scalar(k, v / num_steps_per_env, iteration)

        if (iteration + 1) % 20 == 0 or iteration == start_iteration:
            print(
                f"[ITER {iteration + 1:05d}] "
                f"stage={stage['name']} reward={reward_mean:.4f} cost={cost_mean:.4f} "
                f"unsafe={paper_info_acc.get('paper/unsafe_rate', 0.0) / num_steps_per_env:.4f} "
                f"collision={paper_info_acc.get('paper/collision_rate', 0.0) / num_steps_per_env:.4f} "
                f"contact={paper_info_acc.get('paper/contact_rate', 0.0) / num_steps_per_env:.4f} "
                f"speed={tracking_info_acc.get('track/actual_speed', 0.0) / num_steps_per_env:.3f}"
            )

        if (iteration + 1) % args_cli.save_interval == 0:
            torch.save({
                "iteration": iteration + 1,
                "policy_state_dict": policy.state_dict(),
                "optimizer_state_dict": alg.optimizer.state_dict(),
                "cost_optimizer_state_dict": alg.cost_value_optimizer.state_dict(),
                "args": vars(args_cli),
            }, os.path.join(log_dir, f"model_{iteration + 1}.pt"))

    torch.save({
        "iteration": final_iteration,
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": alg.optimizer.state_dict(),
        "cost_optimizer_state_dict": alg.cost_value_optimizer.state_dict(),
        "args": vars(args_cli),
    }, os.path.join(log_dir, "model_final.pt"))
    print(f"[INFO] Final model saved to {log_dir}/model_final.pt")
    writer.close()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
