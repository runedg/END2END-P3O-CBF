# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train G1 walking obstacle avoidance with P3O-CBF."""

import argparse
import os
import subprocess
import sys
from datetime import datetime

import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train G1 obstacle avoidance with P3O-CBF.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments.")
parser.add_argument("--max_iterations", type=int, default=15000, help="Training iterations.")
parser.add_argument("--save_interval", type=int, default=2000, help="Save checkpoint every N iterations.")
parser.add_argument("--headless", action="store_true", default=False, help="Run in headless mode.")
parser.add_argument("--device", type=str, default="cuda:0", help="Device to use.")
parser.add_argument("--stage", type=str, default="walk", choices=["walk", "avoid"], help="Training stage.")
parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint to resume from.")
parser.add_argument("--auto_eval", action="store_true", default=False, help="Auto-record evaluation videos at checkpoints.")
parser.add_argument("--eval_steps", type=int, default=450, help="Number of steps per automatic evaluation video.")
parser.add_argument("--eval_num_envs", type=int, default=1, help="Number of environments for automatic evaluation.")

# P3O
parser.add_argument("--cost_limit", type=float, default=25.0, help="Cost limit for P3O.")
parser.add_argument("--kappa", type=float, default=1.0, help="Penalty coefficient.")
parser.add_argument("--cost_gamma", type=float, default=0.99, help="Cost discount factor.")
parser.add_argument("--cost_lam", type=float, default=0.95, help="Cost GAE lambda.")

# CBF
parser.add_argument("--safety_margin", type=float, default=0.8, help="Safe distance to obstacles.")
parser.add_argument("--cbf_gamma", type=float, default=0.5, help="CBF gamma in h_dot + gamma * h.")
parser.add_argument("--collision_distance", type=float, default=0.2, help="Distance counted as collision.")
parser.add_argument("--obstacle_num_rays", type=int, default=9, help="Number of analytical ray-cast rays.")
parser.add_argument("--obstacle_ray_max_distance", type=float, default=6.0, help="Max ray-cast distance.")
parser.add_argument("--obstacle_ray_fov_deg", type=float, default=180.0, help="Ray-cast field of view.")
parser.add_argument("--curriculum_max_obstacles", type=int, default=4, help="Maximum number of obstacles to reveal.")

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from torch.utils.tensorboard import SummaryWriter

from rsl_rl.algorithms.p3o_eco import P3O_ECO
from rsl_rl.modules import ActorCriticSafe


def compute_cbf_cost(env, cfg, stage_progress: float):
    """Compute obstacle-avoidance cost using the same ray-cast model as the observation."""
    from unitree_rl_lab.tasks.locomotion import mdp

    robot = env.scene["robot"]
    robot_vel_xy = robot.data.root_lin_vel_w[:, :2]

    distances, directions = mdp.closest_obstacle_raycast(
        env,
        num_rays=cfg.obstacle_num_rays,
        max_distance=cfg.obstacle_ray_max_distance,
        fov_deg=cfg.obstacle_ray_fov_deg,
    )

    h = distances - cfg.safety_margin
    h_dot = -torch.sum(robot_vel_xy * directions, dim=-1)
    cbf_violation = torch.relu(-(h_dot + cfg.cbf_gamma * h))
    distance_violation = torch.relu(cfg.safety_margin - distances)
    projected_gravity = robot.data.projected_gravity_b
    tilt_cost = torch.relu(torch.abs(projected_gravity[:, 0]) - 0.6)
    tilt_cost += torch.relu(torch.abs(projected_gravity[:, 1]) - 0.6)
    height_cost = torch.relu(0.45 - robot.data.root_pos_w[:, 2])
    speed = torch.linalg.norm(robot_vel_xy, dim=-1)
    frozen_cost = torch.relu(0.15 - speed)

    # Keep the same CBF structure in both stages, but start with a softer constraint
    # so the policy does not collapse to standing still.
    if cfg.stage == "walk":
        cbf_weight = 0.25 + 0.75 * stage_progress
        distance_weight = 0.10 + 0.40 * stage_progress
        collision_weight = 2.0 + 3.0 * stage_progress
        posture_weight = 2.0
        frozen_weight = 0.20
    else:
        cbf_weight = 0.80 + 0.20 * stage_progress
        distance_weight = 0.60 + 0.90 * stage_progress
        collision_weight = 6.0 + 4.0 * stage_progress
        posture_weight = 1.5
        frozen_weight = 0.10

    collision_penalty = (distances < cfg.collision_distance).float() * collision_weight
    total_cost = (
        cbf_weight * cbf_violation
        + distance_weight * distance_violation
        + collision_penalty
        + posture_weight * tilt_cost
        + posture_weight * height_cost
        + frozen_weight * frozen_cost
    )

    info = {
        "cbf/distance_mean": distances.mean().item(),
        "cbf/distance_min": distances.min().item(),
        "cbf/h_mean": h.mean().item(),
        "cbf/h_min": h.min().item(),
        "cbf/h_dot_mean": h_dot.mean().item(),
        "cbf/violation_rate": (cbf_violation > 0).float().mean().item(),
        "cbf/collision_rate": (distances < cfg.collision_distance).float().mean().item(),
        "cbf/frozen_rate": (speed < 0.15).float().mean().item(),
        "cbf/cbf_weight": cbf_weight,
        "cbf/distance_weight": distance_weight,
    }
    return total_cost, info


def compute_tracking_metrics(env):
    """Compute simple command tracking metrics for debugging and curriculum tuning."""
    from unitree_rl_lab.tasks.locomotion import mdp

    robot = env.scene["robot"]
    command = mdp.generated_commands(env, command_name="base_velocity")
    actual_lin = robot.data.root_lin_vel_b[:, :2]
    actual_yaw = robot.data.root_ang_vel_b[:, 2]
    cmd_lin = command[:, :2]
    cmd_yaw = command[:, 2]

    lin_error = torch.linalg.norm(actual_lin - cmd_lin, dim=-1)
    yaw_error = torch.abs(actual_yaw - cmd_yaw)
    actual_speed = torch.linalg.norm(actual_lin, dim=-1)
    cmd_speed = torch.linalg.norm(cmd_lin, dim=-1)

    return {
        "track/lin_error": lin_error.mean().item(),
        "track/yaw_error": yaw_error.mean().item(),
        "track/actual_speed": actual_speed.mean().item(),
        "track/command_speed": cmd_speed.mean().item(),
    }


def configure_stage(env_cfg, stage: str):
    """Adjust reward emphasis without changing the environment/task definition."""
    cmd_cfg = env_cfg.commands.base_velocity

    if stage == "walk":
        env_cfg.num_obstacles = 2
        env_cfg.min_obstacle_distance = 4.0
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
    else:
        env_cfg.num_obstacles = max(2, args_cli.curriculum_max_obstacles)
        env_cfg.min_obstacle_distance = 2.5
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


def get_stage_progress(iteration: int, max_iterations: int, stage: str) -> float:
    """Return a curriculum progress value in [0, 1]."""
    if max_iterations <= 1:
        return 1.0

    raw_progress = iteration / float(max_iterations - 1)
    if stage == "walk":
        return min(1.0, raw_progress / 0.6)
    return raw_progress


def apply_p3o_schedule(alg: P3O_ECO, cfg, stage_progress: float):
    """Relax the safe-RL constraint early, then tighten it as the gait improves."""
    if cfg.stage == "walk":
        alg.kappa = max(0.1, cfg.kappa * (0.2 + 0.4 * stage_progress))
        alg.cost_limit = cfg.cost_limit * (2.5 - 0.5 * stage_progress)
    else:
        alg.kappa = max(0.1, cfg.kappa * (0.4 + 0.6 * stage_progress))
        alg.cost_limit = cfg.cost_limit * (1.8 - 0.8 * stage_progress)


def apply_obstacle_curriculum(obstacle_manager, cfg, stage_progress: float) -> int:
    """Increase the number of active obstacles over training."""
    max_active = min(cfg.curriculum_max_obstacles, obstacle_manager.max_num_obstacles)
    if cfg.stage == "walk":
        active = 1 if stage_progress < 0.55 else min(2, max_active)
    else:
        if max_active <= 2:
            active = max_active
        elif stage_progress < 0.30:
            active = 2
        elif stage_progress < 0.65:
            active = min(3, max_active)
        else:
            active = max_active
    obstacle_manager.set_active_obstacles(active)
    return active


def get_eval_scenario(iteration: int, stage: str) -> str:
    """Pick a representative evaluation scene for this checkpoint."""
    if stage == "walk":
        return "front_pillars"
    if iteration < 4000:
        return "front_pillars"
    if iteration < 8000:
        return "dense_front"
    return "slalom"


def maybe_record_eval(checkpoint_path: str, iteration: int, cfg):
    """Record an evaluation video and metrics for a saved checkpoint."""
    if not cfg.auto_eval:
        return

    eval_script = os.path.join(os.path.dirname(__file__), "eval_g1_p3o_cbf_fixed.py")
    video_name = f"{iteration // cfg.save_interval:02d}.mp4"
    scenario = get_eval_scenario(iteration, cfg.stage)
    cmd = [
        sys.executable,
        eval_script,
        "--checkpoint",
        checkpoint_path,
        "--steps",
        str(cfg.eval_steps),
        "--num_envs",
        str(cfg.eval_num_envs),
        "--scenario",
        scenario,
        "--name",
        video_name,
        "--headless",
    ]
    print(f"[INFO] Auto-eval: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, cwd=os.getcwd())
    except subprocess.CalledProcessError as exc:
        print(f"[WARN] Auto-eval failed for {checkpoint_path}: exit code {exc.returncode}")


def main():
    import unitree_rl_lab.tasks  # noqa: F401
    from unitree_rl_lab.tasks.locomotion.obstacle_manager import ObstacleManager
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    task_name = "Unitree-G1-29dof-ObstacleAvoidance"
    env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.safety_margin = args_cli.safety_margin
    env_cfg.cbf_gamma = args_cli.cbf_gamma
    env_cfg.collision_distance = args_cli.collision_distance
    env_cfg.obstacle_num_rays = args_cli.obstacle_num_rays
    env_cfg.obstacle_ray_max_distance = args_cli.obstacle_ray_max_distance
    env_cfg.obstacle_ray_fov_deg = args_cli.obstacle_ray_fov_deg
    configure_stage(env_cfg, args_cli.stage)

    env = gym.make(task_name, cfg=env_cfg)
    env.unwrapped.obstacle_manager = ObstacleManager(env.unwrapped, env.unwrapped.cfg)
    env.unwrapped.obstacle_manager.spawn_obstacles()

    device = args_cli.device
    num_envs = env.unwrapped.num_envs

    print(f"[INFO] Training with {num_envs} environments on {device}")
    print(f"[INFO] Stage: {args_cli.stage}")
    print(f"[INFO] Spawned {env.unwrapped.cfg.num_obstacles} pillar obstacles per environment")

    num_obs = env.unwrapped.observation_space["policy"].shape[-1]
    num_privileged_obs = env.unwrapped.observation_space["critic"].shape[-1]
    num_actions = env.unwrapped.action_space.shape[-1]

    policy = ActorCriticSafe(
        num_actor_obs=num_obs,
        num_critic_obs=num_privileged_obs,
        num_actions=num_actions,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
    ).to(device)

    if args_cli.resume:
        checkpoint = torch.load(args_cli.resume, map_location=device)
        policy.load_state_dict(checkpoint["policy_state_dict"])
        print(f"[INFO] Loaded checkpoint: {args_cli.resume}")

    alg = P3O_ECO(
        actor_critic=policy,
        num_learning_epochs=5,
        num_mini_batches=4,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.01,
        learning_rate=1e-3,
        cost_critic_learning_rate=1e-5,
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

    log_dir = os.path.join("logs", f"p3o_g1_cbf_{args_cli.stage}", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] Logging to: {log_dir}")
    writer = SummaryWriter(log_dir=log_dir, flush_secs=10)

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
    current_obs = obs_dict["policy"].to(device)
    current_privileged_obs = obs_dict["critic"].to(device)

    episode_costs = []
    episode_rewards = []
    cbf_info = {}
    tracking_info = {}

    for iteration in range(args_cli.max_iterations):
        stage_progress = get_stage_progress(iteration, args_cli.max_iterations, args_cli.stage)
        apply_p3o_schedule(alg, args_cli, stage_progress)
        active_obstacles = apply_obstacle_curriculum(env.unwrapped.obstacle_manager, args_cli, stage_progress)

        for _ in range(num_steps_per_env):
            with torch.no_grad():
                actions = alg.act(current_obs, current_privileged_obs)

            step_result = env.step(actions)
            next_obs_dict = step_result[0]
            rewards = step_result[1].to(device)
            dones = step_result[2].to(device)
            infos = step_result[3] if len(step_result) > 3 else {}

            costs, cbf_info = compute_cbf_cost(env.unwrapped, args_cli, stage_progress)
            tracking_info = compute_tracking_metrics(env.unwrapped)
            costs = costs.to(device)

            episode_costs.append(costs.mean().item())
            episode_rewards.append(rewards.mean().item())

            next_obs = next_obs_dict["policy"].to(device)
            next_privileged_obs = next_obs_dict["critic"].to(device)

            alg.process_env_step(rewards, costs, dones, infos)

            current_obs = next_obs.clone()
            current_privileged_obs = next_privileged_obs.clone()

        alg.compute_returns(current_privileged_obs)
        loss_dict = alg.update()

        if iteration % 10 == 0:
            avg_cost = np.mean(episode_costs) if episode_costs else 0.0
            avg_reward = np.mean(episode_rewards) if episode_rewards else 0.0

            writer.add_scalar("train/reward", avg_reward, iteration)
            writer.add_scalar("train/cost", avg_cost, iteration)
            writer.add_scalar("train/stage_progress", stage_progress, iteration)
            writer.add_scalar("train/p3o_kappa", alg.kappa, iteration)
            writer.add_scalar("train/p3o_cost_limit", alg.cost_limit, iteration)
            writer.add_scalar("curriculum/active_obstacles", active_obstacles, iteration)
            for key, value in cbf_info.items():
                writer.add_scalar(key, value, iteration)
            for key, value in tracking_info.items():
                writer.add_scalar(key, value, iteration)
            writer.add_scalar("loss/value_function", loss_dict.get("value_function", 0.0), iteration)
            writer.add_scalar("loss/cost_value", loss_dict.get("cost_value", 0.0), iteration)
            writer.add_scalar("loss/penalty", loss_dict.get("penalty", 0.0), iteration)

            print(f"\nIteration {iteration}/{args_cli.max_iterations}")
            print(f"  Reward: {avg_reward:.3f}")
            print(f"  Cost: {avg_cost:.3f}")
            print(f"  CBF Distance Mean: {cbf_info.get('cbf/distance_mean', 0.0):.3f}")
            print(f"  CBF Distance Min: {cbf_info.get('cbf/distance_min', 0.0):.3f}")
            print(f"  CBF h_mean: {cbf_info.get('cbf/h_mean', 0.0):.3f}")
            print(f"  CBF violation_rate: {cbf_info.get('cbf/violation_rate', 0.0):.3f}")
            print(f"  Collision rate: {cbf_info.get('cbf/collision_rate', 0.0):.3f}")
            print(f"  Frozen rate: {cbf_info.get('cbf/frozen_rate', 0.0):.3f}")
            print(f"  Cmd speed: {tracking_info.get('track/command_speed', 0.0):.3f}")
            print(f"  Actual speed: {tracking_info.get('track/actual_speed', 0.0):.3f}")
            print(f"  Lin vel error: {tracking_info.get('track/lin_error', 0.0):.3f}")
            print(f"  Value Loss: {loss_dict.get('value_function', 0):.4f}")
            print(f"  Cost Value Loss: {loss_dict.get('cost_value', 0):.4f}")
            print(f"  Penalty: {loss_dict.get('penalty', 0):.4f}")
            episode_costs = []
            episode_rewards = []

        if iteration % args_cli.save_interval == 0 and iteration > 0:
            checkpoint_path = os.path.join(log_dir, f"model_{iteration}.pt")
            torch.save(
                {
                    "iteration": iteration,
                    "policy_state_dict": alg.actor_critic.state_dict(),
                    "optimizer_state_dict": alg.optimizer.state_dict(),
                    "cost_critic_optimizer_state_dict": alg.cost_value_optimizer.state_dict(),
                },
                checkpoint_path,
            )
            print(f"[INFO] Saved checkpoint to {checkpoint_path}")
            maybe_record_eval(checkpoint_path, iteration, args_cli)

    final_path = os.path.join(log_dir, "model_final.pt")
    torch.save(
        {
            "iteration": args_cli.max_iterations,
            "policy_state_dict": alg.actor_critic.state_dict(),
            "optimizer_state_dict": alg.optimizer.state_dict(),
            "cost_critic_optimizer_state_dict": alg.cost_value_optimizer.state_dict(),
        },
        final_path,
    )
    print(f"[INFO] Training complete. Final model saved to {final_path}")
    maybe_record_eval(final_path, args_cli.max_iterations, args_cli)

    writer.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
