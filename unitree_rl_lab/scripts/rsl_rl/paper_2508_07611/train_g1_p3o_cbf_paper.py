# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Paper-aligned G1 P3O-CBF training variant for arXiv:2508.07611."""

import argparse
import os
from datetime import datetime

import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train G1 obstacle avoidance with a paper-aligned P3O-CBF setup.")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments.")
parser.add_argument("--max_iterations", type=int, default=15000, help="Training iterations.")
parser.add_argument("--save_interval", type=int, default=2000, help="Save checkpoint every N iterations.")
parser.add_argument("--headless", action="store_true", default=False, help="Run in headless mode.")
parser.add_argument("--device", type=str, default="cuda:0", help="Device to use.")
parser.add_argument("--stage", type=str, default="avoid", choices=["walk", "avoid"], help="Training stage.")
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

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from torch.utils.tensorboard import SummaryWriter

from rsl_rl.modules import ActorCriticSafe

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
    if max_iterations <= 1:
        return 1.0
    raw_progress = iteration / float(max_iterations - 1)
    return min(1.0, raw_progress / 0.6) if stage == "walk" else raw_progress


def apply_p3o_schedule(alg: P3OCBFPaper, cfg, stage_progress: float):
    if cfg.stage == "walk":
        alg.kappa = max(0.2, cfg.kappa * (0.3 + 0.5 * stage_progress))
        alg.cost_limit = max(0.15, cfg.cost_limit * (1.8 - 0.8 * stage_progress))
    else:
        alg.kappa = max(0.5, cfg.kappa * (0.6 + 0.8 * stage_progress))
        alg.cost_limit = max(0.05, cfg.cost_limit * (1.2 - 0.7 * stage_progress))


def apply_obstacle_curriculum(obstacle_manager, cfg, stage_progress: float) -> int:
    max_active = min(cfg.curriculum_max_obstacles, obstacle_manager.max_num_obstacles)
    if cfg.stage == "walk":
        active = 1 if stage_progress < 0.55 else min(2, max_active)
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

    env = gym.make(task_name, cfg=env_cfg)
    env.unwrapped.obstacle_manager = ObstacleManager(env.unwrapped, env.unwrapped.cfg)
    env.unwrapped.obstacle_manager.spawn_obstacles()

    device = args_cli.device
    num_envs = env.unwrapped.num_envs
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
        "logs", f"p3o_g1_cbf_paper_{args_cli.stage}", datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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
    obs_dict = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    current_obs = obs_dict["policy"].to(device)
    current_privileged_obs = obs_dict["critic"].to(device)
    episode_costs = []
    episode_rewards = []
    paper_info = {}
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
            costs, paper_info = compute_paper_cost(env.unwrapped, args_cli)
            tracking_info = compute_tracking_metrics(env.unwrapped)
            costs = costs.to(device)

            episode_rewards.append(rewards.mean().item())
            episode_costs.append(costs.mean().item())
            next_obs = next_obs_dict["policy"].to(device)
            next_privileged_obs = next_obs_dict["critic"].to(device)

            alg.process_env_step(rewards, costs, dones, infos)
            current_obs = next_obs.clone()
            current_privileged_obs = next_privileged_obs.clone()

        alg.compute_returns(current_privileged_obs)
        loss_dict = alg.update()

        if iteration % 10 == 0:
            avg_reward = np.mean(episode_rewards) if episode_rewards else 0.0
            avg_cost = np.mean(episode_costs) if episode_costs else 0.0
            writer.add_scalar("train/reward", avg_reward, iteration)
            writer.add_scalar("train/cost", avg_cost, iteration)
            writer.add_scalar("train/stage_progress", stage_progress, iteration)
            writer.add_scalar("train/p3o_kappa", alg.kappa, iteration)
            writer.add_scalar("train/p3o_cost_limit", alg.cost_limit, iteration)
            writer.add_scalar("curriculum/active_obstacles", active_obstacles, iteration)
            for key, value in paper_info.items():
                writer.add_scalar(key, value, iteration)
            for key, value in tracking_info.items():
                writer.add_scalar(key, value, iteration)
            writer.add_scalar("loss/value_function", loss_dict["value_function"], iteration)
            writer.add_scalar("loss/cost_value", loss_dict["cost_value"], iteration)
            writer.add_scalar("loss/penalty", loss_dict["penalty"], iteration)
            writer.add_scalar("loss/constraint_violation", loss_dict["constraint_violation"], iteration)
            writer.add_scalar("loss/rollout_episode_cost", loss_dict["rollout_episode_cost"], iteration)
            writer.add_scalar("loss/raw_cost_adv_mean", loss_dict["raw_cost_adv_mean"], iteration)
            writer.add_scalar("loss/raw_cost_adv_std", loss_dict["raw_cost_adv_std"], iteration)
            print(
                f"Iteration {iteration}/{args_cli.max_iterations} "
                f"reward={avg_reward:.4f} cost={avg_cost:.4f} "
                f"unsafe_rate={paper_info.get('paper/unsafe_rate', 0.0):.4f} "
                f"cbf_violation_rate={paper_info.get('paper/cbf_violation_rate', 0.0):.4f} "
                f"penalty={loss_dict['penalty']:.4f} Jc={loss_dict['rollout_episode_cost'] - alg.cost_limit:.4f}"
            )
            episode_rewards = []
            episode_costs = []

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
    writer.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
