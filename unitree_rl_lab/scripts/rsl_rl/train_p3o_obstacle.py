# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train G1 robot for obstacle avoidance using P3O with CBF-based cost."""

"""Launch Isaac Sim Simulator first."""
import argparse
import sys
import os

# add argparse arguments
parser = argparse.ArgumentParser(description="Train G1 obstacle avoidance with P3O.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--max_iterations", type=int, default=1000, help="RL Policy training iterations.")
parser.add_argument("--headless", action="store_true", default=False, help="Run in headless mode.")
parser.add_argument("--device", type=str, default="cuda:0", help="Device to use.")

# P3O specific arguments
parser.add_argument("--cost_limit", type=float, default=25.0, help="Cost limit for P3O (CBF barrier).")
parser.add_argument("--kappa", type=float, default=1.0, help="Penalty coefficient for P3O.")
parser.add_argument("--cost_gamma", type=float, default=0.99, help="Cost discount factor for P3O.")
parser.add_argument("--cost_lam", type=float, default=0.95, help="Cost GAE lambda for P3O.")

args_cli = parser.parse_args()

# launch omniverse app
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.dict import print_dict

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401

from rsl_rl.algorithms import P3O
from rsl_rl.modules import ActorCritic
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.storage import RolloutStorage


def compute_cbf_cost(env, gamma=0.1):
    """
    Compute CBF-based cost from obstacle distances.

    CBF: h(x) >= 0 defines safe set
    Cost = max(0, -h(x)) where h(x) is the distance to nearest obstacle

    Using exponential CBF: cost = exp(-gamma * distance)
    This penalizes being close to obstacles exponentially.
    """
    robot_pos = env.scene["robot"].data.root_pos_w[:, :2]  # (N, 2)

    # Get lidar scan (obstacle distances)
    if "lidar" in env.scene.sensors:
        lidar = env.scene.sensors["lidar"]
        # Get ray hit distances
        distances = lidar.data.ray_hits_w[..., 2]  # z-coordinate or distance

        # Find minimum distance to obstacle for each env
        min_dist, _ = torch.min(distances, dim=-1)
        min_dist = torch.clamp(min_dist, min=0.01)
    else:
        # Fallback: use random obstacles at fixed positions
        # Define some obstacle positions
        obstacle_positions = torch.tensor([
            [2.0, 0.0], [-2.0, 0.0], [0.0, 2.0], [0.0, -2.0],
            [3.0, 3.0], [-3.0, -3.0], [3.0, -3.0], [-3.0, 3.0],
        ], device=robot_pos.device)

        # Compute distance to all obstacles
        distances = torch.cdist(robot_pos.unsqueeze(0), obstacle_positions.unsqueeze(0)).squeeze(0)
        min_dist, _ = torch.min(distances, dim=-1)

    # CBF-based cost: exponential barrier
    # cost = exp(-gamma * distance) - higher cost when closer to obstacle
    cbf_cost = torch.exp(-gamma * min_dist)

    # Additional cost for very close proximity (hard constraint violation)
    proximity_violation = (min_dist < 0.5).float() * 10.0

    total_cost = cbf_cost + proximity_violation

    return total_cost


def compute_obstacle_penalty(env):
    """Compute additional penalty for being too close to obstacles."""
    robot_pos = env.scene["robot"].data.root_pos_w[:, :2]

    # Define obstacle positions (circular obstacles)
    obstacle_positions = torch.tensor([
        [2.0, 0.0], [-2.0, 0.0], [0.0, 2.0], [0.0, -2.0],
        [3.0, 3.0], [-3.0, -3.0], [3.0, -3.0], [-3.0, 3.0],
        [1.5, 1.5], [-1.5, -1.5], [1.5, -1.5], [-1.5, 1.5],
    ], device=robot_pos.device)

    # Compute distance to nearest obstacle
    distances = torch.cdist(robot_pos.unsqueeze(0), obstacle_positions.unsqueeze(0)).squeeze(0)
    min_dist, _ = torch.min(distances, dim=-1)

    # CBF: h(x) = distance - safety_margin
    safety_margin = 0.5
    h = min_dist - safety_margin

    # Cost is violation of CBF: max(0, -h)
    cbf_violation = torch.relu(-h)

    return cbf_violation


def main():
    """Main training function."""

    # Create environment
    env_cfg_entry_point = "unitree_rl_lab.tasks.locomotion.robots.g1.29dof.obstacle_avoidance_env_cfg:ObstacleAvoidanceEnvCfg"

    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()

    with initialize(version_base=None, config_path="../../../source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof"):
        cfg = compose(config_name="obstacle_avoidance_env_cfg")

    env = gym.make(
        "Unitree-G1-29dof-ObstacleAvoidance",
        cfg=cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )

    # Override num_envs if specified
    if args_cli.num_envs is not None:
        env.unwrapped.scene.num_envs = args_cli.num_envs

    device = args_cli.device
    num_envs = env.unwrapped.num_envs

    # Create policy networks
    obs_shape = env.unwrapped.num_obs
    privileged_obs_shape = env.unwrapped.num_privileged_obs
    actions_shape = env.unwrapped.num_actions

    policy = ActorCritic(
        obs_shape=obs_shape,
        privileged_obs_shape=privileged_obs_shape,
        actions_shape=actions_shape,
        initial_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    ).to(device)

    # Create cost critic (same architecture)
    cost_critic = ActorCritic(
        obs_shape=obs_shape,
        privileged_obs_shape=privileged_obs_shape,
        actions_shape=actions_shape,
        initial_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    ).to(device)

    # Create P3O algorithm
    alg = P3O(
        policy=policy,
        cost_critic=cost_critic,
        num_learning_epochs=5,
        num_mini_batches=4,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.01,
        learning_rate=1e-3,
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

    # Setup logging
    log_dir = os.path.join(
        "logs", "p3o_obstacle",
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] Logging to: {log_dir}")

    # Initialize storage
    max_episode_length = int(env.unwrapped.max_episode_length)
    alg.init_storage(
        training_type="safe_rl",
        num_envs=num_envs,
        num_transitions_per_env=max_episode_length,
        actor_obs_shape=[obs_shape],
        critic_obs_shape=[privileged_obs_shape],
        actions_shape=[actions_shape],
    )

    # Training loop
    obs, privileged_obs = env.reset()
    obs = obs.to(device)
    privileged_obs = privileged_obs.to(device)

    current_obs = obs.clone()
    current_privileged_obs = privileged_obs.clone() if privileged_obs is not None else obs.clone()

    for iteration in range(args_cli.max_iterations):
        # Collect rollouts
        for step in range(max_episode_length):
            with torch.no_grad():
                actions = alg.act(current_obs, current_privileged_obs)

            next_obs, rewards, dones, infos = env.step(actions)

            # Compute CBF-based costs
            costs = compute_cbf_cost(env.unwrapped, gamma=0.5).to(device)

            # Move to device
            next_obs = next_obs.to(device)
            rewards = rewards.to(device)
            costs = costs.to(device)
            dones = dones.to(device)

            # Process step
            alg.process_env_step(rewards, costs, dones, infos)

            # Update current observations
            current_obs = next_obs.clone()
            current_privileged_obs = next_obs.clone()  # Using same obs for critic

            # Reset if done
            if dones.any():
                reset_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                if len(reset_ids) > 0:
                    reset_obs, reset_privileged_obs = env.reset_idx(reset_ids)
                    current_obs[reset_ids] = reset_obs[reset_ids].to(device)

        # Compute returns and update
        alg.compute_returns(current_privileged_obs)
        loss_dict = alg.update()

        # Log
        if iteration % 10 == 0:
            print(f"Iteration {iteration}/{args_cli.max_iterations}")
            print(f"  Reward: {alg.storage.rewards.mean().item():.3f}")
            print(f"  Cost: {alg.storage.costs.mean().item():.3f}")
            print(f"  Value Loss: {loss_dict.get('value_function', 0):.4f}")
            print(f"  Cost Value Loss: {loss_dict.get('cost_value', 0):.4f}")
            print(f"  Penalty: {loss_dict.get('penalty', 0):.4f}")

        # Save checkpoint
        if iteration % 100 == 0 and iteration > 0:
            checkpoint_path = os.path.join(log_dir, f"model_{iteration}.pt")
            torch.save({
                'iteration': iteration,
                'policy_state_dict': alg.policy.state_dict(),
                'cost_critic_state_dict': alg.cost_critic.state_dict(),
                'optimizer_state_dict': alg.optimizer.state_dict(),
            }, checkpoint_path)
            print(f"[INFO] Saved checkpoint to {checkpoint_path}")

    # Final save
    final_path = os.path.join(log_dir, "model_final.pt")
    torch.save({
        'iteration': args_cli.max_iterations,
        'policy_state_dict': alg.policy.state_dict(),
        'cost_critic_state_dict': alg.cost_critic.state_dict(),
        'optimizer_state_dict': alg.optimizer.state_dict(),
    }, final_path)
    print(f"[INFO] Training complete. Final model saved to {final_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
