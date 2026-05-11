# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train G1 walking with P3O (ECO version) with raycast visualization."""

import argparse
import sys
import os
import torch
import numpy as np
from datetime import datetime

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train G1 walking with P3O.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments.")
parser.add_argument("--max_iterations", type=int, default=20000, help="Training iterations.")
parser.add_argument("--save_interval", type=int, default=1000, help="Save checkpoint every N iterations.")
parser.add_argument("--headless", action="store_true", default=False, help="Run in headless mode.")
parser.add_argument("--device", type=str, default="cuda:0", help="Device to use.")

# P3O specific
parser.add_argument("--cost_limit", type=float, default=25.0, help="Cost limit for P3O.")
parser.add_argument("--kappa", type=float, default=1.0, help="Penalty coefficient.")
parser.add_argument("--cost_gamma", type=float, default=0.99, help="Cost discount factor.")
parser.add_argument("--cost_lam", type=float, default=0.95, help="Cost GAE lambda.")

args_cli = parser.parse_args()

# launch omniverse app (without headless - GUI enabled)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import gymnasium as gym
import isaaclab_tasks  # noqa: F401

from rsl_rl.algorithms.p3o_eco import P3O_ECO
from rsl_rl.modules import ActorCriticSafe


def compute_safety_cost(env, robot_pos, robot_vel, actions, cfg):
    """
    Compute safety cost for walking task.

    For walking, we use simple stability-based costs:
    - Cost 1: Base tilt (orientation)
    - Cost 2: Low base height
    - Cost 3: High joint torques
    """
    # Get robot data
    robot = env.scene["robot"]

    # Cost 1: Base tilt (from projected gravity)
    projected_gravity = robot.data.projected_gravity_b
    base_tilt_cost = torch.relu(torch.abs(projected_gravity[:, 0]) - 0.3)  # Roll
    base_tilt_cost += torch.relu(torch.abs(projected_gravity[:, 1]) - 0.3)  # Pitch

    # Cost 2: Low base height
    base_height = robot.data.root_pos_w[:, 2]
    height_cost = torch.relu(0.5 - base_height)

    # Cost 3: High joint torques (energy cost)
    joint_torques = torch.abs(robot.data.applied_torque)
    torque_cost = torch.relu(joint_torques - 50.0).sum(dim=-1) * 0.01

    # Combine costs
    total_cost = base_tilt_cost * 5.0 + height_cost * 2.0 + torque_cost

    info = {
        'cost/base_tilt': base_tilt_cost.mean().item(),
        'cost/height': height_cost.mean().item(),
        'cost/torque': torque_cost.mean().item(),
    }

    return total_cost, info


def main():
    """Main training function."""

    # Import after Isaac Sim is initialized
    import unitree_rl_lab.tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    # Load environment configuration
    env_cfg = load_cfg_from_registry("Unitree-G1-29dof-Velocity", "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    # Create environment
    env = gym.make("Unitree-G1-29dof-Velocity", cfg=env_cfg)

    device = args_cli.device
    num_envs = env.unwrapped.num_envs

    print(f"[INFO] Training with {num_envs} environments on {device}")
    print(f"[INFO] GUI enabled - Raycast visualization active!")

    # Get observation shapes
    num_obs = env.unwrapped.observation_space['policy'].shape[-1]
    num_privileged_obs = env.unwrapped.observation_space['critic'].shape[-1]
    num_actions = env.unwrapped.action_space.shape[-1]

    # Create policy networks (ActorCriticSafe with cost_critic)
    policy = ActorCriticSafe(
        num_actor_obs=num_obs,
        num_critic_obs=num_privileged_obs,
        num_actions=num_actions,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
    ).to(device)

    # Create P3O_ECO algorithm
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

    # Setup logging
    log_dir = os.path.join(
        "logs", "p3o_g1_walking",
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] Logging to: {log_dir}")

    # Initialize storage
    # Use short rollout length for faster training (like PPO)
    num_steps_per_env = 24  # Same as standard PPO config
    alg.init_storage(
        num_envs=num_envs,
        num_transitions_per_env=num_steps_per_env,
        actor_obs_shape=[num_obs],
        critic_obs_shape=[num_privileged_obs],
        action_shape=[num_actions],
    )

    # Training loop
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs_dict = reset_result[0]
    else:
        obs_dict = reset_result
    obs = obs_dict['policy'].to(device)
    privileged_obs = obs_dict['critic'].to(device)

    current_obs = obs.clone()
    current_privileged_obs = privileged_obs.clone()

    # Track metrics for logging
    episode_costs = []
    episode_rewards = []

    for iteration in range(args_cli.max_iterations):
        # Collect rollouts
        for step in range(num_steps_per_env):
            with torch.no_grad():
                actions = alg.act(current_obs, current_privileged_obs)

            step_result = env.step(actions)
            next_obs_dict = step_result[0]
            rewards = step_result[1]
            dones = step_result[2]
            infos = step_result[3] if len(step_result) > 3 else {}

            # Get robot position for cost computation
            robot_pos = env.unwrapped.scene["robot"].data.root_pos_w
            robot_vel = env.unwrapped.scene["robot"].data.root_lin_vel_w

            # Compute safety costs
            costs, cost_info = compute_safety_cost(
                env.unwrapped,
                robot_pos,
                robot_vel,
                actions,
                args_cli
            )
            costs = costs.to(device)

            # Store metrics
            episode_costs.append(costs.mean().item())
            episode_rewards.append(rewards.mean().item())

            # Move to device
            next_obs = next_obs_dict['policy'].to(device)
            next_privileged_obs = next_obs_dict['critic'].to(device)
            rewards = rewards.to(device)
            dones = dones.to(device)

            # Process step with costs
            alg.process_env_step(rewards, costs, dones, infos)

            # Update current observations
            current_obs = next_obs.clone()
            current_privileged_obs = next_privileged_obs.clone()

        # Compute returns and update
        alg.compute_returns(current_privileged_obs)
        loss_dict = alg.update()

        # Log
        if iteration % 10 == 0:
            avg_cost = np.mean(episode_costs) if episode_costs else 0
            avg_reward = np.mean(episode_rewards) if episode_rewards else 0
            print(f"\nIteration {iteration}/{args_cli.max_iterations}")
            print(f"  Reward: {avg_reward:.3f}")
            print(f"  Cost: {avg_cost:.3f}")
            print(f"  Value Loss: {loss_dict.get('value_function', 0):.4f}")
            print(f"  Cost Value Loss: {loss_dict.get('cost_value', 0):.4f}")
            print(f"  Penalty: {loss_dict.get('penalty', 0):.4f}")
            episode_costs = []
            episode_rewards = []

        # Save checkpoint
        if iteration % args_cli.save_interval == 0 and iteration > 0:
            checkpoint_path = os.path.join(log_dir, f"model_{iteration}.pt")
            torch.save({
                'iteration': iteration,
                'policy_state_dict': alg.actor_critic.state_dict(),
                'optimizer_state_dict': alg.optimizer.state_dict(),
                'cost_critic_optimizer_state_dict': alg.cost_value_optimizer.state_dict(),
            }, checkpoint_path)
            print(f"[INFO] Saved checkpoint to {checkpoint_path}")

    # Final save
    final_path = os.path.join(log_dir, "model_final.pt")
    torch.save({
        'iteration': args_cli.max_iterations,
        'policy_state_dict': alg.actor_critic.state_dict(),
        'optimizer_state_dict': alg.optimizer.state_dict(),
        'cost_critic_optimizer_state_dict': alg.cost_value_optimizer.state_dict(),
    }, final_path)
    print(f"[INFO] Training complete. Final model saved to {final_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
