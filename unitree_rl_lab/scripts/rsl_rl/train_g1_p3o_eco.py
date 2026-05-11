# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train G1 with P3O (ECO version) for obstacle avoidance."""

import argparse
import sys
import os
import torch
import numpy as np
from datetime import datetime

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train G1 with P3O (ECO).")
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

# CBF specific
parser.add_argument("--safety_margin", type=float, default=0.5, help="CBF safety margin D_min.")

args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import gymnasium as gym
import isaaclab_tasks  # noqa: F401

from rsl_rl.algorithms.p3o_eco import P3O_ECO
from rsl_rl.modules import ActorCriticSafe


def compute_cbf_cost(env, obstacle_manager, robot_pos, robot_vel, actions, cfg):
    """Compute CBF-based cost."""
    # Get closest obstacle distance and direction
    distances, directions = obstacle_manager.get_closest_obstacle(robot_pos[:, :2])

    # Barrier function: h = distance - D_min
    h = distances - cfg.safety_margin

    # CBF cost is the violation of safety condition
    cbf_cost = torch.relu(-h)

    # Additional penalty for very close proximity (collision avoidance)
    collision_penalty = (distances < 0.1).float() * 10.0

    total_cost = cbf_cost + collision_penalty

    # Log CBF metrics
    info = {
        'cbf/h_mean': h.mean().item(),
        'cbf/h_min': h.min().item(),
        'cbf/distance_mean': distances.mean().item(),
        'cbf/distance_min': distances.min().item(),
        'cbf/violation_rate': (h < 0).float().mean().item(),
    }

    return total_cost, info


def main():
    """Main training function."""

    # Import after Isaac Sim is initialized
    import unitree_rl_lab.tasks  # noqa: F401
    from unitree_rl_lab.tasks.locomotion.obstacle_manager import ObstacleManager
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    # Load environment configuration
    env_cfg = load_cfg_from_registry("Unitree-G1-29dof-ObstacleAvoidance", "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    # Create environment
    env = gym.make("Unitree-G1-29dof-ObstacleAvoidance", cfg=env_cfg)

    device = args_cli.device
    num_envs = env.unwrapped.num_envs

    print(f"[INFO] Training with {num_envs} environments on {device}")

    # Initialize obstacle manager
    obstacle_manager = ObstacleManager(env.unwrapped, env.unwrapped.cfg)
    obstacle_manager.spawn_obstacles()

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
        "logs", "p3o_eco",
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] Logging to: {log_dir}")

    # Initialize storage
    max_episode_length = int(env.unwrapped.max_episode_length)
    alg.init_storage(
        num_envs=num_envs,
        num_transitions_per_env=max_episode_length,
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

    # Track episode costs for logging
    episode_costs = []

    for iteration in range(args_cli.max_iterations):
        # Collect rollouts
        for step in range(max_episode_length):
            with torch.no_grad():
                actions = alg.act(current_obs, current_privileged_obs)

            step_result = env.step(actions)
            next_obs_dict = step_result[0]
            rewards = step_result[1]
            dones = step_result[2]
            infos = step_result[3] if len(step_result) > 3 else {}

            # Get robot position for CBF cost computation
            robot_pos = env.unwrapped.scene["robot"].data.root_pos_w
            robot_vel = env.unwrapped.scene["robot"].data.root_lin_vel_w

            # Compute CBF-based costs
            costs, cbf_info = compute_cbf_cost(
                env.unwrapped,
                obstacle_manager,
                robot_pos,
                robot_vel,
                actions,
                args_cli
            )
            costs = costs.to(device)

            # Store episode costs
            episode_costs.append(costs.mean().item())

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
            print(f"\nIteration {iteration}/{args_cli.max_iterations}")
            print(f"  Reward: {alg.storage.rewards.mean().item():.3f}")
            print(f"  Cost: {avg_cost:.3f}")
            print(f"  Value Loss: {loss_dict.get('value_function', 0):.4f}")
            print(f"  Cost Value Loss: {loss_dict.get('cost_value', 0):.4f}")
            print(f"  Penalty: {loss_dict.get('penalty', 0):.4f}")
            print(f"  CBF h_mean: {cbf_info.get('cbf/h_mean', 0):.3f}")
            print(f"  CBF violation_rate: {cbf_info.get('cbf/violation_rate', 0):.3f}")
            episode_costs = []

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
