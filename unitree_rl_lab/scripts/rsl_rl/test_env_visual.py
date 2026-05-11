"""Visualize the obstacle avoidance environment."""

import argparse
import sys
import torch
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to show")
parser.add_argument("--steps", type=int, default=1000, help="Steps to run")
args_cli = parser.parse_args()

# Launch with GUI
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

# Import after Isaac Sim is initialized
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../source/unitree_rl_lab'))
import unitree_rl_lab.tasks  # noqa: F401

# Import ObstacleManager
from unitree_rl_lab.tasks.locomotion.obstacle_manager import ObstacleManager


def main():
    print("[INFO] Creating environment...")

    # Load environment configuration
    env_cfg = load_cfg_from_registry("Unitree-G1-29dof-ObstacleAvoidance", "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = "cuda:0"

    # Create environment
    env = gym.make("Unitree-G1-29dof-ObstacleAvoidance", cfg=env_cfg)

    print(f"[INFO] Environment created with {args_cli.num_envs} robots")
    print(f"[INFO] Number of obstacles: {env.unwrapped.cfg.num_obstacles}")
    print(f"[INFO] Safety margin: {env.unwrapped.cfg.safety_margin}m")

    # Initialize obstacle manager
    obstacle_manager = ObstacleManager(env.unwrapped, env.unwrapped.cfg)
    obstacle_manager.spawn_obstacles()

    print(f"[INFO] Obstacle positions (first env):")
    if len(obstacle_manager.obstacle_positions) > 0:
        print(obstacle_manager.obstacle_positions[0])

    # Reset environment
    obs, _ = env.reset()
    if isinstance(obs, dict):
        print(f"[INFO] Observation keys: {obs.keys()}")
        print(f"[INFO] Policy obs shape: {obs['policy'].shape}")
    else:
        print(f"[INFO] Observation shape: {obs.shape}")
    print(f"[INFO] Action space: {env.action_space}")

    # Run a few steps with random actions
    print("[INFO] Running simulation with random actions...")
    num_actions = env.unwrapped.action_space.shape[1]
    device = env.unwrapped.device
    for step in range(args_cli.steps):
        # Random actions
        actions = torch.randn(args_cli.num_envs, num_actions, device=device)

        # Step
        step_result = env.step(actions)
        obs = step_result[0]
        rewards = step_result[1]
        dones = step_result[2]
        infos = step_result[3] if len(step_result) > 3 else {}

        # Get robot positions and compute CBF cost
        robot_pos = env.unwrapped.scene["robot"].data.root_pos_w
        distances, directions = obstacle_manager.get_closest_obstacle(robot_pos[:, :2])

        h = distances - env.unwrapped.cfg.safety_margin
        cbf_cost = torch.relu(-h)

        if step % 100 == 0:
            print(f"\nStep {step}:")
            print(f"  Robot positions: {robot_pos[0, :2].cpu().numpy()}")
            print(f"  Closest obstacle distance: {distances[0].item():.3f}m")
            print(f"  CBF h: {h[0].item():.3f}")
            print(f"  CBF cost: {cbf_cost[0].item():.3f}")
            print(f"  Reward: {rewards[0].item():.3f}")

        # Sleep to make visualization visible
        time.sleep(0.01)

    print("\n[INFO] Test complete!")
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
