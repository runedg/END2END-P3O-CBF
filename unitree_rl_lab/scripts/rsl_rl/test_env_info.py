"""Test obstacle avoidance environment (headless)."""

import sys
import os
import torch

from isaaclab.app import AppLauncher

# Launch headless
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

# Import unitree_rl_lab to register environments
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../source/unitree_rl_lab'))
import unitree_rl_lab.tasks  # noqa: F401

# Import ObstacleManager
from unitree_rl_lab.tasks.locomotion.obstacle_manager import ObstacleManager


def main():
    print("=" * 60)
    print("G1 Obstacle Avoidance Environment Test")
    print("=" * 60)

    # Load environment configuration
    print("\n[INFO] Loading environment configuration...")
    env_cfg = load_cfg_from_registry("Unitree-G1-29dof-ObstacleAvoidance", "env_cfg_entry_point")
    env_cfg.scene.num_envs = 4
    env_cfg.sim.device = "cuda:0"

    # Create environment
    print("\n[INFO] Creating environment...")
    env = gym.make("Unitree-G1-29dof-ObstacleAvoidance", cfg=env_cfg)
    print("[INFO] Environment created successfully!")

    # Environment info
    print("\n" + "=" * 60)
    print("Environment Configuration")
    print("=" * 60)
    print(f"  Number of environments: {env.unwrapped.num_envs}")
    print(f"  Observation space: {env.unwrapped.observation_space}")
    print(f"  Action space: {env.unwrapped.action_space}")
    print(f"  Max episode length: {env.unwrapped.max_episode_length}")

    # Obstacle configuration
    print("\n" + "=" * 60)
    print("Obstacle Configuration")
    print("=" * 60)
    cfg = env.unwrapped.cfg
    print(f"  Number of obstacles: {cfg.num_obstacles}")
    print(f"  Obstacle radius: {cfg.obstacle_radius}m")
    print(f"  Obstacle height: {cfg.obstacle_height}m")
    print(f"  Safety margin (D_min): {cfg.safety_margin}m")

    # Initialize obstacle manager
    print("\n[INFO] Initializing obstacle manager...")
    manager = ObstacleManager(env.unwrapped, cfg)
    manager.spawn_obstacles()
    print(f"[INFO] Obstacles spawned for {len(manager.obstacle_positions)} environments")

    # Show obstacle positions for first environment
    print("\n" + "=" * 60)
    print("Obstacle Positions (First Environment)")
    print("=" * 60)
    print("  Format: [x, y, radius]")
    if len(manager.obstacle_positions) > 0:
        for i in range(min(5, len(manager.obstacle_positions[0]))):
            pos = manager.obstacle_positions[0][i].tolist()
            print(f"  Obstacle {i}: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
        if len(manager.obstacle_positions[0]) > 5:
            print(f"  ... and {len(manager.obstacle_positions[0]) - 5} more")

    # Reset environment
    print("\n[INFO] Resetting environment...")
    obs, info = env.reset()
    if isinstance(obs, dict):
        print(f"[INFO] Observation keys: {obs.keys()}")
        print(f"[INFO] Policy observation shape: {obs['policy'].shape}")
        print(f"[INFO] Critic observation shape: {obs['critic'].shape}")
    else:
        print(f"[INFO] Observation shape: {obs.shape}")

    # Get robot positions
    robot_pos = env.unwrapped.scene["robot"].data.root_pos_w
    print("\n" + "=" * 60)
    print("Robot Initial Positions (First 3 Envs)")
    print("=" * 60)
    for i in range(min(3, env.unwrapped.num_envs)):
        pos = robot_pos[i].cpu().numpy()
        print(f"  Env {i}: x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}")

    # Compute CBF statistics
    print("\n" + "=" * 60)
    print("CBF Statistics (Before Simulation)")
    print("=" * 60)
    distances, directions = manager.get_closest_obstacle(robot_pos[:, :2])
    h = distances - cfg.safety_margin
    cbf_cost = torch.relu(-h)

    print(f"  Mean distance to obstacle: {distances.mean():.3f}m")
    print(f"  Min distance to obstacle: {distances.min():.3f}m")
    print(f"  Mean CBF h (h = dist - D_min): {h.mean():.3f}")
    print(f"  Safety violation rate (h < 0): {(h < 0).float().mean():.1%}")
    print(f"  Mean CBF cost: {cbf_cost.mean():.3f}")

    # Run a few steps with random actions
    print("\n" + "=" * 60)
    print("Simulation Test (Random Actions)")
    print("=" * 60)

    num_steps = 10
    total_reward = 0
    total_cost = 0

    num_actions = env.unwrapped.action_space.shape[1]
    for step in range(num_steps):
        actions = torch.randn(env.unwrapped.num_envs, num_actions, device=env.unwrapped.device)
        step_result = env.step(actions)
        obs = step_result[0]
        rewards = step_result[1]
        dones = step_result[2]
        infos = step_result[3] if len(step_result) > 3 else {}

        # Update robot positions and compute CBF cost
        robot_pos = env.unwrapped.scene["robot"].data.root_pos_w
        distances, _ = manager.get_closest_obstacle(robot_pos[:, :2])
        h = distances - cfg.safety_margin
        cbf_cost = torch.relu(-h)

        total_reward += rewards.mean().item()
        total_cost += cbf_cost.mean().item()

        if step == 0 or step == num_steps - 1:
            print(f"\n  Step {step}:")
            print(f"    Reward: {rewards.mean():.3f}")
            print(f"    CBF cost: {cbf_cost.mean():.3f}")
            print(f"    Min distance: {distances.min():.3f}m")

    print(f"\n  Average reward over {num_steps} steps: {total_reward / num_steps:.3f}")
    print(f"  Average CBF cost over {num_steps} steps: {total_cost / num_steps:.3f}")

    # Final statistics
    print("\n" + "=" * 60)
    print("Final CBF Statistics")
    print("=" * 60)
    robot_pos = env.unwrapped.scene["robot"].data.root_pos_w
    distances, _ = manager.get_closest_obstacle(robot_pos[:, :2])
    h = distances - cfg.safety_margin
    print(f"  Mean distance: {distances.mean():.3f}m")
    print(f"  Min distance: {distances.min():.3f}m")
    print(f"  Safety violations (h < 0): {(h < 0).sum().item()} / {len(h)}")

    print("\n" + "=" * 60)
    print("All Tests Passed!")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
