"""Obstacle avoidance reward functions for PPO training."""

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def obstacle_distance_penalty(
    env: ManagerBasedRLEnv,
    obstacle_positions: torch.Tensor,
    obstacle_radius: float,
    safety_margin: float,
    max_penalty: float = 10.0,
) -> torch.Tensor:
    """
    Compute penalty based on distance to closest obstacle.

    This implements a simplified CBF-like cost as a penalty:
    - h = distance - safety_margin
    - penalty = max(0, -h) = max(0, safety_margin - distance)

    Args:
        env: Environment instance
        obstacle_positions: (num_envs, num_obstacles, 3) tensor of obstacle positions
        obstacle_radius: Radius of obstacles
        safety_margin: Minimum safe distance from obstacle surface
        max_penalty: Maximum penalty value

    Returns:
        (num_envs,) tensor of penalties (negative rewards)
    """
    # Get robot positions
    robot_pos = env.scene["robot"].data.root_pos_w[:, :2]  # (num_envs, 2)

    num_envs = robot_pos.shape[0]
    device = robot_pos.device

    penalties = torch.zeros(num_envs, device=device)

    # For each environment, compute distance to closest obstacle
    for env_idx in range(num_envs):
        if env_idx < obstacle_positions.shape[0]:
            robot_xy = robot_pos[env_idx]  # (2,)
            obstacles = obstacle_positions[env_idx]  # (num_obstacles, 3)

            # Compute distance to each obstacle
            diff = obstacles[:, :2] - robot_xy.unsqueeze(0)  # (num_obstacles, 2)
            dist_to_center = torch.norm(diff, dim=1)  # (num_obstacles,)
            dist_to_surface = dist_to_center - obstacle_radius  # (num_obstacles,)

            # Find closest obstacle
            min_dist = torch.min(dist_to_surface)

            # CBF-style penalty: max(0, safety_margin - distance)
            # h = distance - safety_margin, cost = max(0, -h)
            h = min_dist - safety_margin
            penalty = torch.relu(-h)

            # Additional collision penalty
            if min_dist < 0.1:
                penalty += 10.0

            penalties[env_idx] = torch.clamp(penalty, 0, max_penalty)

    return penalties
