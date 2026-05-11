"""Visualize Omni-style LiDAR point clouds in the obstacle environment."""

import argparse
import os
import sys
import time

import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to show")
parser.add_argument("--steps", type=int, default=2000, help="Steps to run")
parser.add_argument("--samples", type=int, default=512, help="Number of sampled Omni LiDAR points")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab.sim as sim_utils
import isaaclab_tasks  # noqa: F401
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim.spawners.shapes import spawn_cylinder
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../source/unitree_rl_lab"))
import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.obstacle_manager import ObstacleManager


def spawn_irregular_obstacles(env):
    """Add connected cylinder clusters that behave like irregular obstacles.

    This keeps the default pillar environment intact and only augments env_0 for
    visual inspection. We use connected cylinders so the shape is visibly
    non-circular without introducing a separate mesh pipeline.
    """
    if env.num_envs < 1:
        return

    env_origin = env.scene.env_origins[0]
    clusters = [
        # L-shape near the left front
        [
            (1.8, 1.0, 0.18, 1.4),
            (2.15, 1.0, 0.18, 1.4),
            (2.50, 1.0, 0.18, 1.4),
            (2.50, 1.35, 0.18, 1.4),
            (2.50, 1.70, 0.18, 1.4),
        ],
        # Bulged wall on the right
        [
            (2.2, -1.2, 0.14, 1.0),
            (2.55, -1.1, 0.20, 1.6),
            (2.95, -1.0, 0.26, 1.2),
            (3.35, -0.95, 0.18, 1.5),
        ],
        # Narrow asymmetric gate ahead
        [
            (4.0, 0.25, 0.16, 1.5),
            (4.25, 0.45, 0.14, 1.5),
            (4.05, -0.45, 0.22, 1.7),
            (4.35, -0.65, 0.12, 1.2),
        ],
    ]

    for cluster_idx, cluster in enumerate(clusters):
        for elem_idx, (x_off, y_off, radius, height) in enumerate(cluster):
            obstacle_path = f"/World/envs/env_0/irregular_{cluster_idx}_{elem_idx}"
            spawn_cylinder(
                prim_path=obstacle_path,
                cfg=sim_utils.CylinderCfg(
                    radius=radius,
                    height=height,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.55, 0.85)),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        friction_combine_mode="multiply",
                        restitution_combine_mode="multiply",
                        static_friction=0.8,
                        dynamic_friction=0.8,
                    ),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                ),
                translation=(
                    env_origin[0].item() + x_off,
                    env_origin[1].item() + y_off,
                    height / 2.0,
                ),
            )


def make_point_visualizer():
    return VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/OmniLidarPoints",
            markers={
                "hit": sim_utils.SphereCfg(
                    radius=0.025,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.15)),
                ),
            },
        )
    )


def main():
    print("[INFO] Creating environment...")
    env_cfg = load_cfg_from_registry("Unitree-G1-29dof-ObstacleAvoidance", "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = "cuda:0"

    env = gym.make("Unitree-G1-29dof-ObstacleAvoidance", cfg=env_cfg)
    env.unwrapped.obstacle_manager = ObstacleManager(env.unwrapped, env.unwrapped.cfg)
    env.unwrapped.obstacle_manager.spawn_obstacles()
    spawn_irregular_obstacles(env.unwrapped)

    lidar_visualizer = make_point_visualizer()
    obs, _ = env.reset()
    print(f"[INFO] Policy obs shape: {obs['policy'].shape}")
    print("[INFO] Visualizing Omni-style LiDAR points and irregular obstacle clusters.")

    num_actions = env.unwrapped.action_space.shape[1]
    device = env.unwrapped.device
    for step in range(args_cli.steps):
        actions = torch.zeros(args_cli.num_envs, num_actions, device=device)
        env.step(actions)

        pointcloud = mdp.omni_lidar_pointcloud(
            env.unwrapped,
            samples=args_cli.samples,
            max_distance=6.0,
        ).view(args_cli.num_envs, args_cli.samples, 3)

        robot = env.unwrapped.scene["robot"]
        sensor_origin = robot.data.root_pos_w[:, :3] + torch.tensor([0.10, 0.0, 0.63], device=device)
        points_world = pointcloud[0] + sensor_origin[0].unsqueeze(0)
        valid = torch.isfinite(points_world).all(dim=-1)
        lidar_visualizer.visualize(points_world[valid])

        if step % 200 == 0:
            print(f"[INFO] step={step} visible_points={int(valid.sum().item())}")

        time.sleep(0.01)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
