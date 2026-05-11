#!/usr/bin/env python3

"""Visualize the paper curriculum obstacle distribution without training."""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=1200)
parser.add_argument("--stage", type=int, default=3, choices=[0, 1, 2, 3])
parser.add_argument("--device", type=str, default="cuda:0")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../source/unitree_rl_lab"))
import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.tasks.locomotion.obstacle_manager import ObstacleManager


CURRICULUM_STAGES = (
    {"num_obstacles": 8, "layout": "ring_passages"},
    {"num_obstacles": 12, "layout": "ring_passages"},
    {"num_obstacles": 16, "layout": "ring_passages"},
    {"num_obstacles": 20, "layout": "surrounded_front_open"},
)


def configure_env(env_cfg):
    import isaaclab.terrains as terrain_gen

    obstacle_env_cfg_mod = __import__(
        "unitree_rl_lab.tasks.locomotion.robots.g1.29dof.obstacle_env_cfg",
        fromlist=["MeshClutterPillarsTerrainCfg"],
    )
    MeshClutterPillarsTerrainCfg = obstacle_env_cfg_mod.MeshClutterPillarsTerrainCfg
    stage = CURRICULUM_STAGES[args_cli.stage]

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.scene.env_spacing = max(env_cfg.scene.env_spacing, 20.0)
    env_cfg.scene.terrain.terrain_generator = terrain_gen.TerrainGeneratorCfg(
        size=(30.0, 24.0),
        border_width=0.0,
        num_rows=1,
        num_cols=1,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        difficulty_range=(0.0, 0.0),
        use_cache=False,
        sub_terrains={
            "clutter_pillars": MeshClutterPillarsTerrainCfg(
                proportion=1.0,
                num_obstacles=stage["num_obstacles"],
                radius=env_cfg.obstacle_radius,
                height=env_cfg.obstacle_height,
                x_range=(-10.5, 10.5),
                y_range=(-8.5, 8.5),
                layout_variant=stage["layout"],
            ),
        },
    )
    env_cfg.num_obstacles = stage["num_obstacles"]
    env_cfg.obstacle_layout_mode = "clutter"
    env_cfg.obstacle_layout_variant = stage["layout"]
    env_cfg.obstacle_x_range = (-10.5, 10.5)
    env_cfg.obstacle_y_range = (-8.5, 8.5)
    env_cfg.obstacle_min_gap = 1.35
    env_cfg.obstacle_robot_clearance = 2.4
    env_cfg.min_obstacle_distance = 1.7
    env_cfg.obstacle_collision_mode = "terrain_mesh"
    return env_cfg


def main():
    task_name = "Unitree-G1-29dof-ObstacleAvoidance-realG1"
    env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    env_cfg = configure_env(env_cfg)

    env = gym.make(task_name, cfg=env_cfg)
    env.unwrapped.obstacle_manager = ObstacleManager(env.unwrapped, env.unwrapped.cfg)
    env.unwrapped.obstacle_manager.spawn_obstacles()
    obs, _ = env.reset()

    positions = env.unwrapped.obstacle_manager.obstacle_positions[0].cpu().numpy()
    radii = positions[:, 2]
    planar = positions[:, :2]
    norms = (planar[:, 0] ** 2 + planar[:, 1] ** 2) ** 0.5
    print(f"[INFO] stage={args_cli.stage} num_obstacles={len(positions)}")
    print(f"[INFO] radius range=({radii.min():.2f}, {radii.max():.2f})")
    print(f"[INFO] obstacle distance range=({norms.min():.2f}, {norms.max():.2f})")
    print("[INFO] first env obstacle xy:")
    for idx, (x, y, r) in enumerate(positions.tolist()):
        print(f"  {idx:02d}: x={x:.2f} y={y:.2f} r={r:.2f}")

    num_actions = env.unwrapped.action_space.shape[1]
    device = env.unwrapped.device
    for step in range(args_cli.steps):
        actions = torch.zeros(args_cli.num_envs, num_actions, device=device)
        env.step(actions)
        robot = env.unwrapped.scene["robot"]
        root_pos = robot.data.root_pos_w[0]
        env.unwrapped.sim.set_camera_view(
            eye=[float(root_pos[0]), float(root_pos[1]), float(root_pos[2] + 18.0)],
            target=[float(root_pos[0]), float(root_pos[1]), float(root_pos[2])],
        )
        if step % 200 == 0:
            print(f"[INFO] step={step}")
        time.sleep(0.01)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
