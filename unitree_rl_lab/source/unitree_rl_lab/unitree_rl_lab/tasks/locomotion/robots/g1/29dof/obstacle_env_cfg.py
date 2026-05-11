# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Obstacle avoidance environment for G1 robot with static pillar obstacles."""

import math

import numpy as np
import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
import trimesh
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import SubTerrainBaseCfg, TerrainImporterCfg
from isaaclab.terrains.trimesh.utils import make_plane
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp


# Terrain with static pillar obstacles
OBSTACLE_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(20.0, 20.0),
    border_width=20.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 0.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
    },
)


def _make_clutter_pillar_offsets(
    num_obstacles: int,
    radius: float,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    layout_variant: str = "lanes",
) -> list[tuple[float, float, float]]:
    """Deterministic passable pillar layout shared by terrain mesh and analytical CBF/LiDAR."""
    if num_obstacles <= 0:
        return []

    if layout_variant == "u_wall":
        base_offsets = [
            (-1.5, 0.0),
            (-1.5, 1.4),
            (-1.5, 2.8),
            (-1.5, 4.2),
            (-1.5, 5.6),
            (-1.5, -1.4),
            (-1.5, -2.8),
            (-1.5, -4.2),
            (-1.5, -5.6),
            (0.0, 5.8),
            (1.6, 5.8),
            (3.2, 5.8),
            (4.8, 5.8),
            (6.4, 5.8),
            (8.0, 5.8),
            (0.0, -5.8),
            (1.6, -5.8),
            (3.2, -5.8),
            (4.8, -5.8),
            (6.4, -5.8),
            (8.0, -5.8),
            (2.2, 3.4),
            (3.8, -3.1),
            (5.4, 2.0),
            (6.8, -1.7),
        ]
        offsets = []
        for offset_x, offset_y in base_offsets[:num_obstacles]:
            offsets.append((float(offset_x), float(offset_y), radius))
        if len(offsets) < num_obstacles:
            x_values = np.linspace(0.5, 8.0, num_obstacles - len(offsets))
            for obs_idx, offset_x in enumerate(x_values):
                offset_y = 2.2 * math.sin(1.1 * obs_idx)
                offsets.append((float(offset_x), float(offset_y), radius))
        return offsets

    if layout_variant == "u_trap_close_solid":
        base_offsets = []
        for offset_y in np.linspace(-3.4, 3.4, 9):
            base_offsets.append((5.4, float(offset_y)))
        for offset_x in np.linspace(-1.2, 5.4, 9):
            base_offsets.append((float(offset_x), 3.4))
            base_offsets.append((float(offset_x), -3.4))
        return [(float(x), float(y), radius) for x, y in base_offsets[:num_obstacles]]

    if layout_variant == "u_wall_solid":
        base_offsets = []
        for offset_y in np.linspace(-5.8, 5.8, 13):
            base_offsets.append((-1.8, float(offset_y)))
        for offset_x in np.linspace(0.0, 8.6, 8):
            base_offsets.append((float(offset_x), 5.8))
            base_offsets.append((float(offset_x), -5.8))
        for inner_x, inner_y in [(2.4, 3.2), (3.8, -3.0), (5.6, 2.0), (6.9, -1.8)]:
            base_offsets.append((inner_x, inner_y))
        return [(float(x), float(y), radius) for x, y in base_offsets[:num_obstacles]]

    if layout_variant == "surround_wide":
        base_offsets = [
            (-4.5, 0.0),
            (-3.8, 5.8),
            (-3.1, -5.2),
            (-0.8, 8.2),
            (-1.6, -7.4),
            (2.8, 8.7),
            (1.9, -8.9),
            (6.6, 7.6),
            (7.4, -7.1),
            (11.0, 6.4),
            (10.7, -5.8),
            (15.0, 3.5),
            (15.9, -4.1),
            (19.0, 0.6),
            (3.2, 2.1),
            (4.6, -1.2),
            (6.8, 0.7),
            (8.4, 2.8),
            (9.3, -2.6),
            (12.2, 1.5),
            (13.6, -0.9),
            (2.4, -3.4),
            (4.1, 4.2),
            (5.7, -3.8),
            (7.9, 4.1),
            (9.8, -4.4),
            (11.1, 3.6),
            (12.9, -3.7),
            (14.8, 2.6),
            (16.6, -2.9),
            (5.2, 5.8),
            (8.9, -5.7),
            (11.8, 5.1),
            (14.1, -5.2),
        ]
        offsets = []
        for offset_x, offset_y in base_offsets[:num_obstacles]:
            offsets.append((float(offset_x), float(offset_y), radius))
        if len(offsets) < num_obstacles:
            x_values = np.linspace(2.0, 22.0, num_obstacles - len(offsets))
            for obs_idx, offset_x in enumerate(x_values):
                offset_y = 2.2 * math.sin(0.8 * obs_idx)
                offsets.append((float(offset_x), float(offset_y), radius))
        return offsets

    if layout_variant == "surround_nonconvex":
        base_offsets = [
            (-2.0, -5.8),
            (-2.0, -2.9),
            (-2.0, 0.0),
            (-2.0, 2.9),
            (-2.0, 5.8),
            (0.6, 7.2),
            (3.2, 7.2),
            (5.8, 7.2),
            (0.8, -7.0),
            (3.6, -7.0),
            (6.4, -7.0),
            (3.1, 2.6),
            (5.0, 2.6),
            (5.0, 4.4),
            (4.0, -4.5),
            (5.8, -4.5),
            (5.8, -2.8),
            (8.4, -2.1),
            (8.4, 0.1),
            (8.4, 2.3),
            (10.3, 4.6),
            (12.9, 4.6),
            (10.4, -4.4),
            (13.0, -4.4),
            (10.1, 2.8),
            (10.3, -2.6),
        ]
        return [(float(x), float(y), radius) for x, y in base_offsets[:num_obstacles]]

    if layout_variant == "surround_hybrid":
        base_offsets = [
            (-4.5, 0.0),
            (-3.8, 5.8),
            (-3.1, -5.2),
            (-0.8, 8.2),
            (-1.6, -7.4),
            (2.8, 8.7),
            (1.9, -8.9),
            (6.6, 7.6),
            (7.4, -7.1),
            (11.0, 6.4),
            (10.7, -5.8),
            (15.0, 3.5),
            (15.9, -4.1),
            (19.0, 0.6),
            (2.6, -3.8),
            (4.2, 4.4),
            (6.0, -4.2),
            (7.6, 3.8),
            (9.3, -4.5),
            (11.4, 3.9),
            (13.1, -3.9),
            (15.2, 2.7),
        ]
        return [(float(x), float(y), radius) for x, y in base_offsets[:num_obstacles]]

    if layout_variant == "end2end_distributed":
        rng = np.random.default_rng(20260420)
        x_min, x_max = float(x_range[0]), float(x_range[1])
        y_min, y_max = float(y_range[0]), float(y_range[1])
        min_center_gap = 2.0 * radius + 0.85
        offsets = []
        protected_corridor = [
            ((-3.0, -1.4), (1.6, 1.4)),
            ((2.0, 4.4), (-0.8, 2.2)),
            ((7.0, 10.8), (-2.4, -0.2)),
            ((11.0, 15.2), (1.0, 3.1)),
            ((15.8, 20.6), (-1.6, 0.8)),
        ]
        max_attempts = max(4000, 300 * num_obstacles)
        while len(offsets) < num_obstacles and max_attempts > 0:
            max_attempts -= 1
            cand_x = float(rng.uniform(x_min, x_max))
            cand_y = float(rng.uniform(y_min, y_max))
            if math.hypot(cand_x, cand_y) - radius < 1.2:
                continue
            blocked = False
            for (corr_x0, corr_x1), (corr_y0, corr_y1) in protected_corridor:
                if corr_x0 <= cand_x <= corr_x1 and corr_y0 <= cand_y <= corr_y1:
                    blocked = True
                    break
            if blocked:
                continue
            if all(math.hypot(cand_x - ox, cand_y - oy) >= min_center_gap for ox, oy, _ in offsets):
                offsets.append((cand_x, cand_y, radius))
        offsets.sort(key=lambda item: item[0])
        return offsets

    if layout_variant == "end2end_frontclose":
        rng = np.random.default_rng(20260422)
        x_min, x_max = float(x_range[0]), float(x_range[1])
        y_min, y_max = float(y_range[0]), float(y_range[1])
        min_center_gap = 2.0 * radius + 0.85
        offsets = [
            (1.45, 0.12, radius),
            (3.2, -1.7, radius),
            (4.8, 2.0, radius),
        ]
        protected_corridor = [
            ((-3.0, -1.0), (1.2, 1.0)),
            ((2.2, 4.0), (-0.4, 1.4)),
            ((6.5, 9.8), (-2.2, -0.2)),
            ((11.0, 15.0), (1.0, 3.2)),
            ((15.5, 20.6), (-1.8, 0.8)),
        ]
        max_attempts = max(4000, 300 * num_obstacles)
        while len(offsets) < num_obstacles and max_attempts > 0:
            max_attempts -= 1
            cand_x = float(rng.uniform(x_min, x_max))
            cand_y = float(rng.uniform(y_min, y_max))
            if math.hypot(cand_x, cand_y) - radius < 1.2:
                continue
            blocked = False
            for (corr_x0, corr_x1), (corr_y0, corr_y1) in protected_corridor:
                if corr_x0 <= cand_x <= corr_x1 and corr_y0 <= cand_y <= corr_y1:
                    blocked = True
                    break
            if blocked:
                continue
            if all(math.hypot(cand_x - ox, cand_y - oy) >= min_center_gap for ox, oy, _ in offsets):
                offsets.append((cand_x, cand_y, radius))
        offsets.sort(key=lambda item: item[0])
        return offsets[:num_obstacles]

    if layout_variant == "surrounded_front_open":
        base_offsets = [
            (3.0, 0.00),
            (-3.0, 2.00),
            (-3.0, -2.00),
            (-1.0, 3.30),
            (-1.0, -3.30),
            (2.0, 2.05),
            (2.0, -2.05),
            (4.2, 2.80),
            (4.2, -2.80),
            (6.2, 1.85),
            (6.2, -1.85),
            (8.6, 2.90),
            (8.6, -2.90),
            (11.0, 1.95),
            (11.0, -1.95),
            (12.5, 2.50),
            (12.5, -2.50),
            (2.4, 5.00),
            (2.4, -5.00),
            (7.2, 7.00),
            (7.2, -7.00),
            (14.2, 1.60),
            (14.2, -1.60),
            (15.0, 3.80),
            (15.0, -3.80),
            (11.0, 5.20),
            (11.0, -5.20),
            (5.0, 6.20),
            (5.0, -6.20),
            (-4.8, 4.20),
            (-4.8, -4.20),
            (-6.8, 2.20),
            (-6.8, -2.20),
            (-8.2, 5.70),
            (-8.2, -5.70),
            (-2.6, 6.60),
            (-2.6, -6.60),
            (0.0, 7.80),
            (0.0, -7.80),
            (10.0, 7.80),
            (10.0, -7.80),
            (13.2, 6.80),
            (13.2, -6.80),
            (16.4, 0.00),
            (16.4, 5.40),
            (16.4, -5.40),
            (6.2, 3.90),
            (6.2, -3.90),
            (8.8, 4.40),
            (8.8, -4.40),
            (3.2, 7.60),
            (3.2, -7.60),
        ]
        return [(float(x), float(y), radius) for x, y in base_offsets[:num_obstacles]]

    if layout_variant in ("continuous_avoidance", "mixed_obstacles_no_wall"):
        base_offsets = [
            (2.8, 0.00),
            (2.0, 2.20),
            (2.0, -2.20),
            (4.4, 1.55),
            (4.4, -2.35),
            (5.5, 3.20),
            (5.5, -3.60),
            (6.8, -0.75),
            (7.5, 2.45),
            (7.5, -2.95),
            (8.8, 0.95),
            (9.5, 3.50),
            (9.5, -2.30),
            (10.6, -0.90),
            (-2.4, 2.80),
            (-2.4, -2.80),
            (-0.2, 4.40),
            (-0.2, -4.40),
            (2.8, 5.20),
            (2.8, -5.20),
            (6.4, 5.40),
            (6.4, -5.40),
            (9.8, 5.00),
            (9.8, -5.00),
            (11.2, 2.20),
            (11.2, -3.80),
            (12.0, 0.90),
            (12.0, -1.90),
        ]
        return [(float(x), float(y), radius) for x, y in base_offsets[:num_obstacles]]

    if layout_variant == "arena_walled":
        rng = np.random.default_rng(20260424)
        x_min, x_max = float(x_range[0]), float(x_range[1])
        y_min, y_max = float(y_range[0]), float(y_range[1])
        min_center_gap = 2.0 * radius + 0.8
        offsets = []
        protected_spawn_zones = [
            ((-3.2, 2.8), (-2.0, 2.0)),
            ((2.8, 6.2), (-1.2, 1.2)),
        ]
        max_attempts = max(6000, 500 * num_obstacles)
        while len(offsets) < num_obstacles and max_attempts > 0:
            max_attempts -= 1
            cand_x = float(rng.uniform(x_min, x_max))
            cand_y = float(rng.uniform(y_min, y_max))
            blocked = False
            for (zone_x0, zone_x1), (zone_y0, zone_y1) in protected_spawn_zones:
                if zone_x0 <= cand_x <= zone_x1 and zone_y0 <= cand_y <= zone_y1:
                    blocked = True
                    break
            if blocked:
                continue
            if all(math.hypot(cand_x - ox, cand_y - oy) >= min_center_gap for ox, oy, _ in offsets):
                offsets.append((cand_x, cand_y, radius))
        offsets.sort(key=lambda item: (item[0], item[1]))
        return offsets[:num_obstacles]

    if layout_variant == "front_dense_hybrid":
        base_offsets = [
            (1.8, 0.0),
            (2.8, 2.2),
            (2.8, -2.2),
            (4.0, 0.9),
            (4.0, -0.9),
            (5.3, 3.8),
            (5.3, -3.8),
            (6.2, 0.0),
            (7.6, 2.5),
            (7.6, -2.5),
            (8.9, 0.9),
            (8.9, -0.9),
            (10.2, 3.3),
            (10.2, -3.3),
            (11.6, 0.0),
            (12.8, 2.2),
            (12.8, -2.2),
            (14.0, 0.7),
            (14.0, -0.7),
            (15.5, 3.0),
            (15.5, -3.0),
            (16.8, 0.0),
        ]
        return [(float(x), float(y), radius) for x, y in base_offsets[:num_obstacles]]

    if layout_variant == "dense_irregular":
        base_offsets = [
            (-2.8, 0.0),
            (-2.1, 3.9),
            (-1.8, -4.4),
            (-0.6, 1.7),
            (-0.2, -2.1),
            (0.8, 5.0),
            (1.4, -5.2),
            (2.1, 0.9),
            (2.7, 3.4),
            (3.0, -3.8),
            (3.8, 5.4),
            (4.2, -1.4),
            (4.9, 2.2),
            (5.3, -5.0),
            (6.1, 0.2),
            (6.8, 4.7),
            (7.4, -3.1),
            (8.0, 1.6),
            (8.6, -5.4),
            (9.2, 3.8),
            (10.0, -0.7),
            (10.8, 5.1),
            (11.5, -4.2),
            (12.3, 2.6),
            (13.1, -2.5),
            (14.0, 4.4),
            (14.7, -5.0),
            (15.6, 0.8),
        ]
        offsets = [(float(x), float(y), radius) for x, y in base_offsets[:num_obstacles]]
        if len(offsets) < num_obstacles:
            rng = np.random.default_rng(20260419)
            x_min, x_max = float(x_range[0]), float(x_range[1])
            y_min, y_max = float(y_range[0]), float(y_range[1])
            min_center_gap = 2.0 * radius + 0.55
            max_attempts = max(2000, 200 * num_obstacles)
            while len(offsets) < num_obstacles and max_attempts > 0:
                max_attempts -= 1
                cand_x = float(rng.uniform(x_min, x_max))
                cand_y = float(rng.uniform(y_min, y_max))
                if math.hypot(cand_x, cand_y) - radius < 1.3:
                    continue
                if all(math.hypot(cand_x - ox, cand_y - oy) >= min_center_gap for ox, oy, _ in offsets):
                    offsets.append((cand_x, cand_y, radius))
        return offsets[:num_obstacles]

    if layout_variant == "ring_passages":
        ring_specs = [
            (3.6, 6, 0.35),
            (5.8, 8, -0.10),
            (8.0, 10, 0.22),
            (10.2, 12, -0.28),
        ]
        gate_centers = (0.0, 1.18, -1.18, math.pi)
        gate_half_width = 0.26
        x_min, x_max = float(x_range[0]), float(x_range[1])
        y_min, y_max = float(y_range[0]), float(y_range[1])
        rng = np.random.default_rng(20260427)

        def _angle_diff(a: float, b: float) -> float:
            return abs(math.atan2(math.sin(a - b), math.cos(a - b)))

        offsets: list[tuple[float, float, float]] = []
        min_center_gap = 2.0 * radius + 1.05

        for ring_idx, (base_radius, nominal_count, phase) in enumerate(ring_specs):
            angles = np.linspace(-math.pi, math.pi, nominal_count, endpoint=False)
            for angle in angles:
                theta = float(angle + phase + rng.uniform(-0.08, 0.08))
                if any(_angle_diff(theta, gate_center) < gate_half_width for gate_center in gate_centers):
                    continue
                radial = float(base_radius + rng.uniform(-0.28, 0.28))
                cand_x = radial * math.cos(theta)
                cand_y = radial * math.sin(theta)
                if cand_x < x_min or cand_x > x_max or cand_y < y_min or cand_y > y_max:
                    continue
                if all(math.hypot(cand_x - ox, cand_y - oy) >= min_center_gap for ox, oy, _ in offsets):
                    offsets.append((cand_x, cand_y, radius))
                    if len(offsets) >= num_obstacles:
                        return offsets[:num_obstacles]

        max_attempts = max(4000, 400 * num_obstacles)
        min_radius = 3.0
        max_radius = min(max(abs(x_min), abs(x_max)), max(abs(y_min), abs(y_max))) + 0.2
        while len(offsets) < num_obstacles and max_attempts > 0:
            max_attempts -= 1
            theta = float(rng.uniform(-math.pi, math.pi))
            if any(_angle_diff(theta, gate_center) < gate_half_width for gate_center in gate_centers):
                continue
            radial = float(rng.uniform(min_radius, max_radius))
            cand_x = radial * math.cos(theta)
            cand_y = radial * math.sin(theta)
            if cand_x < x_min or cand_x > x_max or cand_y < y_min or cand_y > y_max:
                continue
            if all(math.hypot(cand_x - ox, cand_y - oy) >= min_center_gap for ox, oy, _ in offsets):
                offsets.append((cand_x, cand_y, radius))
        return offsets[:num_obstacles]

    x_values = np.linspace(float(x_range[0]), float(x_range[1]), num_obstacles)
    y_min, y_max = float(y_range[0]), float(y_range[1])
    lanes = np.array([0.0, 0.78 * y_max, 0.78 * y_min, 0.38 * y_max, 0.38 * y_min, y_max, y_min])
    offsets = []
    for obs_idx, offset_x in enumerate(x_values):
        lane = lanes[obs_idx % len(lanes)]
        stagger = 0.18 * math.sin(1.7 * obs_idx)
        offset_y = float(np.clip(lane + stagger, y_min, y_max))
        offsets.append((float(offset_x), offset_y, radius))
    return offsets


def clutter_pillars_terrain(
    difficulty: float, cfg: "MeshClutterPillarsTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate one terrain tile with obstacle pillars fused into the terrain mesh."""
    del difficulty
    meshes = [make_plane(cfg.size, 0.0, center_zero=False)]
    terrain_center = np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0])

    if cfg.layout_variant == "u_wall_solid":
        wall_specs = [
            ((terrain_center[0] - 1.8, terrain_center[1], 0.5 * cfg.height), (0.9, 12.8, cfg.height)),
            ((terrain_center[0] + 3.9, terrain_center[1] + 5.8, 0.5 * cfg.height), (10.4, 0.9, cfg.height)),
            ((terrain_center[0] + 3.9, terrain_center[1] - 5.8, 0.5 * cfg.height), (10.4, 0.9, cfg.height)),
        ]
        for pos, dims in wall_specs:
            meshes.append(trimesh.creation.box(dims, trimesh.transformations.translation_matrix(pos)))
        for pos in [(terrain_center[0] + 2.4, terrain_center[1] + 3.2, 0.5 * cfg.height), (terrain_center[0] + 3.8, terrain_center[1] - 3.0, 0.5 * cfg.height), (terrain_center[0] + 5.6, terrain_center[1] + 2.0, 0.5 * cfg.height), (terrain_center[0] + 6.9, terrain_center[1] - 1.8, 0.5 * cfg.height)]:
            meshes.append(trimesh.creation.box((0.9, 0.9, cfg.height), trimesh.transformations.translation_matrix(pos)))
        return meshes, terrain_center

    if cfg.layout_variant == "u_trap_close_solid":
        wall_specs = [
            ((terrain_center[0] + 5.4, terrain_center[1], 0.5 * cfg.height), (0.9, 7.6, cfg.height)),
            ((terrain_center[0] + 2.1, terrain_center[1] + 3.4, 0.5 * cfg.height), (7.2, 0.9, cfg.height)),
            ((terrain_center[0] + 2.1, terrain_center[1] - 3.4, 0.5 * cfg.height), (7.2, 0.9, cfg.height)),
        ]
        for pos, dims in wall_specs:
            meshes.append(trimesh.creation.box(dims, trimesh.transformations.translation_matrix(pos)))
        return meshes, terrain_center

    if cfg.layout_variant == "surround_nonconvex":
        box_specs = [
            ((terrain_center[0] - 2.0, terrain_center[1], 0.5 * cfg.height), (1.2, 13.2, cfg.height)),
            ((terrain_center[0] + 3.2, terrain_center[1] + 7.2, 0.5 * cfg.height), (8.8, 1.2, cfg.height)),
            ((terrain_center[0] + 3.6, terrain_center[1] - 7.0, 0.5 * cfg.height), (9.2, 1.2, cfg.height)),
            ((terrain_center[0] + 11.7, terrain_center[1] + 4.6, 0.5 * cfg.height), (5.6, 1.2, cfg.height)),
            ((terrain_center[0] + 11.8, terrain_center[1] - 4.4, 0.5 * cfg.height), (5.8, 1.2, cfg.height)),
            ((terrain_center[0] + 4.1, terrain_center[1] + 2.6, 0.5 * cfg.height), (3.8, 1.0, cfg.height)),
            ((terrain_center[0] + 5.0, terrain_center[1] + 4.1, 0.5 * cfg.height), (1.0, 3.0, cfg.height)),
            ((terrain_center[0] + 4.9, terrain_center[1] - 4.5, 0.5 * cfg.height), (3.0, 1.0, cfg.height)),
            ((terrain_center[0] + 5.8, terrain_center[1] - 3.2, 0.5 * cfg.height), (1.0, 3.6, cfg.height)),
            ((terrain_center[0] + 8.4, terrain_center[1] + 0.1, 0.5 * cfg.height), (1.2, 5.8, cfg.height)),
            ((terrain_center[0] + 10.2, terrain_center[1] + 2.8, 0.5 * cfg.height), (3.0, 1.0, cfg.height)),
            ((terrain_center[0] + 10.3, terrain_center[1] - 2.6, 0.5 * cfg.height), (3.0, 1.0, cfg.height)),
        ]
        for pos, dims in box_specs:
            meshes.append(trimesh.creation.box(dims, trimesh.transformations.translation_matrix(pos)))
        return meshes, terrain_center

    if cfg.layout_variant == "surround_hybrid":
        wall_specs = [
            ((terrain_center[0] + 4.8, terrain_center[1] + 5.0, 0.5 * cfg.height), (4.4, 0.9, cfg.height)),
            ((terrain_center[0] + 9.4, terrain_center[1] - 4.6, 0.5 * cfg.height), (4.8, 0.9, cfg.height)),
            ((terrain_center[0] + 13.0, terrain_center[1] + 1.7, 0.5 * cfg.height), (0.9, 4.4, cfg.height)),
        ]
        for pos, dims in wall_specs:
            meshes.append(trimesh.creation.box(dims, trimesh.transformations.translation_matrix(pos)))

    if cfg.layout_variant == "end2end_distributed":
        wall_specs = [
            ((terrain_center[0] + 4.6, terrain_center[1] + 5.4, 0.5 * cfg.height), (2.6, 0.35, cfg.height), 0.24),
            ((terrain_center[0] + 9.8, terrain_center[1] - 5.5, 0.5 * cfg.height), (3.0, 0.35, cfg.height), -0.28),
            ((terrain_center[0] + 15.8, terrain_center[1] + 1.4, 0.5 * cfg.height), (0.35, 2.8, cfg.height), 0.18),
            ((terrain_center[0] + 18.2, terrain_center[1] - 4.8, 0.5 * cfg.height), (2.2, 0.35, cfg.height), 0.42),
        ]
        for pos, dims, yaw in wall_specs:
            transform = trimesh.transformations.concatenate_matrices(
                trimesh.transformations.translation_matrix(pos),
                trimesh.transformations.rotation_matrix(yaw, [0.0, 0.0, 1.0]),
            )
            meshes.append(trimesh.creation.box(dims, transform))

        shape_rng = np.random.default_rng(20260421)
        for obs_idx, (offset_x, offset_y, _) in enumerate(
            _make_clutter_pillar_offsets(cfg.num_obstacles, cfg.radius, cfg.x_range, cfg.y_range, cfg.layout_variant)
        ):
            base_transform = trimesh.transformations.translation_matrix(
                (terrain_center[0] + offset_x, terrain_center[1] + offset_y, 0.5 * cfg.height)
            )
            if obs_idx % 5 in (1, 3):
                yaw = float(shape_rng.uniform(-0.85, 0.85))
                dims = (
                    float(shape_rng.uniform(0.75, 1.45)),
                    float(shape_rng.uniform(0.65, 1.95)),
                    cfg.height,
                )
                transform = trimesh.transformations.concatenate_matrices(
                    base_transform,
                    trimesh.transformations.rotation_matrix(yaw, [0.0, 0.0, 1.0]),
                )
                meshes.append(trimesh.creation.box(dims, transform))
            else:
                radius_scale = float(shape_rng.uniform(0.82, 1.18))
                meshes.append(
                    trimesh.creation.cylinder(
                        radius=cfg.radius * radius_scale,
                        height=cfg.height,
                        sections=24,
                        transform=base_transform,
                    )
                )
        return meshes, terrain_center

    if cfg.layout_variant == "end2end_frontclose":
        wall_specs = [
            ((terrain_center[0] + 5.0, terrain_center[1] + 5.0, 0.5 * cfg.height), (2.4, 0.35, cfg.height), 0.20),
            ((terrain_center[0] + 10.4, terrain_center[1] - 5.3, 0.5 * cfg.height), (3.0, 0.35, cfg.height), -0.26),
            ((terrain_center[0] + 16.0, terrain_center[1] + 1.6, 0.5 * cfg.height), (0.35, 2.6, cfg.height), 0.16),
        ]
        for pos, dims, yaw in wall_specs:
            transform = trimesh.transformations.concatenate_matrices(
                trimesh.transformations.translation_matrix(pos),
                trimesh.transformations.rotation_matrix(yaw, [0.0, 0.0, 1.0]),
            )
            meshes.append(trimesh.creation.box(dims, transform))

        shape_rng = np.random.default_rng(20260423)
        for obs_idx, (offset_x, offset_y, _) in enumerate(
            _make_clutter_pillar_offsets(cfg.num_obstacles, cfg.radius, cfg.x_range, cfg.y_range, cfg.layout_variant)
        ):
            base_transform = trimesh.transformations.translation_matrix(
                (terrain_center[0] + offset_x, terrain_center[1] + offset_y, 0.5 * cfg.height)
            )
            if obs_idx == 0:
                meshes.append(
                    trimesh.creation.box(
                        (1.0, 0.9, cfg.height),
                        trimesh.transformations.concatenate_matrices(
                            base_transform,
                            trimesh.transformations.rotation_matrix(0.12, [0.0, 0.0, 1.0]),
                        ),
                    )
                )
            elif obs_idx % 5 in (1, 3):
                yaw = float(shape_rng.uniform(-0.85, 0.85))
                dims = (
                    float(shape_rng.uniform(0.75, 1.45)),
                    float(shape_rng.uniform(0.65, 1.95)),
                    cfg.height,
                )
                transform = trimesh.transformations.concatenate_matrices(
                    base_transform,
                    trimesh.transformations.rotation_matrix(yaw, [0.0, 0.0, 1.0]),
                )
                meshes.append(trimesh.creation.box(dims, transform))
            else:
                radius_scale = float(shape_rng.uniform(0.82, 1.18))
                meshes.append(
                    trimesh.creation.cylinder(
                        radius=cfg.radius * radius_scale,
                        height=cfg.height,
                        sections=24,
                        transform=base_transform,
                    )
                )
        return meshes, terrain_center

    if cfg.layout_variant == "arena_walled":
        wall_thickness = 0.35
        margin = 1.0
        wall_specs = [
            (
                (terrain_center[0] - 0.5 * cfg.size[0] + margin, terrain_center[1], 0.5 * cfg.height),
                (wall_thickness, cfg.size[1] - 2.0 * margin, cfg.height),
            ),
            (
                (terrain_center[0] + 0.5 * cfg.size[0] - margin, terrain_center[1], 0.5 * cfg.height),
                (wall_thickness, cfg.size[1] - 2.0 * margin, cfg.height),
            ),
            (
                (terrain_center[0], terrain_center[1] - 0.5 * cfg.size[1] + margin, 0.5 * cfg.height),
                (cfg.size[0] - 2.0 * margin, wall_thickness, cfg.height),
            ),
            (
                (terrain_center[0], terrain_center[1] + 0.5 * cfg.size[1] - margin, 0.5 * cfg.height),
                (cfg.size[0] - 2.0 * margin, wall_thickness, cfg.height),
            ),
        ]
        for pos, dims in wall_specs:
            meshes.append(trimesh.creation.box(dims, trimesh.transformations.translation_matrix(pos)))

        shape_rng = np.random.default_rng(20260425)
        for obs_idx, (offset_x, offset_y, _) in enumerate(
            _make_clutter_pillar_offsets(cfg.num_obstacles, cfg.radius, cfg.x_range, cfg.y_range, cfg.layout_variant)
        ):
            base_transform = trimesh.transformations.translation_matrix(
                (terrain_center[0] + offset_x, terrain_center[1] + offset_y, 0.5 * cfg.height)
            )
            shape_mode = obs_idx % 6
            if shape_mode in (1, 4):
                yaw = float(shape_rng.uniform(-1.1, 1.1))
                dims = (
                    float(shape_rng.uniform(0.8, 1.7)),
                    float(shape_rng.uniform(0.55, 1.6)),
                    cfg.height,
                )
                transform = trimesh.transformations.concatenate_matrices(
                    base_transform,
                    trimesh.transformations.rotation_matrix(yaw, [0.0, 0.0, 1.0]),
                )
                meshes.append(trimesh.creation.box(dims, transform))
            elif shape_mode == 3:
                yaw = float(shape_rng.uniform(-1.0, 1.0))
                dims = (
                    float(shape_rng.uniform(1.4, 2.2)),
                    float(shape_rng.uniform(0.35, 0.55)),
                    cfg.height,
                )
                transform = trimesh.transformations.concatenate_matrices(
                    base_transform,
                    trimesh.transformations.rotation_matrix(yaw, [0.0, 0.0, 1.0]),
                )
                meshes.append(trimesh.creation.box(dims, transform))
            else:
                radius_scale = float(shape_rng.uniform(0.82, 1.22))
                meshes.append(
                    trimesh.creation.cylinder(
                        radius=cfg.radius * radius_scale,
                        height=cfg.height,
                        sections=24,
                        transform=base_transform,
                    )
                )
        return meshes, terrain_center

    if cfg.layout_variant == "front_dense_hybrid":
        wall_specs = [
            ((terrain_center[0] + 4.6, terrain_center[1] + 4.8, 0.5 * cfg.height), (4.0, 0.9, cfg.height)),
            ((terrain_center[0] + 8.8, terrain_center[1] - 4.6, 0.5 * cfg.height), (4.4, 0.9, cfg.height)),
            ((terrain_center[0] + 12.5, terrain_center[1] + 4.0, 0.5 * cfg.height), (3.6, 0.9, cfg.height)),
            ((terrain_center[0] + 12.2, terrain_center[1] - 3.8, 0.5 * cfg.height), (3.2, 0.9, cfg.height)),
            ((terrain_center[0] + 15.2, terrain_center[1], 0.5 * cfg.height), (0.9, 4.2, cfg.height)),
        ]
        for pos, dims in wall_specs:
            meshes.append(trimesh.creation.box(dims, trimesh.transformations.translation_matrix(pos)))

    if cfg.layout_variant == "mixed_obstacles_no_wall":
        shape_rng = np.random.default_rng(20260429)
        base_offsets = _make_clutter_pillar_offsets(
            cfg.num_obstacles, cfg.radius, cfg.x_range, cfg.y_range, "continuous_avoidance"
        )
        for obs_idx, (offset_x, offset_y, _) in enumerate(base_offsets):
            base_transform = trimesh.transformations.translation_matrix(
                (terrain_center[0] + offset_x, terrain_center[1] + offset_y, 0.5 * cfg.height)
            )
            shape_mode = obs_idx % 6
            if shape_mode == 0:
                radius_scale = float(shape_rng.uniform(0.85, 1.20))
                meshes.append(
                    trimesh.creation.cylinder(
                        radius=cfg.radius * radius_scale,
                        height=cfg.height,
                        sections=24,
                        transform=base_transform,
                    )
                )
            elif shape_mode in (1, 4):
                dims = (
                    float(shape_rng.uniform(0.85, 1.35)),
                    float(shape_rng.uniform(0.70, 1.20)),
                    cfg.height,
                )
                yaw = float(shape_rng.uniform(-0.85, 0.85))
                transform = trimesh.transformations.concatenate_matrices(
                    base_transform,
                    trimesh.transformations.rotation_matrix(yaw, [0.0, 0.0, 1.0]),
                )
                meshes.append(trimesh.creation.box(dims, transform))
            elif shape_mode == 3:
                dims = (
                    float(shape_rng.uniform(1.45, 2.10)),
                    float(shape_rng.uniform(0.38, 0.55)),
                    cfg.height,
                )
                yaw = float(shape_rng.uniform(-0.95, 0.95))
                transform = trimesh.transformations.concatenate_matrices(
                    base_transform,
                    trimesh.transformations.rotation_matrix(yaw, [0.0, 0.0, 1.0]),
                )
                meshes.append(trimesh.creation.box(dims, transform))
            else:
                dims = (
                    float(shape_rng.uniform(0.70, 1.00)),
                    float(shape_rng.uniform(0.70, 1.00)),
                    cfg.height,
                )
                yaw = float(shape_rng.uniform(-0.45, 0.45))
                transform = trimesh.transformations.concatenate_matrices(
                    base_transform,
                    trimesh.transformations.rotation_matrix(yaw, [0.0, 0.0, 1.0]),
                )
                meshes.append(trimesh.creation.box(dims, transform))
        return meshes, terrain_center

    if cfg.layout_variant == "rect_blocks_no_wall":
        shape_rng = np.random.default_rng(20260428)
        for obs_idx, (offset_x, offset_y, _) in enumerate(
            _make_clutter_pillar_offsets(cfg.num_obstacles, cfg.radius, cfg.x_range, cfg.y_range, "continuous_avoidance")
        ):
            base_transform = trimesh.transformations.translation_matrix(
                (terrain_center[0] + offset_x, terrain_center[1] + offset_y, 0.5 * cfg.height)
            )
            if obs_idx % 4 == 0:
                dims = (1.05, 0.65, cfg.height)
                yaw = float(shape_rng.uniform(-0.55, 0.55))
            elif obs_idx % 4 == 1:
                dims = (0.75, 1.10, cfg.height)
                yaw = float(shape_rng.uniform(-0.45, 0.45))
            elif obs_idx % 4 == 2:
                dims = (0.90, 0.90, cfg.height)
                yaw = float(shape_rng.uniform(-0.35, 0.35))
            else:
                dims = (1.20, 0.55, cfg.height)
                yaw = float(shape_rng.uniform(-0.65, 0.65))
            transform = trimesh.transformations.concatenate_matrices(
                base_transform,
                trimesh.transformations.rotation_matrix(yaw, [0.0, 0.0, 1.0]),
            )
            meshes.append(trimesh.creation.box(dims, transform))
        return meshes, terrain_center

    for offset_x, offset_y, _ in _make_clutter_pillar_offsets(
        cfg.num_obstacles, cfg.radius, cfg.x_range, cfg.y_range, cfg.layout_variant
    ):
        transform = trimesh.transformations.translation_matrix(
            (terrain_center[0] + offset_x, terrain_center[1] + offset_y, 0.5 * cfg.height)
        )
        pillar = trimesh.creation.cylinder(radius=cfg.radius, height=cfg.height, sections=24, transform=transform)
        meshes.append(pillar)

    return meshes, terrain_center


@configclass
class MeshClutterPillarsTerrainCfg(SubTerrainBaseCfg):
    """Terrain tile containing fixed passable pillar protrusions."""

    function = clutter_pillars_terrain

    num_obstacles: int = 10
    radius: float = 0.3
    height: float = 2.0
    x_range: tuple[float, float] = (1.8, 12.0)
    y_range: tuple[float, float] = (-3.2, 3.2)
    layout_variant: str = "lanes"


@configclass
class ObstacleSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with pillar obstacles."""

    # ground terrain (flat with obstacles spawned separately)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=OBSTACLE_TERRAIN_CFG,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # robot
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # contact sensor for gait/feet rewards
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    # Optional visual ray-caster for debugging. The training signal uses analytical
    # ray-casting against the spawned cylinders so that sensing and obstacle geometry stay aligned.
    obstacle_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.6)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=[1.2, 1.2]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """Configuration for events."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-8.0, 8.0), "y": (-8.0, 8.0), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )


@configclass
class CommandsCfg:
    """Command for random walking with obstacle avoidance."""

    # Random velocity commands (no specific target)
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 1.0),  # Forward only
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-0.5, 0.5),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observations with obstacle distances."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        obstacle_scan = ObsTerm(
            func=mdp.obstacle_raycast_scan,
            params={"num_rays": 9, "max_distance": 6.0, "fov_deg": 180.0},
            clip=(0.0, 6.0),
        )

        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged observations."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        obstacle_scan = ObsTerm(
            func=mdp.obstacle_raycast_scan,
            params={"num_rays": 9, "max_distance": 6.0, "fov_deg": 180.0},
            clip=(0.0, 6.0),
        )

        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 5

    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Reward terms."""

    # -- Task: follow velocity commands
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    alive = RewTerm(func=mdp.is_alive, weight=0.1)

    # -- Penalties
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    base_height = RewTerm(func=mdp.base_height_l2, weight=-10.0, params={"target_height": 0.78})

    # -- Joint deviation (posture control, from velocity_env_cfg)
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*",
                ],
            )
        },
    )
    joint_deviation_waists = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["waist.*"],
            )
        },
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )

    gait = RewTerm(
        func=mdp.feet_gait,
        weight=0.35,
        params={
            "period": 0.72,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.08,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    feet_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=1.0,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
        },
    )

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.5,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )

    proxemic_comfort = RewTerm(
        func=mdp.proxemic_comfort,
        weight=-0.2,
        params={"comfort_margin": 1.2},
    )

    safe_approach_velocity = RewTerm(
        func=mdp.safe_approach_velocity,
        weight=-0.12,
        params={"comfort_margin": 1.4},
    )

    safe_approach_acceleration = RewTerm(
        func=mdp.safe_approach_acceleration,
        weight=-0.02,
        params={"comfort_margin": 1.4},
    )

    tangential_avoidance = RewTerm(
        func=mdp.tangential_avoidance,
        weight=0.18,
        params={"comfort_margin": 1.6, "command_name": "base_velocity"},
    )


@configclass
class TerminationsCfg:
    """Termination terms."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class ObstacleAvoidanceEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for obstacle avoidance environment."""

    scene: ObstacleSceneCfg = ObstacleSceneCfg(num_envs=1024, env_spacing=5.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    # Obstacle parameters (reduced for memory efficiency)
    num_obstacles: int = 2  # Reduced from 3 to save memory
    obstacle_radius: float = 0.3  # Radius of each pillar
    obstacle_height: float = 2.0  # Height of each pillar
    min_obstacle_distance: float = 2.0  # Minimum distance between obstacles
    obstacle_layout_mode: str = "radial"
    obstacle_x_range: tuple[float, float] = (1.5, 10.0)
    obstacle_y_range: tuple[float, float] = (-3.0, 3.0)
    obstacle_min_gap: float = 1.0
    obstacle_robot_clearance: float = 1.2
    obstacle_collision_enabled: bool = True
    obstacle_collision_mode: str = "rigid_cylinder"
    safety_margin: float = 0.5  # CBF safety margin D_min
    obstacle_num_rays: int = 9
    obstacle_ray_max_distance: float = 6.0
    obstacle_ray_fov_deg: float = 180.0
    cbf_gamma: float = 0.5
    collision_distance: float = 0.2

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        # PhysX GPU memory configuration (optimized for 4096 envs)
        self.sim.physx.gpu_max_rigid_patch_count = 5 * 2**15  # Reduced to save memory
        self.sim.physx.gpu_found_lost_pairs_capacity = 10 * 1024 * 1024  # 10M pairs
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 10 * 1024 * 1024
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 10 * 1024 * 1024
        self.sim.physx.gpu_max_soft_body_contacts = 2**15
        self.sim.physx.gpu_max_partition_iteration_counts = 4096

        self.scene.obstacle_scanner.update_period = self.decimation * self.sim.dt
        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class ObstacleAvoidancePlayEnvCfg(ObstacleAvoidanceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32


# Import ObstacleManager from parent module
from unitree_rl_lab.tasks.locomotion.obstacle_manager import ObstacleManager
