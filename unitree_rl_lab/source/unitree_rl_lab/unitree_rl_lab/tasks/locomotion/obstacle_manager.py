"""Obstacle manager for static pillar obstacles."""

import torch
import math

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.shapes import spawn_cylinder
from isaaclab.sim.spawners.meshes import spawn_mesh_cylinder
from isaaclab.sim.views import XformPrimView


class ObstacleManager:
    """Manager for spawning and tracking static pillar obstacles."""

    def __init__(self, env, cfg):
        self.env = env
        self.cfg = cfg
        self.obstacle_positions = []  # List of tensors per environment
        self.obstacle_positions_tensor = None
        self.geometry_proxy_tensor = None
        self.obstacle_radius = cfg.obstacle_radius
        self.obstacle_height = cfg.obstacle_height
        self.obstacle_geoms = []  # Track spawned obstacles
        self._obstacle_offsets = []  # Store relative offsets from env origin
        self._obstacle_views = []
        self.max_num_obstacles = cfg.num_obstacles
        self.active_num_obstacles = cfg.num_obstacles

    def _make_radial_template_offsets(self, num_obstacles: int, radius: float) -> torch.Tensor:
        template_offsets = []
        for _ in range(num_obstacles):
            max_dist = 8.0
            angle = torch.rand(1).item() * 2 * math.pi
            dist = torch.rand(1).item() * max_dist + 2.0

            offset_x = dist * math.cos(angle)
            offset_y = dist * math.sin(angle)
            template_offsets.append([offset_x, offset_y, radius])
        return torch.tensor(template_offsets, dtype=torch.float32)

    def _make_clutter_template_offsets(self, num_obstacles: int, radius: float) -> torch.Tensor:
        if getattr(self.cfg, "obstacle_collision_mode", "rigid_cylinder") == "terrain_mesh":
            return self._make_terrain_clutter_template_offsets(num_obstacles, radius)

        x_min, x_max = getattr(self.cfg, "obstacle_x_range", (1.5, 10.0))
        y_min, y_max = getattr(self.cfg, "obstacle_y_range", (-3.0, 3.0))
        min_gap = float(getattr(self.cfg, "obstacle_min_gap", 1.0))
        robot_clearance = float(getattr(self.cfg, "obstacle_robot_clearance", 1.2))
        min_center_gap = max(float(getattr(self.cfg, "min_obstacle_distance", 0.0)), 2.0 * radius + min_gap)

        accepted: list[list[float]] = []
        max_attempts = max(1000, 300 * num_obstacles)
        for _ in range(max_attempts):
            if len(accepted) >= num_obstacles:
                break

            offset_x = torch.empty(1).uniform_(float(x_min), float(x_max)).item()
            offset_y = torch.empty(1).uniform_(float(y_min), float(y_max)).item()
            if math.hypot(offset_x, offset_y) - radius < robot_clearance:
                continue
            if all(math.hypot(offset_x - ox, offset_y - oy) >= min_center_gap for ox, oy, _ in accepted):
                accepted.append([offset_x, offset_y, radius])

        if len(accepted) < num_obstacles:
            x_span = max(float(x_max) - float(x_min), 1.0)
            y_span = max(float(y_max) - float(y_min), 1.0)
            cols = max(1, int(math.ceil(math.sqrt(num_obstacles * x_span / y_span))))
            rows = max(1, int(math.ceil(num_obstacles / cols)))
            for row in range(rows):
                for col in range(cols):
                    if len(accepted) >= num_obstacles:
                        break
                    offset_x = float(x_min) + (col + 0.5) * x_span / cols
                    offset_y = float(y_min) + (row + 0.5) * y_span / rows
                    if row % 2 == 1:
                        offset_x = min(float(x_max), offset_x + 0.25 * x_span / cols)
                    if math.hypot(offset_x, offset_y) - radius >= robot_clearance:
                        accepted.append([offset_x, offset_y, radius])

        accepted = sorted(accepted[:num_obstacles], key=lambda item: (item[0], abs(item[1])))
        return torch.tensor(accepted, dtype=torch.float32)

    def _make_terrain_clutter_template_offsets(self, num_obstacles: int, radius: float) -> torch.Tensor:
        layout_variant = getattr(self.cfg, "obstacle_layout_variant", "lanes")
        if layout_variant == "u_trap_close_solid":
            offsets = []
            for offset_y in torch.linspace(-3.4, 3.4, steps=9).tolist():
                offsets.append([5.4, float(offset_y), radius])
            for offset_x in torch.linspace(-1.2, 5.4, steps=9).tolist():
                offsets.append([float(offset_x), 3.4, radius])
                offsets.append([float(offset_x), -3.4, radius])
            return torch.tensor(offsets[:num_obstacles], dtype=torch.float32)

        if layout_variant == "u_wall_solid":
            offsets = []
            for offset_y in torch.linspace(-5.8, 5.8, steps=13).tolist():
                offsets.append([-1.8, float(offset_y), radius])
            for offset_x in torch.linspace(0.0, 8.6, steps=8).tolist():
                offsets.append([float(offset_x), 5.8, radius])
                offsets.append([float(offset_x), -5.8, radius])
            offsets.extend([
                [2.4, 3.2, radius],
                [3.8, -3.0, radius],
                [5.6, 2.0, radius],
                [6.9, -1.8, radius],
            ])
            return torch.tensor(offsets[:num_obstacles], dtype=torch.float32)

        if layout_variant == "u_wall":
            base_offsets = [
                [-1.5, 0.0, radius],
                [-1.5, 1.4, radius],
                [-1.5, 2.8, radius],
                [-1.5, 4.2, radius],
                [-1.5, 5.6, radius],
                [-1.5, -1.4, radius],
                [-1.5, -2.8, radius],
                [-1.5, -4.2, radius],
                [-1.5, -5.6, radius],
                [0.0, 5.8, radius],
                [1.6, 5.8, radius],
                [3.2, 5.8, radius],
                [4.8, 5.8, radius],
                [6.4, 5.8, radius],
                [8.0, 5.8, radius],
                [0.0, -5.8, radius],
                [1.6, -5.8, radius],
                [3.2, -5.8, radius],
                [4.8, -5.8, radius],
                [6.4, -5.8, radius],
                [8.0, -5.8, radius],
                [2.2, 3.4, radius],
                [3.8, -3.1, radius],
                [5.4, 2.0, radius],
                [6.8, -1.7, radius],
            ]
            offsets = base_offsets[:num_obstacles]
            if len(offsets) < num_obstacles:
                extra_x = torch.linspace(0.5, 8.0, steps=num_obstacles - len(offsets))
                for obs_idx, offset_x in enumerate(extra_x.tolist()):
                    offsets.append([float(offset_x), 2.2 * math.sin(1.1 * obs_idx), radius])
            return torch.tensor(offsets, dtype=torch.float32)

        if layout_variant == "surround_wide":
            base_offsets = [
                [-4.5, 0.0, radius],
                [-3.8, 5.8, radius],
                [-3.1, -5.2, radius],
                [-0.8, 8.2, radius],
                [-1.6, -7.4, radius],
                [2.8, 8.7, radius],
                [1.9, -8.9, radius],
                [6.6, 7.6, radius],
                [7.4, -7.1, radius],
                [11.0, 6.4, radius],
                [10.7, -5.8, radius],
                [15.0, 3.5, radius],
                [15.9, -4.1, radius],
                [19.0, 0.6, radius],
                [3.2, 2.1, radius],
                [4.6, -1.2, radius],
                [6.8, 0.7, radius],
                [8.4, 2.8, radius],
                [9.3, -2.6, radius],
                [12.2, 1.5, radius],
                [13.6, -0.9, radius],
                [2.4, -3.4, radius],
                [4.1, 4.2, radius],
                [5.7, -3.8, radius],
                [7.9, 4.1, radius],
                [9.8, -4.4, radius],
                [11.1, 3.6, radius],
                [12.9, -3.7, radius],
                [14.8, 2.6, radius],
                [16.6, -2.9, radius],
                [5.2, 5.8, radius],
                [8.9, -5.7, radius],
                [11.8, 5.1, radius],
                [14.1, -5.2, radius],
            ]
            offsets = base_offsets[:num_obstacles]
            if len(offsets) < num_obstacles:
                extra_x = torch.linspace(2.0, 22.0, steps=num_obstacles - len(offsets))
                for obs_idx, offset_x in enumerate(extra_x.tolist()):
                    offsets.append([float(offset_x), 2.2 * math.sin(0.8 * obs_idx), radius])
            return torch.tensor(offsets, dtype=torch.float32)

        if layout_variant == "surround_nonconvex":
            offsets = [
                [-2.0, -5.8, radius],
                [-2.0, -2.9, radius],
                [-2.0, 0.0, radius],
                [-2.0, 2.9, radius],
                [-2.0, 5.8, radius],
                [0.6, 7.2, radius],
                [3.2, 7.2, radius],
                [5.8, 7.2, radius],
                [0.8, -7.0, radius],
                [3.6, -7.0, radius],
                [6.4, -7.0, radius],
                [3.1, 2.6, radius],
                [5.0, 2.6, radius],
                [5.0, 4.4, radius],
                [4.0, -4.5, radius],
                [5.8, -4.5, radius],
                [5.8, -2.8, radius],
                [8.4, -2.1, radius],
                [8.4, 0.1, radius],
                [8.4, 2.3, radius],
                [10.3, 4.6, radius],
                [12.9, 4.6, radius],
                [10.4, -4.4, radius],
                [13.0, -4.4, radius],
                [10.1, 2.8, radius],
                [10.3, -2.6, radius],
            ]
            return torch.tensor(offsets[:num_obstacles], dtype=torch.float32)

        if layout_variant == "surround_hybrid":
            offsets = [
                [-4.5, 0.0, radius],
                [-3.8, 5.8, radius],
                [-3.1, -5.2, radius],
                [-0.8, 8.2, radius],
                [-1.6, -7.4, radius],
                [2.8, 8.7, radius],
                [1.9, -8.9, radius],
                [6.6, 7.6, radius],
                [7.4, -7.1, radius],
                [11.0, 6.4, radius],
                [10.7, -5.8, radius],
                [15.0, 3.5, radius],
                [15.9, -4.1, radius],
                [19.0, 0.6, radius],
                [2.6, -3.8, radius],
                [4.2, 4.4, radius],
                [6.0, -4.2, radius],
                [7.6, 3.8, radius],
                [9.3, -4.5, radius],
                [11.4, 3.9, radius],
                [13.1, -3.9, radius],
                [15.2, 2.7, radius],
            ]
            return torch.tensor(offsets[:num_obstacles], dtype=torch.float32)

        if layout_variant == "surrounded_front_open":
            offsets = [
                [3.0, 0.00, radius],
                [-3.0, 2.00, radius],
                [-3.0, -2.00, radius],
                [-1.0, 3.30, radius],
                [-1.0, -3.30, radius],
                [2.0, 2.05, radius],
                [2.0, -2.05, radius],
                [4.2, 2.80, radius],
                [4.2, -2.80, radius],
                [6.2, 1.85, radius],
                [6.2, -1.85, radius],
                [8.6, 2.90, radius],
                [8.6, -2.90, radius],
                [11.0, 1.95, radius],
                [11.0, -1.95, radius],
                [12.5, 2.50, radius],
                [12.5, -2.50, radius],
                [2.4, 5.00, radius],
                [2.4, -5.00, radius],
                [7.2, 7.00, radius],
                [7.2, -7.00, radius],
                [14.2, 1.60, radius],
                [14.2, -1.60, radius],
                [15.0, 3.80, radius],
                [15.0, -3.80, radius],
                [11.0, 5.20, radius],
                [11.0, -5.20, radius],
                [5.0, 6.20, radius],
                [5.0, -6.20, radius],
                [-4.8, 4.20, radius],
                [-4.8, -4.20, radius],
                [-6.8, 2.20, radius],
                [-6.8, -2.20, radius],
                [-8.2, 5.70, radius],
                [-8.2, -5.70, radius],
                [-2.6, 6.60, radius],
                [-2.6, -6.60, radius],
                [0.0, 7.80, radius],
                [0.0, -7.80, radius],
                [10.0, 7.80, radius],
                [10.0, -7.80, radius],
                [13.2, 6.80, radius],
                [13.2, -6.80, radius],
                [16.4, 0.00, radius],
                [16.4, 5.40, radius],
                [16.4, -5.40, radius],
                [6.2, 3.90, radius],
                [6.2, -3.90, radius],
                [8.8, 4.40, radius],
                [8.8, -4.40, radius],
                [3.2, 7.60, radius],
                [3.2, -7.60, radius],
            ]
            return torch.tensor(offsets[:num_obstacles], dtype=torch.float32)

        if layout_variant in ("continuous_avoidance", "rect_blocks_no_wall", "mixed_obstacles_no_wall"):
            offsets = [
                [2.8, 0.00, radius],
                [2.0, 2.20, radius],
                [2.0, -2.20, radius],
                [4.4, 1.55, radius],
                [4.4, -2.35, radius],
                [5.5, 3.20, radius],
                [5.5, -3.60, radius],
                [6.8, -0.75, radius],
                [7.5, 2.45, radius],
                [7.5, -2.95, radius],
                [8.8, 0.95, radius],
                [9.5, 3.50, radius],
                [9.5, -2.30, radius],
                [10.6, -0.90, radius],
                [-2.4, 2.80, radius],
                [-2.4, -2.80, radius],
                [-0.2, 4.40, radius],
                [-0.2, -4.40, radius],
                [2.8, 5.20, radius],
                [2.8, -5.20, radius],
                [6.4, 5.40, radius],
                [6.4, -5.40, radius],
                [9.8, 5.00, radius],
                [9.8, -5.00, radius],
                [11.2, 2.20, radius],
                [11.2, -3.80, radius],
                [12.0, 0.90, radius],
                [12.0, -1.90, radius],
            ]
            return torch.tensor(offsets[:num_obstacles], dtype=torch.float32)

        if layout_variant == "ring_passages":
            ring_specs = [
                (3.6, 6, 0.35),
                (5.8, 8, -0.10),
                (8.0, 10, 0.22),
                (10.2, 12, -0.28),
            ]
            gate_centers = (0.0, 1.18, -1.18, math.pi)
            gate_half_width = 0.26
            x_min, x_max = getattr(self.cfg, "obstacle_x_range", (-10.5, 10.5))
            y_min, y_max = getattr(self.cfg, "obstacle_y_range", (-8.5, 8.5))
            generator = torch.Generator().manual_seed(20260427)

            def angle_diff(a: float, b: float) -> float:
                return abs(math.atan2(math.sin(a - b), math.cos(a - b)))

            offsets: list[list[float]] = []
            min_center_gap = 2.0 * radius + 1.05
            for base_radius, nominal_count, phase in ring_specs:
                for angle in torch.linspace(-math.pi, math.pi, steps=nominal_count + 1)[:-1].tolist():
                    theta = float(angle + phase + (-0.08 + 0.16 * torch.rand(1, generator=generator).item()))
                    if any(angle_diff(theta, gate_center) < gate_half_width for gate_center in gate_centers):
                        continue
                    radial = float(base_radius + (-0.28 + 0.56 * torch.rand(1, generator=generator).item()))
                    cand_x = radial * math.cos(theta)
                    cand_y = radial * math.sin(theta)
                    if cand_x < float(x_min) or cand_x > float(x_max) or cand_y < float(y_min) or cand_y > float(y_max):
                        continue
                    if all(math.hypot(cand_x - ox, cand_y - oy) >= min_center_gap for ox, oy, _ in offsets):
                        offsets.append([cand_x, cand_y, radius])
                        if len(offsets) >= num_obstacles:
                            return torch.tensor(offsets[:num_obstacles], dtype=torch.float32)
            return torch.tensor(offsets[:num_obstacles], dtype=torch.float32)

        x_min, x_max = getattr(self.cfg, "obstacle_x_range", (1.8, 12.0))
        y_min, y_max = getattr(self.cfg, "obstacle_y_range", (-3.2, 3.2))
        x_values = torch.linspace(float(x_min), float(x_max), steps=num_obstacles)
        lanes = [0.0, 0.78 * float(y_max), 0.78 * float(y_min), 0.38 * float(y_max), 0.38 * float(y_min), float(y_max), float(y_min)]
        offsets = []
        for obs_idx, offset_x in enumerate(x_values.tolist()):
            offset_y = lanes[obs_idx % len(lanes)] + 0.18 * math.sin(1.7 * obs_idx)
            offset_y = min(float(y_max), max(float(y_min), offset_y))
            offsets.append([float(offset_x), float(offset_y), radius])
        return torch.tensor(offsets, dtype=torch.float32)

    def _make_template_offsets(self, num_obstacles: int, radius: float) -> torch.Tensor:
        layout_mode = getattr(self.cfg, "obstacle_layout_mode", "radial")
        if layout_mode == "clutter":
            return self._make_clutter_template_offsets(num_obstacles, radius)
        return self._make_radial_template_offsets(num_obstacles, radius)

    def _sample_line_segment(
        self,
        start_xy: tuple[float, float],
        end_xy: tuple[float, float],
        proxy_radius: float,
        spacing: float = 0.55,
    ) -> list[list[float]]:
        start = torch.tensor(start_xy, dtype=torch.float32)
        end = torch.tensor(end_xy, dtype=torch.float32)
        length = torch.linalg.norm(end - start).item()
        steps = max(2, int(math.ceil(length / spacing)) + 1)
        proxies = []
        for alpha in torch.linspace(0.0, 1.0, steps=steps).tolist():
            point = start + alpha * (end - start)
            proxies.append([float(point[0]), float(point[1]), float(proxy_radius)])
        return proxies

    def _make_geometry_proxy_template_offsets(self, num_obstacles: int, radius: float) -> torch.Tensor:
        collision_mode = getattr(self.cfg, "obstacle_collision_mode", "rigid_cylinder")
        layout_variant = getattr(self.cfg, "obstacle_layout_variant", "lanes")
        if collision_mode != "terrain_mesh":
            return self._make_template_offsets(num_obstacles, radius)

        proxies: list[list[float]] = []
        if layout_variant == "u_trap_close_solid":
            proxies.extend(self._sample_line_segment((5.4, -3.8), (5.4, 3.8), 0.45))
            proxies.extend(self._sample_line_segment((-1.2, 3.4), (5.4, 3.4), 0.45))
            proxies.extend(self._sample_line_segment((-1.2, -3.4), (5.4, -3.4), 0.45))
            return torch.tensor(proxies, dtype=torch.float32)

        if layout_variant == "u_wall_solid":
            proxies.extend(self._sample_line_segment((-1.8, -6.4), (-1.8, 6.4), 0.45))
            proxies.extend(self._sample_line_segment((-1.3, 5.8), (9.1, 5.8), 0.45))
            proxies.extend(self._sample_line_segment((-1.3, -5.8), (9.1, -5.8), 0.45))
            proxies.extend([[2.4, 3.2, 0.45], [3.8, -3.0, 0.45], [5.6, 2.0, 0.45], [6.9, -1.8, 0.45]])
            return torch.tensor(proxies, dtype=torch.float32)

        if layout_variant == "surround_nonconvex":
            segments = [
                ((-2.0, -6.6), (-2.0, 6.6), 0.6),
                ((-1.2, 7.2), (7.6, 7.2), 0.6),
                ((-1.0, -7.0), (8.2, -7.0), 0.6),
                ((8.9, 4.6), (14.5, 4.6), 0.6),
                ((8.9, -4.4), (14.7, -4.4), 0.6),
                ((2.2, 2.6), (6.0, 2.6), 0.5),
                ((5.0, 2.6), (5.0, 5.6), 0.5),
                ((3.4, -4.5), (6.4, -4.5), 0.5),
                ((5.8, -5.0), (5.8, -1.4), 0.5),
                ((8.4, -2.8), (8.4, 3.0), 0.6),
                ((8.7, 2.8), (11.7, 2.8), 0.5),
                ((8.8, -2.6), (11.8, -2.6), 0.5),
            ]
            for start_xy, end_xy, proxy_radius in segments:
                proxies.extend(self._sample_line_segment(start_xy, end_xy, proxy_radius))
            return torch.tensor(proxies, dtype=torch.float32)

        if layout_variant == "surround_hybrid":
            proxies.extend(self._make_terrain_clutter_template_offsets(num_obstacles, radius).tolist())
            segments = [
                ((2.6, 5.0), (7.0, 5.0), 0.45),
                ((7.0, -4.6), (11.8, -4.6), 0.45),
                ((13.0, -0.5), (13.0, 3.9), 0.45),
            ]
            for start_xy, end_xy, proxy_radius in segments:
                proxies.extend(self._sample_line_segment(start_xy, end_xy, proxy_radius))
            return torch.tensor(proxies, dtype=torch.float32)

        if layout_variant == "front_dense_hybrid":
            proxies.extend(self._make_terrain_clutter_template_offsets(num_obstacles, radius).tolist())
            segments = [
                ((2.6, 4.8), (6.6, 4.8), 0.5),
                ((6.6, -4.6), (11.0, -4.6), 0.5),
                ((10.7, 4.0), (14.3, 4.0), 0.5),
                ((10.6, -3.8), (13.8, -3.8), 0.5),
                ((15.2, -2.1), (15.2, 2.1), 0.5),
            ]
            for start_xy, end_xy, proxy_radius in segments:
                proxies.extend(self._sample_line_segment(start_xy, end_xy, proxy_radius))
            return torch.tensor(proxies, dtype=torch.float32)

        if layout_variant == "arena_walled":
            # The terrain generator uses a mix of cylinders and elongated boxes.
            # Conservative circular proxies make metrics closer to visual contacts.
            base_offsets = self._make_terrain_clutter_template_offsets(num_obstacles, radius)
            for obs_idx, (offset_x, offset_y, _) in enumerate(base_offsets.tolist()):
                shape_mode = obs_idx % 6
                if shape_mode in (1, 3, 4):
                    proxy_radius = 1.15
                else:
                    proxy_radius = max(radius, radius * 1.22)
                proxies.append([float(offset_x), float(offset_y), float(proxy_radius)])
            return torch.tensor(proxies, dtype=torch.float32)

        if layout_variant == "mixed_obstacles_no_wall":
            base_offsets = self._make_terrain_clutter_template_offsets(num_obstacles, radius)
            for obs_idx, (offset_x, offset_y, _) in enumerate(base_offsets.tolist()):
                shape_mode = obs_idx % 6
                if shape_mode in (1, 3, 4):
                    proxy_radius = 0.72 if shape_mode != 3 else 1.05
                else:
                    proxy_radius = max(radius, radius * 1.20)
                proxies.append([float(offset_x), float(offset_y), float(proxy_radius)])
            return torch.tensor(proxies, dtype=torch.float32)

        if layout_variant == "rect_blocks_no_wall":
            base_offsets = self._make_terrain_clutter_template_offsets(num_obstacles, radius)
            for obs_idx, (offset_x, offset_y, _) in enumerate(base_offsets.tolist()):
                if obs_idx % 4 in (0, 1, 3):
                    proxy_radius = 0.65
                else:
                    proxy_radius = 0.58
                proxies.append([float(offset_x), float(offset_y), float(proxy_radius)])
            return torch.tensor(proxies, dtype=torch.float32)

        return self._make_terrain_clutter_template_offsets(num_obstacles, radius)

    def _refresh_geometry_proxy_tensor(self):
        proxy_offsets = self._make_geometry_proxy_template_offsets(self.active_num_obstacles, self.obstacle_radius)
        if proxy_offsets.numel() == 0:
            self.geometry_proxy_tensor = torch.zeros((self.env.num_envs, 0, 3), dtype=torch.float32)
            return

        env_origins = self.env.scene.env_origins
        proxy_positions = []
        for env_idx in range(self.env.num_envs):
            env_origin = env_origins[env_idx]
            env_proxy = proxy_offsets.clone()
            env_proxy[:, 0] += env_origin[0].item()
            env_proxy[:, 1] += env_origin[1].item()
            proxy_positions.append(env_proxy)
        self.geometry_proxy_tensor = torch.stack(proxy_positions, dim=0)

    def spawn_obstacles(self):
        """Spawn pillar obstacles in the environment."""
        num_envs = self.env.num_envs
        num_obstacles = self.cfg.num_obstacles
        radius = self.cfg.obstacle_radius
        height = self.cfg.obstacle_height

        self.obstacle_positions = []
        self.obstacle_positions_tensor = None
        self.geometry_proxy_tensor = None
        self.obstacle_geoms = []
        self._obstacle_offsets = []

        # IsaacLab replicates env_0 scene prims into the other environments.
        # Use one shared obstacle template so the replicated geometry matches the
        # analytical obstacle tensor used by observations and CBF costs.
        template_offsets = self._make_template_offsets(num_obstacles, radius)
        for _ in range(num_envs):
            self._obstacle_offsets.append(template_offsets.clone())

        collision_mode = getattr(self.cfg, "obstacle_collision_mode", "rigid_cylinder")

        # Spawn physical cylinders only when obstacles are represented as standalone prims.
        # Terrain-mesh obstacles are already fused into /World/ground; the manager only
        # maintains matching analytical centers for LiDAR and CBF costs.
        cylinder_cfg = sim_utils.CylinderCfg(
            radius=radius,
            height=height,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),  # Red color
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=0.8,
                dynamic_friction=0.8,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,  # Static obstacles
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
        )
        static_mesh_cylinder_cfg = sim_utils.MeshCylinderCfg(
            radius=radius,
            height=height,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=0.8,
                dynamic_friction=0.8,
            ),
            rigid_props=None,
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=bool(getattr(self.cfg, "obstacle_collision_enabled", True)),
            ),
        )

        template_origin = self.env.scene.env_origins[0]
        if collision_mode != "terrain_mesh":
            for obs_idx in range(num_obstacles):
                offsets = template_offsets[obs_idx]
                obs_x = template_origin[0].item() + offsets[0].item()
                obs_y = template_origin[1].item() + offsets[1].item()
                obstacle_path = f"/World/envs/env_0/obstacle_{obs_idx}"
                if collision_mode == "static_mesh":
                    spawn_mesh_cylinder(
                        prim_path=obstacle_path,
                        cfg=static_mesh_cylinder_cfg,
                        translation=(obs_x, obs_y, height / 2.0),
                    )
                else:
                    spawn_cylinder(
                        prim_path=obstacle_path,
                        cfg=cylinder_cfg,
                        translation=(obs_x, obs_y, height / 2.0),
                    )

        for env_idx in range(num_envs):
            env_origin = self.env.scene.env_origins[env_idx]
            env_obstacle_paths = []
            obstacles_for_env = []

            for obs_idx in range(num_obstacles):
                offsets = self._obstacle_offsets[env_idx][obs_idx]
                obs_x = env_origin[0].item() + offsets[0].item()
                obs_y = env_origin[1].item() + offsets[1].item()

                env_obstacle_paths.append(f"/World/envs/env_{env_idx}/obstacle_{obs_idx}")
                obstacles_for_env.append([obs_x, obs_y, radius])

            self.obstacle_geoms.append(env_obstacle_paths)
            self.obstacle_positions.append(torch.tensor(obstacles_for_env, dtype=torch.float32))

        if self.obstacle_positions:
            self.obstacle_positions_tensor = torch.stack(self.obstacle_positions, dim=0)
        else:
            self.obstacle_positions_tensor = torch.zeros((0, 0, 3), dtype=torch.float32)

        self.max_num_obstacles = num_obstacles
        self.active_num_obstacles = num_obstacles
        self._refresh_geometry_proxy_tensor()
        if collision_mode == "terrain_mesh":
            self._obstacle_views = []
        else:
            self._obstacle_views = [
                XformPrimView(
                    f"/World/envs/env_.*/obstacle_{obs_idx}",
                    device=self.env.device,
                    validate_xform_ops=False,
                )
                for obs_idx in range(num_obstacles)
            ]
        self.set_active_obstacles(num_obstacles)

    def set_active_obstacles(self, active_num_obstacles: int):
        """Activate a prefix of spawned obstacles and move the rest far away."""
        if not self._obstacle_offsets:
            return

        active_num_obstacles = int(max(1, min(active_num_obstacles, self.max_num_obstacles)))
        if self.active_num_obstacles == active_num_obstacles and self.obstacle_positions_tensor is not None:
            return

        num_envs = self.env.num_envs
        env_origins = self.env.scene.env_origins
        obstacle_positions = []
        world_positions_by_obs = [[] for _ in range(self.max_num_obstacles)]
        hidden_x_base = 100.0
        hidden_y_base = 100.0

        for env_idx in range(num_envs):
            env_origin = env_origins[env_idx]
            env_positions = []
            for obs_idx in range(self.max_num_obstacles):
                if obs_idx < active_num_obstacles:
                    offset = self._obstacle_offsets[env_idx][obs_idx]
                    pos_x = env_origin[0].item() + offset[0].item()
                    pos_y = env_origin[1].item() + offset[1].item()
                else:
                    pos_x = env_origin[0].item() + hidden_x_base + 8.0 * obs_idx
                    pos_y = env_origin[1].item() + hidden_y_base + 8.0 * env_idx

                world_positions_by_obs[obs_idx].append([pos_x, pos_y, self.obstacle_height / 2.0])
                env_positions.append([pos_x, pos_y, self.obstacle_radius])
            obstacle_positions.append(torch.tensor(env_positions, dtype=torch.float32))

        self.obstacle_positions = obstacle_positions
        self.obstacle_positions_tensor = torch.stack(obstacle_positions, dim=0)
        self.active_num_obstacles = active_num_obstacles
        self._refresh_geometry_proxy_tensor()

        if self._obstacle_views:
            orientations = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * num_envs, device=self.env.device)
            for obs_idx, view in enumerate(self._obstacle_views):
                positions = torch.tensor(world_positions_by_obs[obs_idx], dtype=torch.float32, device=self.env.device)
                view.set_world_poses(positions=positions, orientations=orientations)

    def get_obstacle_tensor(self, device):
        """Return obstacle positions as a dense tensor on the requested device."""
        if self.obstacle_positions_tensor is None:
            return torch.zeros((self.env.num_envs, 0, 3), device=device, dtype=torch.float32)
        return self.obstacle_positions_tensor.to(device)

    def get_geometry_proxy_tensor(self, device):
        """Return a unified obstacle proxy tensor for LiDAR, scans and CBF distances."""
        if self.geometry_proxy_tensor is None:
            return self.get_obstacle_tensor(device)
        return self.geometry_proxy_tensor.to(device)

    def get_closest_geometry(self, robot_positions: torch.Tensor):
        """Return closest proxy-surface distance and direction for each robot."""
        geometry = self.get_geometry_proxy_tensor(robot_positions.device)
        if geometry.shape[1] == 0:
            distances = torch.full((robot_positions.shape[0],), 10.0, device=robot_positions.device)
            directions = torch.zeros((robot_positions.shape[0], 2), device=robot_positions.device)
            directions[:, 0] = 1.0
            return distances, directions

        diff = geometry[:, :, :2] - robot_positions.unsqueeze(1)
        center_dist = torch.linalg.norm(diff, dim=-1)
        surface_dist = center_dist - geometry[:, :, 2]
        min_dist, min_idx = surface_dist.min(dim=1)
        batch_idx = torch.arange(robot_positions.shape[0], device=robot_positions.device)
        closest_diff = diff[batch_idx, min_idx]
        closest_center_dist = center_dist[batch_idx, min_idx].clamp_min(1.0e-6)
        directions = closest_diff / closest_center_dist.unsqueeze(-1)
        return min_dist, directions

    def get_closest_obstacle(self, robot_positions: torch.Tensor):
        """
        Get distance and direction to closest obstacle for each robot.

        Args:
            robot_positions: (N, 2) tensor of robot (x, y) positions

        Returns:
            distances: (N,) tensor of distances to closest obstacle surface
            directions: (N, 2) tensor of unit vectors from robot to obstacle
        """
        return self.get_closest_geometry(robot_positions)
