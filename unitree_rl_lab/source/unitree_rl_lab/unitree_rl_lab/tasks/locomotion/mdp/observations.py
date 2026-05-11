from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from typing import TYPE_CHECKING

try:
    import torch_fpsample
except ImportError:
    torch_fpsample = None

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_root_yaw(root_quat_w: torch.Tensor) -> torch.Tensor:
    """Extract yaw from Isaac Lab root quaternions in wxyz format."""
    qw = root_quat_w[:, 0]
    qx = root_quat_w[:, 1]
    qy = root_quat_w[:, 2]
    qz = root_quat_w[:, 3]
    return torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def _build_raycast_dirs(root_yaw: torch.Tensor, num_rays: int, fov_deg: float) -> torch.Tensor:
    """Build a fan of 2D ray directions in the world frame."""
    half_fov = math.radians(fov_deg) * 0.5
    ray_offsets = torch.linspace(
        -half_fov,
        half_fov,
        steps=num_rays,
        device=root_yaw.device,
        dtype=root_yaw.dtype,
    )
    ray_angles = root_yaw.unsqueeze(1) + ray_offsets.unsqueeze(0)
    return torch.stack((torch.cos(ray_angles), torch.sin(ray_angles)), dim=-1)


@lru_cache(maxsize=16)
def _load_omni_pattern(pattern_file: str, samples: int) -> tuple[np.ndarray, np.ndarray]:
    pattern = np.load(pattern_file)
    if pattern.ndim != 2 or pattern.shape[1] != 2:
        raise ValueError(f"Expected LiDAR pattern with shape (N, 2), got {pattern.shape}")
    if samples <= 0:
        raise ValueError(f"samples must be positive, got {samples}")
    if samples >= pattern.shape[0]:
        selected = pattern
    else:
        idx = np.linspace(0, pattern.shape[0] - 1, num=samples, dtype=np.int64)
        selected = pattern[idx]
    return selected[:, 0].astype(np.float32), selected[:, 1].astype(np.float32)


def _quat_apply(quat_wxyz: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by quaternions in wxyz format."""
    q_w = quat_wxyz[:, 0:1]
    q_xyz = quat_wxyz[:, 1:4]
    uv = torch.cross(q_xyz.unsqueeze(1).expand_as(vec), vec, dim=-1)
    uuv = torch.cross(q_xyz.unsqueeze(1).expand_as(vec), uv, dim=-1)
    return vec + 2.0 * (q_w.unsqueeze(1) * uv + uuv)


def _quat_conjugate(quat_wxyz: torch.Tensor) -> torch.Tensor:
    quat_conj = quat_wxyz.clone()
    quat_conj[:, 1:] = -quat_conj[:, 1:]
    return quat_conj


def _raycast_against_cylinders(
    ray_origins_xy: torch.Tensor,
    ray_dirs_xy: torch.Tensor,
    obstacle_positions: torch.Tensor,
    max_distance: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Analytical 2D ray-cast against cylindrical pillar obstacles."""
    num_envs, num_rays, _ = ray_dirs_xy.shape
    if obstacle_positions.shape[1] == 0:
        distances = torch.full((num_envs, num_rays), max_distance, device=ray_dirs_xy.device, dtype=ray_dirs_xy.dtype)
        hit_dirs = ray_dirs_xy.clone()
        return distances, hit_dirs

    obstacle_xy = obstacle_positions[:, :, :2]
    obstacle_radius = obstacle_positions[:, :, 2]

    origin_to_center = ray_origins_xy[:, None, None, :] - obstacle_xy[:, None, :, :]
    ray_dirs_expanded = ray_dirs_xy[:, :, None, :]

    b_term = torch.sum(ray_dirs_expanded * origin_to_center, dim=-1)
    c_term = torch.sum(origin_to_center * origin_to_center, dim=-1) - obstacle_radius[:, None, :] ** 2
    discriminant = b_term**2 - c_term

    valid_intersection = discriminant >= 0.0
    sqrt_discriminant = torch.sqrt(torch.clamp(discriminant, min=0.0))
    t1 = -b_term - sqrt_discriminant
    t2 = -b_term + sqrt_discriminant

    inf = torch.full_like(t1, float("inf"))
    t1 = torch.where((t1 > 0.0) & valid_intersection, t1, inf)
    t2 = torch.where((t2 > 0.0) & valid_intersection, t2, inf)
    hit_distance = torch.minimum(t1, t2)
    hit_distance = torch.where(hit_distance.isfinite(), hit_distance, torch.full_like(hit_distance, max_distance))

    min_hit_distance, hit_indices = torch.min(hit_distance, dim=-1)
    min_hit_distance = torch.clamp(min_hit_distance, max=max_distance)

    env_ids = torch.arange(num_envs, device=ray_dirs_xy.device).unsqueeze(1).expand(num_envs, num_rays)
    hit_dirs = ray_dirs_xy[env_ids, torch.arange(num_rays, device=ray_dirs_xy.device).unsqueeze(0).expand(num_envs, num_rays)]
    _ = hit_indices
    return min_hit_distance, hit_dirs


def _raycast_pointcloud_against_cylinders(
    ray_origins_w: torch.Tensor,
    ray_dirs_w: torch.Tensor,
    obstacle_positions: torch.Tensor,
    obstacle_height: float,
    max_distance: float,
) -> torch.Tensor:
    """Analytical 3D ray-cast against finite vertical cylinders.

    Returns hit points in sensor frame as distances along each ray; no-hit rays
    are assigned max_distance.
    """
    num_envs, num_rays, _ = ray_dirs_w.shape
    if obstacle_positions.shape[1] == 0:
        return torch.full((num_envs, num_rays), max_distance, device=ray_dirs_w.device, dtype=ray_dirs_w.dtype)

    ox = ray_origins_w[:, None, None, 0]
    oy = ray_origins_w[:, None, None, 1]
    oz = ray_origins_w[:, None, None, 2]
    dx = ray_dirs_w[:, :, None, 0]
    dy = ray_dirs_w[:, :, None, 1]
    dz = ray_dirs_w[:, :, None, 2]

    cx = obstacle_positions[:, None, :, 0]
    cy = obstacle_positions[:, None, :, 1]
    radius = obstacle_positions[:, None, :, 2]
    z_min = torch.zeros_like(radius)
    z_max = torch.full_like(radius, obstacle_height)

    a = dx * dx + dy * dy
    b = 2.0 * ((ox - cx) * dx + (oy - cy) * dy)
    c = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) - radius * radius

    discriminant = b * b - 4.0 * a * c
    valid = (discriminant >= 0.0) & (a > 1.0e-8)
    sqrt_disc = torch.sqrt(torch.clamp(discriminant, min=0.0))

    inf = torch.full_like(a, float("inf"))
    t1 = torch.where(valid, (-b - sqrt_disc) / (2.0 * a), inf)
    t2 = torch.where(valid, (-b + sqrt_disc) / (2.0 * a), inf)
    z1 = oz + t1 * dz
    z2 = oz + t2 * dz

    t1 = torch.where((t1 > 0.0) & (z1 >= z_min) & (z1 <= z_max), t1, inf)
    t2 = torch.where((t2 > 0.0) & (z2 >= z_min) & (z2 <= z_max), t2, inf)

    side_hit = torch.minimum(t1, t2)
    side_hit = torch.where(side_hit.isfinite(), side_hit, inf)
    min_hit_distance, _ = torch.min(side_hit, dim=-1)
    min_hit_distance = torch.where(min_hit_distance.isfinite(), min_hit_distance, torch.full_like(min_hit_distance, max_distance))
    return torch.clamp(min_hit_distance, max=max_distance)


def obstacle_raycast_scan(
    env: ManagerBasedRLEnv,
    num_rays: int = 9,
    max_distance: float = 6.0,
    fov_deg: float = 180.0,
) -> torch.Tensor:
    """Return a fan of obstacle distances derived from analytical ray-casting against spawned pillars."""
    if not hasattr(env, "obstacle_manager"):
        return torch.full((env.num_envs, num_rays), max_distance, device=env.device)

    robot = env.scene["robot"]
    ray_origins_xy = robot.data.root_pos_w[:, :2]
    root_yaw = _get_root_yaw(robot.data.root_quat_w)
    ray_dirs_xy = _build_raycast_dirs(root_yaw, num_rays, fov_deg)
    obstacle_positions = env.obstacle_manager.get_geometry_proxy_tensor(ray_origins_xy.device)

    distances, _ = _raycast_against_cylinders(ray_origins_xy, ray_dirs_xy, obstacle_positions, max_distance)
    return distances


def omni_lidar_pointcloud(
    env: ManagerBasedRLEnv,
    pattern_file: str = "/home/ubuntu/P3O-CBF/OmniPerception/LidarSensor/LidarSensor/sensor_pattern/sensor_lidar/scan_mode/mid360.npy",
    samples: int = 512,
    max_distance: float = 6.0,
    sensor_offset: tuple[float, float, float] = (0.10, 0.0, 0.63),
) -> torch.Tensor:
    """Generate a flattened local point cloud using OmniPerception's Livox pattern.

    This keeps the existing obstacle environment and analytically ray-casts
    against the spawned cylinders, but replaces the low-dimensional fan scan
    with a Mid-360 style point-cloud observation.
    """
    robot = env.scene["robot"]
    device = robot.data.root_pos_w.device
    dtype = robot.data.root_pos_w.dtype

    theta_np, phi_np = _load_omni_pattern(pattern_file, samples)
    theta = torch.as_tensor(theta_np, device=device, dtype=dtype)
    phi = torch.as_tensor(phi_np, device=device, dtype=dtype)

    local_dirs = torch.stack(
        (
            torch.cos(theta) * torch.cos(phi),
            torch.sin(theta) * torch.cos(phi),
            torch.sin(phi),
        ),
        dim=-1,
    ).unsqueeze(0).expand(env.num_envs, -1, -1)

    root_quat = robot.data.root_quat_w
    sensor_offset_local = torch.tensor(sensor_offset, device=device, dtype=dtype).unsqueeze(0).expand(env.num_envs, -1)
    sensor_origin_w = robot.data.root_pos_w + _quat_apply(root_quat, sensor_offset_local.unsqueeze(1)).squeeze(1)
    ray_dirs_w = _quat_apply(root_quat, local_dirs)

    if not hasattr(env, "obstacle_manager"):
        return (local_dirs * max_distance).reshape(env.num_envs, -1)

    obstacle_positions = env.obstacle_manager.get_geometry_proxy_tensor(device)
    obstacle_height = float(getattr(env.unwrapped.cfg, "obstacle_height", 2.0))
    hit_distances = _raycast_pointcloud_against_cylinders(
        sensor_origin_w,
        ray_dirs_w,
        obstacle_positions,
        obstacle_height=obstacle_height,
        max_distance=max_distance,
    )
    pointcloud_local = local_dirs * hit_distances.unsqueeze(-1)
    return pointcloud_local.reshape(env.num_envs, -1)


def _batched_fps_fallback(points: torch.Tensor, valid_mask: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Vectorized batched FPS fallback when torch_fpsample is unavailable."""
    batch_size, max_points, _ = points.shape
    device = points.device
    dtype = points.dtype
    points = points.contiguous()
    valid_mask = valid_mask.contiguous()

    if max_points == 0 or num_samples <= 0:
        return torch.zeros(batch_size, max(num_samples, 0), 3, device=device, dtype=dtype)

    valid_counts = valid_mask.sum(dim=1)
    any_valid = valid_counts > 0
    if not torch.any(any_valid):
        return torch.zeros(batch_size, num_samples, 3, device=device, dtype=dtype)

    range_sq = torch.sum(points * points, dim=-1)
    invalid_large = torch.full_like(range_sq, float("inf"))
    first_idx = torch.argmin(torch.where(valid_mask, range_sq, invalid_large), dim=1)
    first_idx = torch.where(any_valid, first_idx, torch.zeros_like(first_idx))

    batch_ids = torch.arange(batch_size, device=device)
    min_dist = torch.full((batch_size, max_points), float("inf"), device=device, dtype=dtype)
    selected_indices = torch.zeros((batch_size, num_samples), device=device, dtype=torch.long)
    current_idx = first_idx

    for i in range(num_samples):
        selected_indices[:, i] = current_idx
        current_points = points[batch_ids, current_idx].unsqueeze(1)
        sq_dist = torch.sum((points - current_points) ** 2, dim=-1)
        min_dist = torch.minimum(min_dist, sq_dist)
        masked_dist = torch.where(valid_mask, min_dist, torch.full_like(min_dist, -1.0))
        current_idx = torch.argmax(masked_dist, dim=1)

    sampled = points[batch_ids.unsqueeze(1), selected_indices]
    keep_mask = torch.arange(num_samples, device=device).unsqueeze(0) < valid_counts.unsqueeze(1)
    return sampled * keep_mask.unsqueeze(-1)


def _fps_sample_pointcloud(points: torch.Tensor, valid_mask: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Sample a fixed-size point set with FPS, zero-padding invalid outputs."""
    batch_size, max_points, _ = points.shape
    device = points.device
    dtype = points.dtype

    if max_points == 0 or num_samples <= 0:
        return torch.zeros(batch_size, max(num_samples, 0), 3, device=device, dtype=dtype)

    valid_counts = valid_mask.sum(dim=1)
    any_valid = valid_counts > 0
    if not torch.any(any_valid):
        return torch.zeros(batch_size, num_samples, 3, device=device, dtype=dtype)

    # Fill padding points with the first valid point to avoid degenerate FPS distance fields.
    first_valid_idx = torch.argmax(valid_mask.long(), dim=1)
    batch_ids = torch.arange(batch_size, device=device)
    first_valid_points = points[batch_ids, first_valid_idx].unsqueeze(1).expand(-1, max_points, -1)
    padded_points = torch.where(valid_mask.unsqueeze(-1), points, first_valid_points)

    if torch_fpsample is not None:
        k_eff = min(num_samples, max_points)
        sample_points_input = padded_points
        sample_valid_counts = valid_counts
        used_cpu_fps = False
        if padded_points.is_cuda:
            try:
                sample_points_input = sample_points_input.contiguous()
                sampled_points, sampled_indices = torch_fpsample.sample(sample_points_input, k_eff)
            except (NotImplementedError, RuntimeError) as e:
                print(f"[WARN] torch_fpsample.sample failed on CUDA ({type(e).__name__}: {e}), using pure-PyTorch fallback")
                return _batched_fps_fallback(padded_points, valid_mask, num_samples)
        else:
            sampled_points, sampled_indices = torch_fpsample.sample(sample_points_input, k_eff)

        if used_cpu_fps:
            sampled_points = sampled_points.to(device=device, dtype=dtype)
            sampled_indices = sampled_indices.to(device=device)

        invalid_samples = sampled_indices >= sample_valid_counts.to(device=device).unsqueeze(1)
        sampled_points = sampled_points.masked_fill(invalid_samples.unsqueeze(-1), 0.0)
        if k_eff < num_samples:
            output = torch.zeros(batch_size, num_samples, 3, device=device, dtype=dtype)
            output[:, :k_eff, :] = sampled_points
            return output
        return sampled_points

    return _batched_fps_fallback(padded_points, valid_mask, num_samples)


def omni_lidar_range_features(
    env: ManagerBasedRLEnv,
    pattern_file: str = "/home/ubuntu/P3O-CBF/OmniPerception/LidarSensor/LidarSensor/sensor_pattern/sensor_lidar/scan_mode/mid360.npy",
    samples: int = 512,
    feature_dim: int = 64,
    max_distance: float = 6.0,
    sensor_offset: tuple[float, float, float] = (0.10, 0.0, 0.63),
) -> torch.Tensor:
    """Compress an Omni/Mid360 point cloud into a compact 64-D LiDAR feature.

    The paper-style policy uses a compact LiDAR embedding before the policy
    trunk. This observation keeps the Mid360 sampling geometry but pools dense
    ray distances into a fixed low-dimensional feature so rollout storage does
    not carry the full flattened point cloud.
    """
    if feature_dim <= 0:
        raise ValueError(f"feature_dim must be positive, got {feature_dim}")

    pointcloud = omni_lidar_pointcloud(
        env,
        pattern_file=pattern_file,
        samples=samples,
        max_distance=max_distance,
        sensor_offset=sensor_offset,
    ).view(env.num_envs, -1, 3)
    ranges = torch.linalg.norm(pointcloud, dim=-1).clamp(max=max_distance)
    num_samples = ranges.shape[1]

    if num_samples == feature_dim:
        return ranges
    if num_samples < feature_dim:
        pad = torch.full(
            (env.num_envs, feature_dim - num_samples),
            max_distance,
            device=ranges.device,
            dtype=ranges.dtype,
        )
        return torch.cat((ranges, pad), dim=-1)

    # Deterministic angular-order pooling: each feature stores the closest
    # return inside a contiguous block of Mid360 rays.
    boundaries = torch.linspace(0, num_samples, feature_dim + 1, device=ranges.device)
    features = []
    for i in range(feature_dim):
        start = int(torch.floor(boundaries[i]).item())
        end = int(torch.ceil(boundaries[i + 1]).item())
        end = max(start + 1, min(end, num_samples))
        features.append(ranges[:, start:end].amin(dim=1))
    return torch.stack(features, dim=-1)


def omni_lidar_realg1_pointcloud_fps(
    env: ManagerBasedRLEnv,
    pattern_file: str = "/home/ubuntu/P3O-CBF/OmniPerception/LidarSensor/LidarSensor/sensor_pattern/sensor_lidar/scan_mode/mid360.npy",
    samples: int = 1024,
    num_fps_points: int = 128,
    max_distance: float = 6.0,
    sensor_offset: tuple[float, float, float] = (0.10, 0.0, 0.63),
    horizontal_fov_deg: float = 180.0,
    roi_x_min: float = -0.5,
    roi_x_max: float = 6.0,
    roi_abs_y_max: float = 3.0,
    roi_z_min: float = -1.0,
    roi_z_max: float = 0.8,
    min_planar_distance: float = 0.2,
    enable_sensor_noise: bool = True,
    random_distance_noise: float = 0.02,
    pixel_dropout_prob: float = 0.01,
    sector_dropout_prob: float = 0.10,
    sector_dropout_width_deg: float = 8.0,
    random_translation_range: tuple[float, float, float] = (0.015, 0.015, 0.015),
    random_rotation_deg_range: tuple[float, float, float] = (2.0, 2.0, 2.0),
) -> torch.Tensor:
    """Paper-style Mid360 point cloud with ROI filtering, corruption and FPS downsampling."""
    pointcloud_sensor = omni_lidar_pointcloud(
        env,
        pattern_file=pattern_file,
        samples=samples,
        max_distance=max_distance,
        sensor_offset=sensor_offset,
    ).view(env.num_envs, -1, 3)

    points_base = _apply_realg1_mid360_corruption(
        pointcloud_sensor=pointcloud_sensor,
        max_distance=max_distance,
        sensor_offset=sensor_offset,
        enable_sensor_noise=enable_sensor_noise,
        random_distance_noise=random_distance_noise,
        pixel_dropout_prob=pixel_dropout_prob,
        sector_dropout_prob=sector_dropout_prob,
        sector_dropout_width_deg=sector_dropout_width_deg,
        random_translation_range=random_translation_range,
        random_rotation_deg_range=random_rotation_deg_range,
    )

    x = points_base[..., 0]
    y = points_base[..., 1]
    z = points_base[..., 2]
    planar_range = torch.sqrt(torch.clamp(x * x + y * y, min=1.0e-9))
    theta = torch.atan2(y, x)
    half_fov = math.radians(horizontal_fov_deg) * 0.5
    valid = (
        (x >= roi_x_min)
        & (x <= roi_x_max)
        & (torch.abs(y) <= roi_abs_y_max)
        & (z >= roi_z_min)
        & (z <= roi_z_max)
        & (planar_range >= min_planar_distance)
        & (planar_range <= max_distance)
        & (torch.abs(theta) <= half_fov)
    )

    sampled = _fps_sample_pointcloud(points_base, valid, num_fps_points)
    return sampled.reshape(env.num_envs, -1)


def _euler_xyz_to_matrix(angles: torch.Tensor) -> torch.Tensor:
    """Convert XYZ Euler angles to rotation matrices."""
    roll = angles[:, 0]
    pitch = angles[:, 1]
    yaw = angles[:, 2]

    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)

    row0 = torch.stack((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr), dim=-1)
    row1 = torch.stack((sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr), dim=-1)
    row2 = torch.stack((-sp, cp * sr, cp * cr), dim=-1)
    return torch.stack((row0, row1, row2), dim=1)


def _apply_realg1_mid360_corruption(
    pointcloud_sensor: torch.Tensor,
    max_distance: float,
    sensor_offset: tuple[float, float, float],
    enable_sensor_noise: bool,
    random_distance_noise: float,
    pixel_dropout_prob: float,
    sector_dropout_prob: float,
    sector_dropout_width_deg: float,
    random_translation_range: tuple[float, float, float],
    random_rotation_deg_range: tuple[float, float, float],
) -> torch.Tensor:
    """Approximate deployment-side Mid360 corruption before range64 compression."""
    num_envs, _, _ = pointcloud_sensor.shape
    device = pointcloud_sensor.device
    dtype = pointcloud_sensor.dtype

    points = pointcloud_sensor

    offset = torch.tensor(sensor_offset, device=device, dtype=dtype).view(1, 1, 3).expand(num_envs, -1, -1)
    if enable_sensor_noise:
        trans_range = torch.tensor(random_translation_range, device=device, dtype=dtype).view(1, 1, 3)
        trans_noise = (2.0 * torch.rand(num_envs, 1, 3, device=device, dtype=dtype) - 1.0) * trans_range
        offset = offset + trans_noise

        rot_range = torch.tensor(random_rotation_deg_range, device=device, dtype=dtype) * (math.pi / 180.0)
        euler_noise = (2.0 * torch.rand(num_envs, 3, device=device, dtype=dtype) - 1.0) * rot_range
        rot_mat = _euler_xyz_to_matrix(euler_noise)
        points = torch.matmul(points, rot_mat.transpose(1, 2))

    points_base = points + offset

    if enable_sensor_noise and random_distance_noise > 0.0:
        ray_norm = torch.linalg.norm(points, dim=-1, keepdim=True).clamp_min(1.0e-6)
        ray_dir = points / ray_norm
        distance_noise = torch.randn_like(ray_norm) * random_distance_noise
        points_base = points_base + ray_dir * distance_noise

    if enable_sensor_noise and pixel_dropout_prob > 0.0:
        dropout_mask = torch.rand(num_envs, points_base.shape[1], 1, device=device, dtype=dtype) < pixel_dropout_prob
        far_points = points_base / torch.linalg.norm(points_base, dim=-1, keepdim=True).clamp_min(1.0e-6) * max_distance
        points_base = torch.where(dropout_mask, far_points, points_base)

    if enable_sensor_noise and sector_dropout_prob > 0.0 and sector_dropout_width_deg > 0.0:
        planar_theta = torch.atan2(points_base[..., 1], points_base[..., 0])
        half_width = math.radians(sector_dropout_width_deg) * 0.5
        sector_centers = (2.0 * torch.rand(num_envs, 1, device=device, dtype=dtype) - 1.0) * math.pi
        sector_active = torch.rand(num_envs, 1, device=device, dtype=dtype) < sector_dropout_prob
        theta_diff = torch.atan2(torch.sin(planar_theta - sector_centers), torch.cos(planar_theta - sector_centers))
        sector_mask = sector_active.unsqueeze(-1) & (torch.abs(theta_diff).unsqueeze(-1) <= half_width)
        far_points = points_base / torch.linalg.norm(points_base, dim=-1, keepdim=True).clamp_min(1.0e-6) * max_distance
        points_base = torch.where(sector_mask, far_points, points_base)

    return points_base


def omni_lidar_realg1_range_features(
    env: ManagerBasedRLEnv,
    pattern_file: str = "/home/ubuntu/P3O-CBF/OmniPerception/LidarSensor/LidarSensor/sensor_pattern/sensor_lidar/scan_mode/mid360.npy",
    samples: int = 1024,
    feature_dim: int = 64,
    max_distance: float = 6.0,
    sensor_offset: tuple[float, float, float] = (0.10, 0.0, 0.63),
    horizontal_fov_deg: float = 180.0,
    roi_x_min: float = -0.5,
    roi_x_max: float = 6.0,
    roi_abs_y_max: float = 3.0,
    roi_z_min: float = -1.0,
    roi_z_max: float = 0.8,
    min_planar_distance: float = 0.2,
    enable_sensor_noise: bool = True,
    random_distance_noise: float = 0.02,
    pixel_dropout_prob: float = 0.01,
    sector_dropout_prob: float = 0.10,
    sector_dropout_width_deg: float = 8.0,
    random_translation_range: tuple[float, float, float] = (0.015, 0.015, 0.015),
    random_rotation_deg_range: tuple[float, float, float] = (2.0, 2.0, 2.0),
) -> torch.Tensor:
    """Compress Mid360-style pointclouds into deployment-oriented 64-D realG1 features.

    The output is defined in the robot base frame so the same compression can be
    reproduced later inside rl_sar on the real G1.
    """
    if feature_dim <= 0:
        raise ValueError(f"feature_dim must be positive, got {feature_dim}")

    pointcloud_sensor = omni_lidar_pointcloud(
        env,
        pattern_file=pattern_file,
        samples=samples,
        max_distance=max_distance,
        sensor_offset=sensor_offset,
    ).view(env.num_envs, -1, 3)

    points_base = _apply_realg1_mid360_corruption(
        pointcloud_sensor=pointcloud_sensor,
        max_distance=max_distance,
        sensor_offset=sensor_offset,
        enable_sensor_noise=enable_sensor_noise,
        random_distance_noise=random_distance_noise,
        pixel_dropout_prob=pixel_dropout_prob,
        sector_dropout_prob=sector_dropout_prob,
        sector_dropout_width_deg=sector_dropout_width_deg,
        random_translation_range=random_translation_range,
        random_rotation_deg_range=random_rotation_deg_range,
    )

    x = points_base[..., 0]
    y = points_base[..., 1]
    z = points_base[..., 2]
    planar_range = torch.sqrt(torch.clamp(x * x + y * y, min=1.0e-9))
    theta = torch.atan2(y, x)

    half_fov = math.radians(horizontal_fov_deg) * 0.5
    valid = (
        (x >= roi_x_min)
        & (x <= roi_x_max)
        & (torch.abs(y) <= roi_abs_y_max)
        & (z >= roi_z_min)
        & (z <= roi_z_max)
        & (planar_range >= min_planar_distance)
        & (planar_range <= max_distance)
        & (torch.abs(theta) <= half_fov)
    )

    bin_boundaries = torch.linspace(-half_fov, half_fov, feature_dim + 1, device=points_base.device, dtype=points_base.dtype)
    features = torch.full((env.num_envs, feature_dim), max_distance, device=points_base.device, dtype=points_base.dtype)
    inf = torch.full_like(planar_range, float("inf"))

    for i in range(feature_dim):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]
        in_bin = valid & (theta >= lower) & (theta < upper if i < feature_dim - 1 else theta <= upper)
        bin_ranges = torch.where(in_bin, planar_range, inf)
        features[:, i] = torch.where(
            torch.any(in_bin, dim=1),
            torch.amin(bin_ranges, dim=1),
            torch.full((env.num_envs,), max_distance, device=points_base.device, dtype=points_base.dtype),
        )

    features = torch.clamp(features, min=min_planar_distance, max=max_distance)
    all_invalid = ~torch.any(valid, dim=1)
    features[all_invalid] = max_distance
    return features


def mid360_realg1_range_features_from_raycaster(
    env: ManagerBasedRLEnv,
    sensor_name: str = "mid360_lidar",
    feature_dim: int = 64,
    max_distance: float = 6.0,
    sensor_offset: tuple[float, float, float] = (0.10, 0.0, 0.63),
    horizontal_fov_deg: float = 180.0,
    roi_x_min: float = -0.5,
    roi_x_max: float = 6.0,
    roi_abs_y_max: float = 3.0,
    roi_z_min: float = -1.0,
    roi_z_max: float = 0.8,
    min_planar_distance: float = 0.2,
    enable_sensor_noise: bool = True,
    random_distance_noise: float = 0.02,
    pixel_dropout_prob: float = 0.01,
    sector_dropout_prob: float = 0.10,
    sector_dropout_width_deg: float = 8.0,
    random_translation_range: tuple[float, float, float] = (0.015, 0.015, 0.015),
    random_rotation_deg_range: tuple[float, float, float] = (2.0, 2.0, 2.0),
) -> torch.Tensor:
    """Compress Mid360 ray hits from an IsaacLab RayCaster into realG1 deployment features."""
    if feature_dim <= 0:
        raise ValueError(f"feature_dim must be positive, got {feature_dim}")

    sensor = env.scene[sensor_name]
    sensor_pos_w = sensor.data.pos_w
    sensor_quat_w = sensor.data.quat_w
    hit_points_w = sensor.data.ray_hits_w

    sensor_pos_expanded = sensor_pos_w.unsqueeze(1).expand_as(hit_points_w)
    hit_vectors_w = hit_points_w - sensor_pos_expanded
    hit_vectors_w = torch.where(torch.isfinite(hit_vectors_w), hit_vectors_w, torch.zeros_like(hit_vectors_w))
    quat_inv = _quat_conjugate(sensor_quat_w)
    pointcloud_sensor = _quat_apply(
        quat_inv,
        hit_vectors_w,
    )

    no_hit_mask = torch.any(torch.isinf(hit_points_w), dim=-1, keepdim=True)
    if torch.any(no_hit_mask):
        ray_norm = torch.linalg.norm(pointcloud_sensor, dim=-1, keepdim=True).clamp_min(1.0e-6)
        far_points_sensor = pointcloud_sensor / ray_norm * max_distance
        pointcloud_sensor = torch.where(no_hit_mask, far_points_sensor, pointcloud_sensor)

    points_base = _apply_realg1_mid360_corruption(
        pointcloud_sensor=pointcloud_sensor,
        max_distance=max_distance,
        sensor_offset=sensor_offset,
        enable_sensor_noise=enable_sensor_noise,
        random_distance_noise=random_distance_noise,
        pixel_dropout_prob=pixel_dropout_prob,
        sector_dropout_prob=sector_dropout_prob,
        sector_dropout_width_deg=sector_dropout_width_deg,
        random_translation_range=random_translation_range,
        random_rotation_deg_range=random_rotation_deg_range,
    )

    x = points_base[..., 0]
    y = points_base[..., 1]
    z = points_base[..., 2]
    planar_range = torch.sqrt(torch.clamp(x * x + y * y, min=1.0e-9))
    theta = torch.atan2(y, x)

    half_fov = math.radians(horizontal_fov_deg) * 0.5
    valid = (
        (x >= roi_x_min)
        & (x <= roi_x_max)
        & (torch.abs(y) <= roi_abs_y_max)
        & (z >= roi_z_min)
        & (z <= roi_z_max)
        & (planar_range >= min_planar_distance)
        & (planar_range <= max_distance)
        & (torch.abs(theta) <= half_fov)
    )

    bin_boundaries = torch.linspace(-half_fov, half_fov, feature_dim + 1, device=points_base.device, dtype=points_base.dtype)
    features = torch.full((env.num_envs, feature_dim), max_distance, device=points_base.device, dtype=points_base.dtype)
    inf = torch.full_like(planar_range, float("inf"))

    for i in range(feature_dim):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]
        in_bin = valid & (theta >= lower) & (theta < upper if i < feature_dim - 1 else theta <= upper)
        bin_ranges = torch.where(in_bin, planar_range, inf)
        features[:, i] = torch.where(
            torch.any(in_bin, dim=1),
            torch.amin(bin_ranges, dim=1),
            torch.full((env.num_envs,), max_distance, device=points_base.device, dtype=points_base.dtype),
        )

    features = torch.clamp(features, min=min_planar_distance, max=max_distance)
    all_invalid = ~torch.any(valid, dim=1)
    features[all_invalid] = max_distance
    return features


def closest_mid360_obstacle(
    env: ManagerBasedRLEnv,
    sensor_name: str = "mid360_lidar",
    max_distance: float = 6.0,
    horizontal_fov_deg: float = 180.0,
    roi_x_min: float = -0.5,
    roi_x_max: float = 6.0,
    roi_abs_y_max: float = 3.0,
    roi_z_min: float = -1.0,
    roi_z_max: float = 0.8,
    min_planar_distance: float = 0.2,
    robot_body_radius: float = 0.4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Closest obstacle surface estimate from real Mid360 ray hits.

    The returned distance is derived only from the ray-caster point cloud. It
    intentionally does not use obstacle-manager geometry, so P3O/CBF costs stay
    tied to the same sensing signal used by deployment.
    """
    robot = env.scene["robot"]
    try:
        sensor = env.scene[sensor_name]
    except KeyError as exc:
        raise RuntimeError(f"Mid360 sensor '{sensor_name}' is required for lidar-derived safety distance.") from exc
    hit_points_w = sensor.data.ray_hits_w
    sensor_pos_w = sensor.data.pos_w

    hit_vectors_w = hit_points_w - sensor_pos_w.unsqueeze(1).expand_as(hit_points_w)
    finite = torch.all(torch.isfinite(hit_vectors_w), dim=-1)
    base_quat_inv = _quat_conjugate(robot.data.root_quat_w)
    points_base = _quat_apply(base_quat_inv, hit_vectors_w)

    x = points_base[..., 0]
    y = points_base[..., 1]
    z = points_base[..., 2]
    planar_range = torch.sqrt(torch.clamp(x * x + y * y, min=1.0e-9))
    theta = torch.atan2(y, x)
    half_fov = math.radians(horizontal_fov_deg) * 0.5

    valid = (
        finite
        & (x >= roi_x_min)
        & (x <= roi_x_max)
        & (torch.abs(y) <= roi_abs_y_max)
        & (z >= roi_z_min)
        & (z <= roi_z_max)
        & (planar_range >= min_planar_distance)
        & (planar_range <= max_distance)
        & (torch.abs(theta) <= half_fov)
    )

    masked_range = torch.where(valid, planar_range, torch.full_like(planar_range, float("inf")))
    nearest_planar, nearest_idx = torch.min(masked_range, dim=1)
    has_hit = torch.isfinite(nearest_planar)

    env_ids = torch.arange(env.num_envs, device=points_base.device)
    nearest_xy = points_base[env_ids, nearest_idx, :2]
    nearest_xy = torch.where(has_hit.unsqueeze(1), nearest_xy, torch.tensor([max_distance, 0.0], device=points_base.device, dtype=points_base.dtype))
    nearest_norm = torch.linalg.norm(nearest_xy, dim=-1).clamp_min(1.0e-6)
    directions = nearest_xy / nearest_norm.unsqueeze(-1)
    distances = torch.clamp(nearest_planar - robot_body_radius, min=0.0, max=max_distance)
    distances = torch.where(has_hit, distances, torch.full_like(distances, max_distance))
    return distances, directions


def closest_obstacle_raycast(
    env: ManagerBasedRLEnv,
    num_rays: int = 9,
    max_distance: float = 6.0,
    fov_deg: float = 180.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return closest obstacle distance and corresponding ray direction for each environment."""
    robot = env.scene["robot"]
    ray_origins_xy = robot.data.root_pos_w[:, :2]
    root_yaw = _get_root_yaw(robot.data.root_quat_w)
    ray_dirs_xy = _build_raycast_dirs(root_yaw, num_rays, fov_deg)

    if not hasattr(env, "obstacle_manager"):
        distances = torch.full((env.num_envs,), max_distance, device=env.device)
        default_dirs = ray_dirs_xy[:, num_rays // 2, :]
        return distances, default_dirs

    obstacle_positions = env.obstacle_manager.get_geometry_proxy_tensor(ray_origins_xy.device)
    ray_distances, ray_dirs = _raycast_against_cylinders(ray_origins_xy, ray_dirs_xy, obstacle_positions, max_distance)

    closest_distance, ray_indices = torch.min(ray_distances, dim=1)
    env_ids = torch.arange(env.num_envs, device=env.device)
    closest_dirs = ray_dirs[env_ids, ray_indices]
    return closest_distance, closest_dirs


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase
