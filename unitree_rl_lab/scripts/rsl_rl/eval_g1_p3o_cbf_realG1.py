"""Record fixed-scenario evaluation videos for realG1 G1 P3O-CBF checkpoints."""

import argparse
import csv
import importlib
import math
import os
import sys

import cv2
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a deployment-oriented realG1 G1 P3O-CBF checkpoint.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
parser.add_argument("--device", type=str, default="cuda:0", help="Device to use.")
parser.add_argument("--steps", type=int, default=600, help="Number of steps to record.")
parser.add_argument("--num_envs", type=int, default=9, help="Number of environments to show.")
parser.add_argument("--follow_robot", action="store_true", default=False, help="Use a chase camera that follows env_0 robot.")
parser.add_argument("--topdown_follow", action="store_true", default=False, help="Use a top-down camera that follows env_0 robot.")
parser.add_argument("--headless", action="store_true", default=False, help="Run headless.")
parser.add_argument("--show_lidar_points", action="store_true", default=False, help="Render Mid360 hit points.")
parser.add_argument("--lidar_ring_vis", action="store_true", default=False, help="Project LiDAR points onto a scan plane for a ring-like top view.")
parser.add_argument("--lidar_ring_z_offset", type=float, default=0.12, help="Vertical offset from robot base for ring-style lidar visualization.")
parser.add_argument("--lidar_vis_max_points", type=int, default=512, help="Maximum number of LiDAR hit points to draw.")
parser.add_argument("--name", type=str, default=None, help="Optional output filename like realg1_front_pillars.mp4.")
parser.add_argument("--no_video", action="store_true", default=False, help="Skip rendering and only write metrics.")
parser.add_argument("--video_fps", type=float, default=5.0, help="Output video fps. Lower fps makes longer videos from the same number of sim steps.")
parser.add_argument("--history_length", type=int, default=5, help="Observation history length.")
parser.add_argument("--omni_point_samples", type=int, default=1024, help="Mid360 sample count used before 64-D compression.")
parser.add_argument("--lidar_feature_dim", type=int, default=64, help="Compressed LiDAR feature dimension.")
parser.add_argument(
    "--omni_pattern_file",
    type=str,
    default="/home/ubuntu/P3O-CBF/OmniPerception/LidarSensor/LidarSensor/sensor_pattern/sensor_lidar/scan_mode/mid360.npy",
    help="Path to OmniPerception Mid360 pattern file.",
)
parser.add_argument("--lidar_max_distance", type=float, default=6.0, help="Maximum LiDAR range.")
parser.add_argument("--compression_fov_deg", type=float, default=180.0, help="Compressed horizontal field-of-view.")
parser.add_argument("--roi_x_min", type=float, default=-0.5)
parser.add_argument("--roi_x_max", type=float, default=6.0)
parser.add_argument("--roi_abs_y_max", type=float, default=3.0)
parser.add_argument("--roi_z_min", type=float, default=-1.0)
parser.add_argument("--roi_z_max", type=float, default=0.8)
parser.add_argument("--min_planar_distance", type=float, default=0.2)
parser.add_argument("--sensor_offset_x", type=float, default=0.10)
parser.add_argument("--sensor_offset_y", type=float, default=0.0)
parser.add_argument("--sensor_offset_z", type=float, default=0.63)
parser.add_argument("--enable_sensor_noise", action="store_true", default=True)
parser.add_argument("--random_distance_noise", type=float, default=0.02)
parser.add_argument("--pixel_dropout_prob", type=float, default=0.01)
parser.add_argument("--sector_dropout_prob", type=float, default=0.10)
parser.add_argument("--sector_dropout_width_deg", type=float, default=8.0)
parser.add_argument("--translation_noise_x", type=float, default=0.015)
parser.add_argument("--translation_noise_y", type=float, default=0.015)
parser.add_argument("--translation_noise_z", type=float, default=0.015)
parser.add_argument("--rotation_noise_roll_deg", type=float, default=2.0)
parser.add_argument("--rotation_noise_pitch_deg", type=float, default=2.0)
parser.add_argument("--rotation_noise_yaw_deg", type=float, default=2.0)
parser.add_argument("--terrain_clutter", action="store_true", default=False, help="Use fused terrain protrusions instead of spawned obstacle prims.")
parser.add_argument("--terrain_obstacles", type=int, default=10, help="Number of terrain protrusion obstacles when --terrain_clutter is enabled.")
parser.add_argument("--terrain_layout", type=str, default="surround_hybrid", choices=["lanes", "surround_wide", "surround_hybrid", "surround_nonconvex", "front_dense_hybrid", "u_wall", "u_wall_solid", "u_trap_close_solid", "end2end_distributed", "end2end_frontclose", "arena_walled"], help="Terrain protrusion layout used for preview videos.")
parser.add_argument("--terrain_size_x", type=float, default=52.0)
parser.add_argument("--terrain_size_y", type=float, default=34.0)
parser.add_argument("--terrain_x_min", type=float, default=-4.5)
parser.add_argument("--terrain_x_max", type=float, default=19.0)
parser.add_argument("--terrain_y_span", type=float, default=9.0)
parser.add_argument("--safety_margin", type=float, default=0.8)
parser.add_argument("--cbf_gamma", type=float, default=0.5)
parser.add_argument("--collision_distance", type=float, default=0.2)
parser.add_argument("--cmd_vx", type=float, default=0.55)
parser.add_argument("--cmd_vy", type=float, default=0.0)
parser.add_argument("--cmd_wz", type=float, default=0.0)
parser.add_argument("--obstacle_radius_scale", type=float, default=1.0)
parser.add_argument(
    "--scenario",
    type=str,
    default="front_pillars",
    choices=[
        "single_far",
        "clutter_field",
        "wide_surround",
        "front_pillars",
        "dense_front",
        "front_wide",
        "semi_enclosure",
        "u_trap",
        "v_trap",
        "corridor",
        "slalom",
    ],
)

args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab.sim as sim_utils
import isaaclab_tasks  # noqa: F401
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from rsl_rl.modules import ActorCriticSafe

from unitree_rl_lab.tasks.locomotion import mdp

SCRIPT_DIR = os.path.dirname(__file__)
PAPER_DIR = os.path.join(SCRIPT_DIR, "paper_2508_07611")
if PAPER_DIR not in sys.path:
    sys.path.insert(0, PAPER_DIR)
from actor_critic_safe_perception import ActorCriticSafePerception, ObsTermSpec  # noqa: E402


def quat_apply(quat_wxyz: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    q_w = quat_wxyz[:, 0:1]
    q_xyz = quat_wxyz[:, 1:4]
    uv = torch.cross(q_xyz.unsqueeze(1).expand_as(vec), vec, dim=-1)
    uuv = torch.cross(q_xyz.unsqueeze(1).expand_as(vec), uv, dim=-1)
    return vec + 2.0 * (q_w.unsqueeze(1) * uv + uuv)


def make_lidar_visualizer():
    return VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/EvalRealG1LidarPoints",
            markers={
                "hit": sim_utils.SphereCfg(
                    radius=0.08,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
            },
        )
    )


def update_lidar_visualization(env, visualizer):
    sensor = env.unwrapped.scene["mid360_lidar"]
    robot = env.unwrapped.scene["robot"]
    hit_points_w = sensor.data.ray_hits_w[0]
    finite_mask = torch.isfinite(hit_points_w).all(dim=-1)
    points = hit_points_w[finite_mask]
    if points.numel() == 0:
        root = robot.data.root_pos_w[0]
        points = root.unsqueeze(0).clone()
    if points.shape[0] > args_cli.lidar_vis_max_points:
        stride = max(1, points.shape[0] // args_cli.lidar_vis_max_points)
        points = points[::stride][: args_cli.lidar_vis_max_points]
    points = points.clone()
    if args_cli.lidar_ring_vis:
        points[:, 2] = robot.data.root_pos_w[0, 2] + args_cli.lidar_ring_z_offset
    else:
        points[:, 2] += 0.04
    visualizer.visualize(points)


def get_scenario_offsets(radius: float, scenario: str) -> torch.Tensor:
    if scenario == "single_far":
        offsets = [[4.2, 0.0, radius]]
    elif scenario == "clutter_field":
        offsets = [
            [1.4, 3.0, radius], [1.6, -3.0, radius], [2.6, 1.8, radius], [2.9, -2.1, radius],
            [4.1, 0.2, radius], [4.4, 2.7, radius], [4.9, -2.9, radius], [6.0, 1.5, radius],
            [6.4, -1.6, radius], [7.6, 3.0, radius], [7.9, -2.8, radius], [8.8, 0.5, radius],
            [10.0, 2.1, radius], [10.2, -2.2, radius], [11.4, 0.8, radius], [12.0, -3.0, radius],
            [12.3, 3.0, radius],
        ]
    elif scenario == "wide_surround":
        offsets = [
            [-1.8, 0.0, radius], [-1.0, 2.4, radius], [-1.0, -2.4, radius], [1.2, 3.0, radius],
            [1.2, -3.0, radius], [2.8, 2.2, radius], [2.8, -2.2, radius], [4.0, 1.1, radius],
            [4.0, -1.1, radius], [5.4, 0.0, radius], [5.8, 2.4, radius], [5.8, -2.4, radius],
        ]
    elif scenario == "front_pillars":
        offsets = [[2.5, 0.0, radius], [4.0, 0.9, radius], [4.0, -0.9, radius], [5.7, 0.0, radius]]
    elif scenario == "dense_front":
        offsets = [[2.2, 0.0, radius], [3.2, 0.8, radius], [3.2, -0.8, radius], [4.2, 0.0, radius], [5.2, 0.8, radius], [5.2, -0.8, radius]]
    elif scenario == "front_wide":
        offsets = [[2.4, 0.0, radius], [3.2, 1.3, radius], [3.2, -1.3, radius], [4.1, 0.6, radius], [4.1, -0.6, radius], [5.0, 1.8, radius], [5.0, -1.8, radius]]
    elif scenario == "semi_enclosure":
        offsets = [[1.8, 1.1, radius], [1.8, -1.1, radius], [2.8, 1.6, radius], [2.8, -1.6, radius], [3.8, 1.0, radius], [3.8, -1.0, radius], [4.7, 0.0, radius]]
    elif scenario == "u_trap":
        wall_gap = radius * 1.85
        side_y = radius * 2.9
        start_x = 0.9
        back_x = start_x + wall_gap * 3.0
        offsets = [
            [start_x, side_y, radius], [start_x + wall_gap, side_y, radius], [start_x + wall_gap * 2.0, side_y, radius], [start_x + wall_gap * 3.0, side_y, radius],
            [start_x, -side_y, radius], [start_x + wall_gap, -side_y, radius], [start_x + wall_gap * 2.0, -side_y, radius], [start_x + wall_gap * 3.0, -side_y, radius],
            [back_x, radius * 2.0, radius], [back_x, 0.0, radius], [back_x, -radius * 2.0, radius],
        ]
    elif scenario == "v_trap":
        offsets = [[2.0, 1.6, radius], [3.0, 1.0, radius], [4.0, 0.4, radius], [2.0, -1.6, radius], [3.0, -1.0, radius], [4.0, -0.4, radius], [4.8, 0.0, radius]]
    elif scenario == "corridor":
        offsets = [[2.5, 1.0, radius], [2.5, -1.0, radius], [4.0, 1.0, radius], [4.0, -1.0, radius], [5.5, 1.0, radius], [5.5, -1.0, radius]]
    else:
        offsets = [[2.0, 0.8, radius], [3.2, -0.8, radius], [4.4, 0.8, radius], [5.6, -0.8, radius], [6.8, 0.8, radius], [8.0, -0.8, radius], [9.2, 0.8, radius]]
    return torch.tensor(offsets, dtype=torch.float32)


def set_camera(env):
    robot = env.unwrapped.scene["robot"]
    if args_cli.topdown_follow:
        root_pos = robot.data.root_pos_w[0]
        env.unwrapped.sim.set_camera_view(
            eye=[float(root_pos[0] + 0.2), float(root_pos[1]), float(root_pos[2] + 16.0)],
            target=[float(root_pos[0] + 1.5), float(root_pos[1]), float(root_pos[2] + 0.2)],
        )
        return
    if args_cli.follow_robot:
        root_pos = robot.data.root_pos_w[0]
        root_quat = robot.data.root_quat_w[0]
        forward = quat_apply(root_quat.unsqueeze(0), torch.tensor([[[1.0, 0.0, 0.0]]], device=root_pos.device))[0, 0]
        env.unwrapped.sim.set_camera_view(
            eye=[float(root_pos[0] - 3.2 * forward[0]), float(root_pos[1] - 3.2 * forward[1] - 1.2), float(root_pos[2] + 2.0)],
            target=[float(root_pos[0] + 2.8 * forward[0]), float(root_pos[1] + 2.8 * forward[1]), float(root_pos[2] + 0.8)],
        )
        return

    env_origins = env.unwrapped.scene.env_origins.cpu()
    center = env_origins.mean(dim=0)
    extent = env_origins.max(dim=0).values - env_origins.min(dim=0).values
    span = max(float(extent[0].item()), float(extent[1].item()), 8.0)
    env.unwrapped.sim.set_camera_view(
        eye=[center[0].item() + span * 0.55 + 8.0, center[1].item() - span * 0.05, center[2].item() + span * 1.2 + 10.0],
        target=[center[0].item() + 3.5, center[1].item(), center[2].item() + 0.5],
    )


def capture_frame(env, lidar_visualizer=None):
    set_camera(env)
    if lidar_visualizer is not None:
        update_lidar_visualization(env, lidar_visualizer)

    # Headless camera output can lag one frame behind camera updates.
    env.unwrapped.sim.render()
    frame = env.render()
    if frame is None:
        raise RuntimeError("env.render() returned None")

    if frame.dtype != "uint8":
        frame = frame.clip(0, 255).astype("uint8")

    # Retry once if the frame is effectively black.
    if float(frame.mean()) < 2.0:
        env.unwrapped.sim.render()
        retry = env.render()
        if retry is not None:
            if retry.dtype != "uint8":
                retry = retry.clip(0, 255).astype("uint8")
            if float(retry.mean()) >= float(frame.mean()):
                frame = retry

    return frame


def build_fixed_obstacles(env, cfg):
    if args_cli.terrain_clutter:
        env.unwrapped.obstacle_manager.spawn_obstacles()
        return

    from isaaclab.sim.spawners.shapes import spawn_cylinder

    radius = cfg.obstacle_radius * args_cli.obstacle_radius_scale
    height = cfg.obstacle_height
    offsets = get_scenario_offsets(radius, args_cli.scenario)
    cylinder_cfg = sim_utils.CylinderCfg(
        radius=radius,
        height=height,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.8,
            dynamic_friction=0.8,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    )

    template_origin = env.unwrapped.scene.env_origins[0]
    for obs_idx, offset in enumerate(offsets):
        spawn_cylinder(
            prim_path=f"/World/envs/env_0/eval_obstacle_{obs_idx}",
            cfg=cylinder_cfg,
            translation=(template_origin[0].item() + offset[0].item(), template_origin[1].item() + offset[1].item(), height / 2.0),
        )

    env_origins = env.unwrapped.scene.env_origins
    obstacle_positions = []
    obstacle_geoms = []
    for env_idx in range(env.unwrapped.num_envs):
        env_origin = env_origins[env_idx]
        world_positions = []
        geom_paths = []
        for obs_idx, offset in enumerate(offsets):
            geom_paths.append(f"/World/envs/env_{env_idx}/eval_obstacle_{obs_idx}")
            world_positions.append([env_origin[0].item() + offset[0].item(), env_origin[1].item() + offset[1].item(), radius])
        obstacle_positions.append(torch.tensor(world_positions, dtype=torch.float32))
        obstacle_geoms.append(geom_paths)

    manager = env.unwrapped.obstacle_manager
    manager.obstacle_positions = obstacle_positions
    manager.obstacle_positions_tensor = torch.stack(obstacle_positions, dim=0).clone()
    manager.obstacle_geoms = obstacle_geoms
    manager._obstacle_offsets = [offsets.clone() for _ in range(env.unwrapped.num_envs)]


def set_realg1_observations(env_cfg):
    from isaaclab.managers import ObservationTermCfg as ObsTerm

    params = {
        "sensor_name": "mid360_lidar",
        "feature_dim": args_cli.lidar_feature_dim,
        "max_distance": args_cli.lidar_max_distance,
        "sensor_offset": (args_cli.sensor_offset_x, args_cli.sensor_offset_y, args_cli.sensor_offset_z),
        "horizontal_fov_deg": args_cli.compression_fov_deg,
        "roi_x_min": args_cli.roi_x_min,
        "roi_x_max": args_cli.roi_x_max,
        "roi_abs_y_max": args_cli.roi_abs_y_max,
        "roi_z_min": args_cli.roi_z_min,
        "roi_z_max": args_cli.roi_z_max,
        "min_planar_distance": args_cli.min_planar_distance,
        "enable_sensor_noise": args_cli.enable_sensor_noise,
        "random_distance_noise": args_cli.random_distance_noise,
        "pixel_dropout_prob": args_cli.pixel_dropout_prob,
        "sector_dropout_prob": args_cli.sector_dropout_prob,
        "sector_dropout_width_deg": args_cli.sector_dropout_width_deg,
        "random_translation_range": (args_cli.translation_noise_x, args_cli.translation_noise_y, args_cli.translation_noise_z),
        "random_rotation_deg_range": (args_cli.rotation_noise_roll_deg, args_cli.rotation_noise_pitch_deg, args_cli.rotation_noise_yaw_deg),
    }
    scale = tuple([1.0 / args_cli.lidar_max_distance] * args_cli.lidar_feature_dim)

    env_cfg.scene.mid360_lidar.max_distance = args_cli.lidar_max_distance
    env_cfg.scene.mid360_lidar.offset.pos = (args_cli.sensor_offset_x, args_cli.sensor_offset_y, args_cli.sensor_offset_z)
    env_cfg.scene.mid360_lidar.pattern_cfg.pattern_file = args_cli.omni_pattern_file
    env_cfg.scene.mid360_lidar.pattern_cfg.samples = args_cli.omni_point_samples

    env_cfg.observations.policy.obstacle_scan = ObsTerm(
        func=mdp.mid360_realg1_range_features_from_raycaster,
        params=params,
        clip=(0.0, args_cli.lidar_max_distance),
        scale=scale,
    )
    env_cfg.observations.policy.history_length = args_cli.history_length
    env_cfg.observations.critic.obstacle_scan = ObsTerm(
        func=mdp.mid360_realg1_range_features_from_raycaster,
        params=params,
        clip=(0.0, args_cli.lidar_max_distance),
        scale=scale,
    )
    env_cfg.observations.critic.history_length = args_cli.history_length


def build_actor_specs() -> list[ObsTermSpec]:
    h = args_cli.history_length
    return [
        ObsTermSpec("base_ang_vel", 3, h),
        ObsTermSpec("projected_gravity", 3, h),
        ObsTermSpec("velocity_commands", 3, h),
        ObsTermSpec("obstacle_scan", args_cli.lidar_feature_dim, h),
        ObsTermSpec("joint_pos_rel", 29, h),
        ObsTermSpec("joint_vel_rel", 29, h),
        ObsTermSpec("last_action", 29, h),
    ]


def build_critic_specs() -> list[ObsTermSpec]:
    h = args_cli.history_length
    return [
        ObsTermSpec("base_lin_vel", 3, h),
        ObsTermSpec("base_ang_vel", 3, h),
        ObsTermSpec("projected_gravity", 3, h),
        ObsTermSpec("velocity_commands", 3, h),
        ObsTermSpec("obstacle_scan", args_cli.lidar_feature_dim, h),
        ObsTermSpec("joint_pos_rel", 29, h),
        ObsTermSpec("joint_vel_rel", 29, h),
        ObsTermSpec("last_action", 29, h),
    ]


def closest_obstacle_geometry(env):
    robot_xy = env.unwrapped.scene["robot"].data.root_pos_w[:, :2]
    return env.unwrapped.obstacle_manager.get_closest_geometry(robot_xy)


def compute_metrics(env):
    robot = env.unwrapped.scene["robot"]
    command = env.unwrapped.command_manager.get_command("base_velocity")[0]
    actual_lin = robot.data.root_lin_vel_b[0, :2]
    actual_yaw = robot.data.root_ang_vel_b[0, 2]
    nearest_distances, nearest_dirs = closest_obstacle_geometry(env)
    nearest_dist = nearest_distances[0]
    nearest_dir = nearest_dirs[0]
    h_value = nearest_dist - args_cli.safety_margin
    h_dot = -torch.sum(actual_lin * nearest_dir, dim=-1)
    dt = float(env.unwrapped.cfg.decimation * env.unwrapped.cfg.sim.dt)
    h_next_est = h_value + dt * h_dot
    cbf_margin = h_next_est - (1.0 - args_cli.cbf_gamma) * h_value
    root_height = robot.data.root_pos_w[0, 2]
    gravity_xy_norm = torch.linalg.norm(robot.data.projected_gravity_b[0, :2])
    return {
        "cmd_vx": command[0].item(),
        "cmd_vy": command[1].item(),
        "cmd_wz": command[2].item(),
        "act_vx": actual_lin[0].item(),
        "act_vy": actual_lin[1].item(),
        "act_wz": actual_yaw.item(),
        "nearest_obs": nearest_dist.item(),
        "h": h_value.item(),
        "h_dot": h_dot.item(),
        "cbf_margin": cbf_margin.item(),
        "unsafe": float(nearest_dist < args_cli.safety_margin),
        "collision": float(nearest_dist < args_cli.collision_distance),
        "root_height": root_height.item(),
        "gravity_xy_norm": gravity_xy_norm.item(),
        "fallen": float((root_height < 0.45) or (gravity_xy_norm > 0.8)),
    }


def main():
    import isaaclab.terrains as terrain_gen
    import unitree_rl_lab.tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from unitree_rl_lab.tasks.locomotion.obstacle_manager import ObstacleManager

    task_name = "Unitree-G1-29dof-ObstacleAvoidance-realG1"
    env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.safety_margin = args_cli.safety_margin
    env_cfg.cbf_gamma = args_cli.cbf_gamma
    env_cfg.collision_distance = args_cli.collision_distance
    env_cfg.num_obstacles = args_cli.terrain_obstacles if args_cli.terrain_clutter else int(get_scenario_offsets(0.3, args_cli.scenario).shape[0])
    if args_cli.terrain_clutter:
        obstacle_env_cfg_mod = importlib.import_module("unitree_rl_lab.tasks.locomotion.robots.g1.29dof.obstacle_env_cfg")
        MeshClutterPillarsTerrainCfg = obstacle_env_cfg_mod.MeshClutterPillarsTerrainCfg
        terrain_rows = int(math.ceil(math.sqrt(args_cli.num_envs)))
        terrain_cols = int(math.ceil(args_cli.num_envs / terrain_rows))
        env_cfg.scene.env_spacing = max(env_cfg.scene.env_spacing, 20.0)
        env_cfg.scene.terrain.max_init_terrain_level = terrain_rows - 1
        env_cfg.scene.terrain.terrain_generator = terrain_gen.TerrainGeneratorCfg(
            size=(args_cli.terrain_size_x, args_cli.terrain_size_y),
            border_width=0.0,
            num_rows=terrain_rows,
            num_cols=terrain_cols,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            difficulty_range=(0.0, 0.0),
            use_cache=False,
            sub_terrains={
                "clutter_pillars": MeshClutterPillarsTerrainCfg(
                    proportion=1.0,
                    num_obstacles=env_cfg.num_obstacles,
                    radius=env_cfg.obstacle_radius,
                    height=env_cfg.obstacle_height,
                    x_range=(args_cli.terrain_x_min, args_cli.terrain_x_max),
                    y_range=(-args_cli.terrain_y_span, args_cli.terrain_y_span),
                    layout_variant=args_cli.terrain_layout,
                ),
            },
        )
        env_cfg.obstacle_layout_mode = "clutter"
        env_cfg.obstacle_layout_variant = args_cli.terrain_layout
        env_cfg.obstacle_x_range = (args_cli.terrain_x_min, args_cli.terrain_x_max)
        env_cfg.obstacle_y_range = (-args_cli.terrain_y_span, args_cli.terrain_y_span)
        env_cfg.obstacle_collision_mode = "terrain_mesh"
    env_cfg.episode_length_s = max(120.0, args_cli.steps * env_cfg.decimation * env_cfg.sim.dt + 5.0)
    set_realg1_observations(env_cfg)

    if hasattr(env_cfg, "events") and hasattr(env_cfg.events, "reset_base"):
        env_cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "base_velocity"):
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (args_cli.cmd_vx, args_cli.cmd_vx)
        env_cfg.commands.base_velocity.ranges.lin_vel_y = (args_cli.cmd_vy, args_cli.cmd_vy)
        env_cfg.commands.base_velocity.ranges.ang_vel_z = (args_cli.cmd_wz, args_cli.cmd_wz)
        env_cfg.commands.base_velocity.resampling_time_range = (120.0, 120.0)
        env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    if hasattr(env_cfg, "terminations"):
        if hasattr(env_cfg.terminations, "base_height"):
            env_cfg.terminations.base_height.params["minimum_height"] = -1.0
        if hasattr(env_cfg.terminations, "bad_orientation"):
            env_cfg.terminations.bad_orientation.params["limit_angle"] = 3.14

    env = gym.make(task_name, cfg=env_cfg, render_mode=None if args_cli.no_video else "rgb_array")
    env.unwrapped.obstacle_manager = ObstacleManager(env.unwrapped, env.unwrapped.cfg)
    build_fixed_obstacles(env, env_cfg)

    num_obs = env.unwrapped.observation_space["policy"].shape[-1]
    num_privileged_obs = env.unwrapped.observation_space["critic"].shape[-1]
    num_actions = env.unwrapped.action_space.shape[-1]
    policy = ActorCriticSafePerception(
        actor_term_specs=build_actor_specs(),
        critic_term_specs=build_critic_specs(),
        num_actions=num_actions,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
        activation="elu",
        init_noise_std=1.0,
        proprio_hidden_dim=128,
        scan_hidden_dim=64,
        rnn_hidden_dim=64,
    ).to(args_cli.device)
    if policy.actor_encoder.input_dim != num_obs:
        raise RuntimeError(f"Actor obs dim mismatch: encoder expects {policy.actor_encoder.input_dim}, env provides {num_obs}")
    if policy.critic_encoder.input_dim != num_privileged_obs:
        raise RuntimeError(f"Critic obs dim mismatch: encoder expects {policy.critic_encoder.input_dim}, env provides {num_privileged_obs}")

    checkpoint = torch.load(args_cli.checkpoint, map_location=args_cli.device, weights_only=False)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()

    reset_result = env.reset()
    obs_dict = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    obs = obs_dict["policy"].to(args_cli.device)
    lidar_visualizer = make_lidar_visualizer() if args_cli.show_lidar_points else None

    video_dir = os.path.join(os.path.dirname(args_cli.checkpoint), "videos")
    os.makedirs(video_dir, exist_ok=True)
    video_name = args_cli.name if args_cli.name else "realg1_eval.mp4"
    if not (video_name.endswith(".mp4") or video_name.endswith(".avi")):
        video_name = f"{video_name}.mp4"
    video_path = os.path.join(video_dir, video_name)
    metrics_path = os.path.join(
        video_dir,
        video_name[:-4] + "_metrics.csv",
    )

    writer = None
    if not args_cli.no_video:
        frame = capture_frame(env, lidar_visualizer)
        height, width = frame.shape[:2]
        if video_name.endswith(".avi"):
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        else:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, float(args_cli.video_fps), (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {video_path}")

    with open(metrics_path, "w", newline="", encoding="utf-8") as metrics_file:
        metrics_writer = csv.DictWriter(
            metrics_file,
            fieldnames=["step", "cmd_vx", "cmd_vy", "cmd_wz", "act_vx", "act_vy", "act_wz", "nearest_obs", "h", "h_dot", "cbf_margin", "unsafe", "collision", "root_height", "gravity_xy_norm", "fallen"],
        )
        metrics_writer.writeheader()
        print(f"[INFO] Recording eval to {video_path}" if not args_cli.no_video else "[INFO] Running metrics-only eval")
        print(f"[INFO] Metrics csv: {metrics_path}")
        print(f"[INFO] Scenario: {args_cli.scenario}")
        print(f"[INFO] Checkpoint: {args_cli.checkpoint}")

        for step in range(args_cli.steps):
            with torch.no_grad():
                actions = policy.act_inference(obs)
            step_result = env.step(actions)
            next_obs_dict = step_result[0]
            obs = next_obs_dict["policy"].to(args_cli.device)
            if writer is not None:
                frame = capture_frame(env, lidar_visualizer)
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            metrics = compute_metrics(env)
            metrics["step"] = step
            metrics_writer.writerow(metrics)
            if step % 100 == 0:
                print(f"[INFO] Step {step}/{args_cli.steps}")

    if writer is not None:
        writer.release()
    env.close()
    print(f"[INFO] Eval saved to {video_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
