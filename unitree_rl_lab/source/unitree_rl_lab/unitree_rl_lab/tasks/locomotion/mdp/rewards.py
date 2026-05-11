from __future__ import annotations

import torch
from typing import TYPE_CHECKING

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""
Joint penalties.
"""


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def stand_still(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    reward = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return reward * (cmd_norm < 0.1)


"""
Robot.
"""


def orientation_l2(
    env: ManagerBasedRLEnv, desired_gravity: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward the agent for aligning its gravity with the desired gravity vector using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    desired_gravity = torch.tensor(desired_gravity, device=env.device)
    cos_dist = torch.sum(asset.data.projected_gravity_b * desired_gravity, dim=-1)  # cosine distance
    normalized = 0.5 * cos_dist + 0.5  # map from [-1, 1] to [0, 1]
    return torch.square(normalized)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)


"""
Feet rewards.
"""


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footpos_translated[:, i, :])
        footvel_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footvel_translated[:, i, :])
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def feet_too_near(
    env: ManagerBasedRLEnv, threshold: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity"
) -> torch.Tensor:
    """
    Reward for feet contact when the command is zero.
    """
    # asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    reward = torch.sum(is_contact, dim=-1).float()
    return reward * (command_norm < 0.1)


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


"""
Feet Gait rewards.
"""


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward


"""
Navigation rewards.
"""


def target_position_distance(
    env: ManagerBasedRLEnv, command_name: str, threshold: float = 0.5
) -> torch.Tensor:
    """Reward for getting closer to target position."""
    target_pos = env.command_manager.get_command(command_name)
    robot_pos = env.scene["robot"].data.root_pos_w[:, :2]
    distance = torch.norm(target_pos - robot_pos, dim=-1)
    # Reward for being close to target
    reward = torch.exp(-distance)
    # Bonus for reaching target
    reward += (distance < threshold).float() * 10.0
    return reward


def reach_target(
    env: ManagerBasedRLEnv, command_name: str, threshold: float = 0.5
) -> torch.Tensor:
    """Check if robot reached target."""
    target_pos = env.command_manager.get_command(command_name)
    robot_pos = env.scene["robot"].data.root_pos_w[:, :2]
    distance = torch.norm(target_pos - robot_pos, dim=-1)
    return (distance < threshold).float()


"""
Other rewards.
"""


def _closest_obstacle_geometry(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    from unitree_rl_lab.tasks.locomotion import mdp

    return mdp.closest_mid360_obstacle(
        env,
        sensor_name="mid360_lidar",
        max_distance=float(getattr(env.cfg, "lidar_max_distance", 6.0)),
        horizontal_fov_deg=float(getattr(env.cfg, "compression_fov_deg", 180.0)),
        roi_x_min=float(getattr(env.cfg, "roi_x_min", -0.5)),
        roi_x_max=float(getattr(env.cfg, "roi_x_max", 6.0)),
        roi_abs_y_max=float(getattr(env.cfg, "roi_abs_y_max", 3.0)),
        roi_z_min=float(getattr(env.cfg, "roi_z_min", -1.0)),
        roi_z_max=float(getattr(env.cfg, "roi_z_max", 0.8)),
        min_planar_distance=float(getattr(env.cfg, "min_planar_distance", 0.2)),
        robot_body_radius=float(getattr(env.cfg, "robot_body_radius", 0.4)),
    )


def proxemic_comfort(
    env: ManagerBasedRLEnv,
    comfort_margin: float = 1.2,
) -> torch.Tensor:
    """Penalty for getting too close to obstacles outside the hard safety set."""
    distances, _ = _closest_obstacle_geometry(env)
    return torch.square(torch.relu(comfort_margin - distances))


def safe_approach_velocity(
    env: ManagerBasedRLEnv,
    comfort_margin: float = 1.4,
) -> torch.Tensor:
    """Penalty for approaching the nearest obstacle too quickly."""
    distances, directions = _closest_obstacle_geometry(env)
    robot_vel_xy = env.scene["robot"].data.root_lin_vel_w[:, :2]
    closing_speed = torch.relu(torch.sum(robot_vel_xy * directions, dim=-1))
    gate = (distances < comfort_margin).float()
    return gate * torch.square(closing_speed)


def safe_approach_acceleration(
    env: ManagerBasedRLEnv,
    comfort_margin: float = 1.4,
) -> torch.Tensor:
    """Penalty for increasing closing speed near the nearest obstacle."""
    distances, directions = _closest_obstacle_geometry(env)
    robot_vel_xy = env.scene["robot"].data.root_lin_vel_w[:, :2]
    closing_speed = torch.relu(torch.sum(robot_vel_xy * directions, dim=-1))
    prev = getattr(env, "_prev_closing_speed", None)
    if prev is None or prev.shape != closing_speed.shape:
        prev = torch.zeros_like(closing_speed)
    dt = float(getattr(env, "step_dt", getattr(env, "sim", None).cfg.dt if hasattr(getattr(env, "sim", None), "cfg") else 0.02))
    dt = max(dt, 1.0e-3)
    closing_acc = torch.relu((closing_speed - prev) / dt)
    env._prev_closing_speed = closing_speed.detach()
    gate = (distances < comfort_margin).float()
    return gate * torch.square(closing_acc)


def tangential_avoidance(
    env: ManagerBasedRLEnv,
    comfort_margin: float = 1.6,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Reward tangential motion around nearby obstacles instead of straight-line closing."""
    distances, directions = _closest_obstacle_geometry(env)
    robot_vel_xy = env.scene["robot"].data.root_lin_vel_w[:, :2]
    tangential = robot_vel_xy - torch.sum(robot_vel_xy * directions, dim=-1, keepdim=True) * directions
    tangential_speed = torch.linalg.norm(tangential, dim=-1)
    cmd_speed = torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2], dim=-1)
    gate = ((distances < comfort_margin) & (cmd_speed > 0.1)).float()
    return gate * torch.tanh(2.0 * tangential_speed)


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        reward += torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    return reward
