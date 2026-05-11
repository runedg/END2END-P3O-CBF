# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isolated realG1 obstacle environment with Mid360-style multi-mesh sensing."""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.realg1_mid360 import Mid360PatternCfg

from .obstacle_env_cfg import ActionsCfg, CommandsCfg, EventCfg, ObstacleManager, RewardsCfg, TerminationsCfg
from .obstacle_env_cfg import ObstacleAvoidanceEnvCfg as BaseObstacleAvoidanceEnvCfg
from .obstacle_env_cfg import ObstacleSceneCfg as BaseObstacleSceneCfg


MID360_PATTERN_FILE = (
    "/home/ubuntu/P3O-CBF/OmniPerception/LidarSensor/LidarSensor/sensor_pattern/sensor_lidar/scan_mode/mid360.npy"
)


@configclass
class RealG1SceneCfg(BaseObstacleSceneCfg):
    """Scene configuration isolated for deployment-oriented Mid360 sensing."""

    mid360_lidar = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.10, 0.0, 0.63)),
        ray_alignment="base",
        pattern_cfg=Mid360PatternCfg(
            pattern_file=MID360_PATTERN_FILE,
            samples=1024,
        ),
        debug_vis=False,
        max_distance=6.0,
        mesh_prim_paths=[
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="/World/ground",
                is_shared=True,
                track_mesh_transforms=False,
            ),
        ],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)


@configclass
class RealG1ObservationsCfg:
    """Observations that read from the isolated Mid360 ray-caster."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        obstacle_scan = ObsTerm(
            func=mdp.mid360_realg1_range_features_from_raycaster,
            params={
                "sensor_name": "mid360_lidar",
                "feature_dim": 64,
                "max_distance": 6.0,
                "sensor_offset": (0.10, 0.0, 0.63),
                "horizontal_fov_deg": 180.0,
                "roi_x_min": -0.5,
                "roi_x_max": 6.0,
                "roi_abs_y_max": 3.0,
                "roi_z_min": -1.0,
                "roi_z_max": 0.8,
                "min_planar_distance": 0.2,
                "enable_sensor_noise": True,
                "random_distance_noise": 0.02,
                "pixel_dropout_prob": 0.01,
                "sector_dropout_prob": 0.10,
                "sector_dropout_width_deg": 8.0,
                "random_translation_range": (0.015, 0.015, 0.015),
                "random_rotation_deg_range": (2.0, 2.0, 2.0),
            },
            clip=(0.0, 6.0),
            scale=tuple([1.0 / 6.0] * 64),
        )
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 3
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        obstacle_scan = ObsTerm(
            func=mdp.mid360_realg1_range_features_from_raycaster,
            params={
                "sensor_name": "mid360_lidar",
                "feature_dim": 64,
                "max_distance": 6.0,
                "sensor_offset": (0.10, 0.0, 0.63),
                "horizontal_fov_deg": 180.0,
                "roi_x_min": -0.5,
                "roi_x_max": 6.0,
                "roi_abs_y_max": 3.0,
                "roi_z_min": -1.0,
                "roi_z_max": 0.8,
                "min_planar_distance": 0.2,
                "enable_sensor_noise": True,
                "random_distance_noise": 0.02,
                "pixel_dropout_prob": 0.01,
                "sector_dropout_prob": 0.10,
                "sector_dropout_width_deg": 8.0,
                "random_translation_range": (0.015, 0.015, 0.015),
                "random_rotation_deg_range": (2.0, 2.0, 2.0),
            },
            clip=(0.0, 6.0),
            scale=tuple([1.0 / 6.0] * 64),
        )
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 3

    critic: CriticCfg = CriticCfg()


@configclass
class RealG1ObstacleAvoidanceEnvCfg(BaseObstacleAvoidanceEnvCfg):
    """Deployment-oriented G1 obstacle environment isolated from the legacy path."""

    scene: RealG1SceneCfg = RealG1SceneCfg(num_envs=1024, env_spacing=5.0)
    observations: RealG1ObservationsCfg = RealG1ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.mid360_lidar.update_period = self.decimation * self.sim.dt
        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class RealG1ObstacleAvoidancePlayEnvCfg(RealG1ObstacleAvoidanceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
