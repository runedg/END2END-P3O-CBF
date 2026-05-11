# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Obstacle avoidance environment for G1 robot using CBF-based cost."""

import torch
import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp

# Flat terrain for obstacle avoidance
FLAT_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(20.0, 20.0),  # Larger terrain for navigation
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


@configclass
class ObstacleSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with obstacles."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=FLAT_TERRAIN_CFG,
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

    # sensors
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    # Lidar for obstacle detection
    lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.0, 1.0]),
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

    # Obstacles (will be spawned randomly)
    obstacles = AssetBaseCfg(
        prim_path="/World/obstacles",
        spawn=None,  # Will be spawned dynamically
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
            "pose_range": {"x": (-5.0, 5.0), "y": (-5.0, 5.0), "yaw": (-3.14, 3.14)},
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
    """Command specifications for navigation to target."""

    target_position = mdp.UniformTargetPositionCommandCfg(
        asset_name="robot",
        resampling_time_range=(20.0, 20.0),
        debug_vis=True,
        ranges=mdp.UniformTargetPositionCommandCfg.Ranges(
            x=(-8.0, 8.0), y=(-8.0, 8.0)
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
    """Observation specifications with obstacle detection."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))

        # Target direction (for navigation)
        target_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "target_position"})

        # Obstacle distances (from ray caster)
        obstacle_distances = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("lidar")},
            clip=(-1.0, 5.0),
        )

        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
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
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        target_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "target_position"})
        obstacle_distances = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("lidar")},
            clip=(-1.0, 5.0),
        )
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 5

    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Reward terms for obstacle avoidance."""

    # -- Task: reach target
    reach_target = RewTerm(
        func=mdp.target_position_distance,
        weight=2.0,
        params={"command_name": "target_position", "threshold": 0.5},
    )

    alive = RewTerm(func=mdp.is_alive, weight=0.1)

    # -- Penalties
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)


@configclass
class TerminationsCfg:
    """Termination terms."""

    time_out = EventTerm(func=mdp.time_out, time_out=True)
    base_height = EventTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    bad_orientation = EventTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})
    reach_target = EventTerm(
        func=mdp.reach_target,
        params={"command_name": "target_position", "threshold": 0.5},
    )


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

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 30.0  # Longer episodes for navigation
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.lidar.update_period = self.decimation * self.sim.dt


@configclass
class RobotPlayEnvCfg(ObstacleAvoidanceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
