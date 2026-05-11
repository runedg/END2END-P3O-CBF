"""Play G1 walking with P3O model, raycast visualization and video recording."""

import argparse
import torch
import numpy as np
import os
from datetime import datetime

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play G1 with raycast and video recording.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--device", type=str, default="cuda:0", help="Device to use.")
parser.add_argument("--video_length", type=int, default=3000, help="Video length in steps.")
parser.add_argument("--headless", action="store_true", default=False, help="Run headless.")

args_cli = parser.parse_args()

# launch omniverse app (args_cli has headless from parser, don't duplicate)
if args_cli.headless:
    args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import gymnasium as gym
import isaaclab_tasks  # noqa: F401

from rsl_rl.algorithms.p3o_eco import P3O_ECO
from rsl_rl.modules import ActorCriticSafe


def main():
    """Main play function with raycast and video recording."""

    import unitree_rl_lab.tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from isaaclab.envs import ManagerBasedRLEnv

    # Load environment configuration
    env_cfg = load_cfg_from_registry("Unitree-G1-29dof-Velocity", "play_env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    # Ensure height_scanner debug_vis is enabled
    if hasattr(env_cfg.scene, 'height_scanner'):
        env_cfg.scene.height_scanner.debug_vis = True
        print("[INFO] Enabled height_scanner debug visualization")

    # Create video directory
    video_dir = os.path.join(os.path.dirname(args_cli.checkpoint), "videos")
    os.makedirs(video_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create environment with video recording
    print("[INFO] Creating environment with video recording...")
    env = gym.make("Unitree-G1-29dof-Velocity", cfg=env_cfg, render_mode="rgb_array")

    # Wrap for video recording
    video_kwargs = {
        "video_folder": video_dir,
        "step_trigger": lambda step: step == 0,
        "video_length": args_cli.video_length,
        "name_prefix": f"p3o_raycast_{timestamp}",
        "disable_logger": True,
    }
    print(f"[INFO] Recording video to: {video_dir}")
    print(f"[INFO] Recording {args_cli.video_length} steps (~{args_cli.video_length // 30} seconds)")
    env = gym.wrappers.RecordVideo(env, **video_kwargs)

    device = args_cli.device
    num_envs = env.unwrapped.num_envs

    print(f"[INFO] Playing with {num_envs} environments on {device}")
    print(f"[INFO] Loading checkpoint: {args_cli.checkpoint}")

    # Get observation shapes
    num_obs = env.unwrapped.observation_space['policy'].shape[-1]
    num_privileged_obs = env.unwrapped.observation_space['critic'].shape[-1]
    num_actions = env.unwrapped.action_space.shape[-1]

    # Create policy networks
    policy = ActorCriticSafe(
        num_actor_obs=num_obs,
        num_critic_obs=num_privileged_obs,
        num_actions=num_actions,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(args_cli.checkpoint, map_location=device)
    policy.load_state_dict(checkpoint['policy_state_dict'])
    print(f"[INFO] Loaded checkpoint from iteration {checkpoint.get('iteration', 'unknown')}")

    # Set to eval mode
    policy.eval()

    # Reset environment
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs_dict = reset_result[0]
    else:
        obs_dict = reset_result
    obs = obs_dict['policy'].to(device)

    # Set camera to focus on first robot
    robot_pos = env.unwrapped.scene["robot"].data.root_pos_w[0].cpu().numpy()
    print(f"[INFO] Robot position: {robot_pos}")

    # Configure camera view
    env.unwrapped.sim.set_camera_view(
        eye=[robot_pos[0] + 2.0, robot_pos[1] - 2.0, robot_pos[2] + 1.5],  # Camera position
        target=[robot_pos[0], robot_pos[1], robot_pos[2] + 0.5]  # Look at robot
    )
    print("[INFO] Camera set to focus on robot 0")

    print("[INFO] Starting visualization and recording...")
    print("[INFO] Look for dense rays shooting down from the robot!")

    # Play loop
    step_count = 0
    try:
        while step_count < args_cli.video_length:
            with torch.no_grad():
                actions = policy.act(obs)
                actions = policy.action_mean

            step_result = env.step(actions)
            next_obs_dict = step_result[0]
            obs = next_obs_dict['policy'].to(device)
            step_count += 1

            if step_count % 100 == 0:
                print(f"\rStep {step_count}/{args_cli.video_length}", end="")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user")

    finally:
        print(f"\n[INFO] Closing environment and saving video...")
        env.close()
        print(f"[INFO] Video saved to: {video_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
