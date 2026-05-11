# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL using P3O (Safe RL)."""

"""Launch Isaac Sim Simulator first."""


import gymnasium as gym
import pathlib
import sys

sys.path.insert(0, f"{pathlib.Path(__file__).parent.parent}")
from list_envs import import_packages  # noqa: F401

sys.path.pop(0)

tasks = []
for task_spec in gym.registry.values():
    if "Unitree" in task_spec.id and "Isaac" not in task_spec.id:
        tasks.append(task_spec.id)

import argparse

import argcomplete

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL P3O.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, choices=tasks, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
# P3O specific arguments
parser.add_argument("--cost_limit", type=float, default=25.0, help="Cost limit for P3O.")
parser.add_argument("--kappa", type=float, default=1.0, help="Penalty coefficient for P3O.")
parser.add_argument("--cost_gamma", type=float, default=0.99, help="Cost discount factor for P3O.")
parser.add_argument("--cost_lam", type=float, default=0.95, help="Cost GAE lambda for P3O.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
argcomplete.autocomplete(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# for distributed training, check minimum supported rsl-rl version
RSL_RL_VERSION = "2.3.1"
installed_version = metadata.version("rsl-rl-lib")
if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import inspect
import os
import shutil
import time
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner
from rsl_rl.algorithms import P3O
from rsl_rl.modules import ActorCritic

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.export_deploy_cfg import export_deploy_cfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


class SafeRLEnvWrapper(gym.Wrapper):
    """Wrapper to add cost computation for Safe RL."""

    def __init__(self, env, cost_limit=25.0):
        super().__init__(env)
        self.cost_limit = cost_limit

    def step(self, action):
        """Step environment and compute costs."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Compute costs from environment state
        # Cost 1: Base tilt (orientation)
        if hasattr(self.env, 'scene') and 'robot' in self.env.scene:
            robot = self.env.scene['robot']
            projected_gravity = robot.data.projected_gravity
            base_tilt_cost = torch.relu(torch.abs(projected_gravity[:, 0]) - 0.3)  # Roll
            base_tilt_cost += torch.relu(torch.abs(projected_gravity[:, 1]) - 0.3)  # Pitch

            # Cost 2: Low base height
            base_height = robot.data.root_pos_w[:, 2]
            height_cost = torch.relu(0.6 - base_height)

            # Cost 3: Large joint torques
            joint_torques = torch.abs(robot.data.applied_torque)
            torque_cost = torch.relu(joint_torques - 60.0).sum(dim=-1) * 0.01

            # Combine costs
            costs = base_tilt_cost * 10.0 + height_cost * 5.0 + torque_cost
        else:
            # Fallback: zero costs
            costs = torch.zeros(self.env.num_envs, device=obs.device)

        # Add cost to info
        if 'episode' not in info:
            info['episode'] = {}
        info['episode']['cost'] = costs.mean().item()

        return obs, reward, terminated, truncated, info

    def compute_costs(self):
        """Compute current costs for all environments."""
        if hasattr(self.env, 'scene') and 'robot' in self.env.scene:
            robot = self.env.scene['robot']
            projected_gravity = robot.data.projected_gravity
            base_tilt_cost = torch.relu(torch.abs(projected_gravity[:, 0]) - 0.3)
            base_tilt_cost += torch.relu(torch.abs(projected_gravity[:, 1]) - 0.3)

            base_height = robot.data.root_pos_w[:, 2]
            height_cost = torch.relu(0.6 - base_height)

            joint_torques = torch.abs(robot.data.applied_torque)
            torque_cost = torch.relu(joint_torques - 60.0).sum(dim=-1) * 0.01

            costs = base_tilt_cost * 10.0 + height_cost * 5.0 + torque_cost
            return costs
        return torch.zeros(self.env.num_envs, device=self.env.device)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with P3O agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl_p3o", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for safe RL (cost computation)
    env = SafeRLEnvWrapper(env, cost_limit=args_cli.cost_limit)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Create policy (ActorCritic)
    policy = ActorCritic(
        obs_shape=env.num_obs,
        privileged_obs_shape=env.num_privileged_obs if hasattr(env, 'num_privileged_obs') else env.num_obs,
        actions_shape=env.num_actions,
        initial_std=agent_cfg.policy.init_noise_std if hasattr(agent_cfg.policy, 'init_noise_std') else 1.0,
    ).to(agent_cfg.device)

    # Create cost critic (same architecture as critic)
    cost_critic = ActorCritic(
        obs_shape=env.num_obs,
        privileged_obs_shape=env.num_privileged_obs if hasattr(env, 'num_privileged_obs') else env.num_obs,
        actions_shape=env.num_actions,
        initial_std=1.0,
    ).to(agent_cfg.device)

    # Create P3O algorithm
    alg_cfg = agent_cfg.algorithm
    alg = P3O(
        policy=policy,
        cost_critic=cost_critic,
        num_learning_epochs=alg_cfg.num_learning_epochs,
        num_mini_batches=alg_cfg.num_mini_batches,
        clip_param=alg_cfg.clip_param,
        gamma=alg_cfg.gamma,
        lam=alg_cfg.lam,
        value_loss_coef=alg_cfg.value_loss_coef,
        entropy_coef=alg_cfg.entropy_coef,
        learning_rate=alg_cfg.learning_rate,
        max_grad_norm=alg_cfg.max_grad_norm,
        use_clipped_value_loss=alg_cfg.use_clipped_value_loss,
        schedule=alg_cfg.schedule,
        desired_kl=alg_cfg.desired_kl,
        device=agent_cfg.device,
        normalize_advantage_per_mini_batch=getattr(alg_cfg, 'normalize_advantage_per_mini_batch', False),
        cost_gamma=args_cli.cost_gamma,
        cost_lam=args_cli.cost_lam,
        cost_limit=args_cli.cost_limit,
        kappa=args_cli.kappa,
    )

    # Create custom runner for P3O
    from rsl_rl.runners.on_policy_runner import OnPolicyRunner

    class P3ORunner(OnPolicyRunner):
        """Modified runner for P3O with cost handling."""

        def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
            """Learn with cost handling."""
            # initialize
            if init_at_random_ep_len:
                self.env.episode_length_buf = torch.randint_like(
                    self.env.episode_length_buf, high=int(self.env.max_episode_length)
                )
            obs, privileged_obs = self.env.get_observations()
            obs, privileged_obs = obs.to(self.device), privileged_obs.to(self.device)
            self.alg.storage.train_type = "safe_rl"  # Enable safe RL mode
            self.alg.init_storage(
                self.env.num_envs,
                self.env.max_episode_length,
                [self.env.num_obs],
                [self.env.num_privileged_obs] if hasattr(self.env, 'num_privileged_obs') else [self.env.num_obs],
                [self.env.num_actions],
            )

            obs = self.alg.storage.observations[0]
            privileged_obs = self.alg.storage.privileged_observations[0] if self.alg.storage.privileged_observations is not None else obs

            # training loop
            self.current_learning_iteration = 0
            for it in range(self.current_learning_iteration, num_learning_iterations):
                start = time.time()

                # Rollout
                for step in range(self.env.max_episode_length):
                    with torch.no_grad():
                        actions = self.alg.act(obs, privileged_obs)
                    obs, rewards, dones, infos = self.env.step(actions)

                    # Get costs from wrapper
                    if hasattr(self.env, 'compute_costs'):
                        costs = self.env.compute_costs()
                    else:
                        costs = torch.zeros_like(rewards)

                    obs, rewards, costs, dones = (
                        obs.to(self.device),
                        rewards.to(self.device),
                        costs.to(self.device),
                        dones.to(self.device),
                    )
                    self.alg.process_env_step(rewards, costs, dones, infos)

                    if self.log_dir is not None:
                        self.log(infos)

                # Update
                stop = time.time()
                self.collection_time = stop - start

                # Compute returns
                start = stop
                self.alg.compute_returns(privileged_obs)

                # Update policy
                loss_dict = self.alg.update()
                stop = time.time()
                self.learn_time = stop - start

                if self.log_dir is not None:
                    self.log_losses(loss_dict)

                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

                self.current_learning_iteration += 1

            return self.current_learning_iteration

    # Create runner
    runner = P3ORunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.alg = alg  # Replace algorithm with P3O

    # write git state to logs
    runner.add_git_repo_to_log(__file__)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    export_deploy_cfg(env.unwrapped, log_dir)
    shutil.copy(
        inspect.getfile(env_cfg.__class__),
        os.path.join(log_dir, "params", os.path.basename(inspect.getfile(env_cfg.__class__))),
    )

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
