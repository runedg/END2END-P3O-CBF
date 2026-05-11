# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""P3O (Penalized Proximal Policy Optimization) for Safe RL.

Based on ECO-humanoid implementation.
Reference:
    - Zhang et al. "Penalized Proximal Policy Optimization for Safe Reinforcement Learning."
      arXiv preprint arXiv:2205.11814 (2022).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from rsl_rl.modules.actor_critic_safe import ActorCriticSafe
from rsl_rl.storage.rollout_storage_safe import RolloutStorageSafe


class P3O_ECO:
    """P3O: Penalized Proximal Policy Optimization for Safe RL."""

    def __init__(
        self,
        actor_critic: ActorCriticSafe,
        num_learning_epochs: int = 1,
        num_mini_batches: int = 1,
        clip_param: float = 0.2,
        gamma: float = 0.998,
        lam: float = 0.95,
        cost_gamma: float = 0.998,
        cost_lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.0,
        learning_rate: float = 1e-3,
        cost_critic_learning_rate: float = 1e-5,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "fixed",
        desired_kl: float = 0.01,
        device: str = "cpu",
        cost_limit: float = 25.0,
        kappa: float = 1.0,
    ):
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.cost_critic_learning_rate = cost_critic_learning_rate

        # Actor-Critic components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)

        # Optimizers
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.cost_value_optimizer = optim.Adam(
            self.actor_critic.cost_critic.parameters(), lr=cost_critic_learning_rate
        )

        # Storage
        self.storage = None
        self.transition = RolloutStorageSafe.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # P3O parameters
        self.cost_gamma = cost_gamma
        self.cost_lam = cost_lam
        self.cost_limit = cost_limit
        self.kappa = kappa

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        """Initialize storage."""
        self.storage = RolloutStorageSafe(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            action_shape,
            self.device,
        )

    def test_mode(self):
        self.actor_critic.eval()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        """Compute actions and values."""
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        rew_value = self.actor_critic.evaluate(critic_obs)
        cost_value = self.actor_critic.cost_evaluate(critic_obs)

        self.transition.values = rew_value.detach()
        self.transition.cost_values = cost_value.detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.transition.actions
        ).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()

        # Record observations before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs

        return self.transition.actions

    def process_env_step(self, rewards, costs, dones, infos):
        """Record environment step."""
        self.transition.rewards = rewards.clone()
        self.transition.costs = costs.clone()
        self.transition.dones = dones

        # Bootstrapping on time outs
        if isinstance(infos, dict) and "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )
            self.transition.costs += self.cost_gamma * torch.squeeze(
                self.transition.cost_values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        """Compute returns and advantages."""
        last_values = self.actor_critic.evaluate(last_critic_obs)
        last_cost_values = self.actor_critic.cost_evaluate(last_critic_obs)

        self.storage.compute_returns(last_values.detach(), self.gamma, self.lam)
        self.storage.compute_cost_returns(last_cost_values.detach(), self.cost_gamma, self.cost_lam)

    def update(self):
        """Update policy."""
        mean_value_loss = 0
        mean_cost_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_penalty_loss = 0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            cost_target_values_batch,
            advantages_batch,
            cost_advantages_batch,
            returns_batch,
            cost_returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:

            # Recompute actions log prob and entropy
            self.actor_critic.act(obs_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL divergence for adaptive learning rate
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        self.cost_critic_learning_rate = max(1e-5, self.cost_critic_learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        self.cost_critic_learning_rate = min(1e-2, self.cost_critic_learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate
                    for param_group in self.cost_value_optimizer.param_groups:
                        param_group["lr"] = self.cost_critic_learning_rate

            # Surrogate loss (PPO)
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # P3O: Compute penalty loss
            with torch.no_grad():
                # Jc = E[cost] - cost_limit
                episode_costs = self.storage.costs.sum(dim=0).squeeze(-1)
                avg_episode_cost = episode_costs.mean()
                Jc = avg_episode_cost - self.cost_limit

            # surr_cadv = E[ratio * cost_advantage]
            surr_cadv = (ratio * torch.squeeze(cost_advantages_batch)).mean()
            penalty_loss = self.kappa * torch.relu(surr_cadv + Jc)

            # Value function loss (reward)
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # Total loss (actor + reward critic)
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean() + penalty_loss

            # Update actor and reward critic
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Update cost critic (separate)
            cost_value_batch = self.actor_critic.cost_evaluate(critic_obs_batch)
            if self.use_clipped_value_loss:
                cost_value_clipped = cost_target_values_batch + (cost_value_batch - cost_target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                cost_value_losses = (cost_value_batch - cost_returns_batch).pow(2)
                cost_value_losses_clipped = (cost_value_clipped - cost_returns_batch).pow(2)
                cost_value_loss = torch.max(cost_value_losses, cost_value_losses_clipped).mean()
            else:
                cost_value_loss = (cost_returns_batch - cost_value_batch).pow(2).mean()

            self.cost_value_optimizer.zero_grad()
            cost_value_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.cost_critic.parameters(), self.max_grad_norm)
            self.cost_value_optimizer.step()

            # Accumulate losses
            mean_value_loss += value_loss.item()
            mean_cost_value_loss += cost_value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_penalty_loss += penalty_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_cost_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_penalty_loss /= num_updates

        self.storage.clear()

        return {
            "value_function": mean_value_loss,
            "cost_value": mean_cost_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "penalty": mean_penalty_loss,
        }
