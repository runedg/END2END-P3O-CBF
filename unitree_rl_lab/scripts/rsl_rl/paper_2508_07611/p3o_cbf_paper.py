from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCriticSafe

from rollout_storage_paper import RolloutStoragePaper


class P3OCBFPaper:
    """Paper-aligned single-constraint P3O variant for the CBF experiment."""

    def __init__(
        self,
        actor_critic: ActorCriticSafe,
        num_learning_epochs: int = 1,
        num_mini_batches: int = 1,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        cost_gamma: float = 0.99,
        cost_lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.0,
        learning_rate: float = 1e-3,
        cost_critic_learning_rate: float = 1e-4,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        device: str = "cpu",
        cost_limit: float = 0.3,
        kappa: float = 1.0,
    ):
        self.device = device
        self.actor_critic = actor_critic.to(device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.cost_value_optimizer = optim.Adam(self.actor_critic.cost_critic.parameters(), lr=cost_critic_learning_rate)
        self.storage = None
        self.transition = RolloutStoragePaper.Transition()
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.clip_param = clip_param
        self.gamma = gamma
        self.lam = lam
        self.cost_gamma = cost_gamma
        self.cost_lam = cost_lam
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.learning_rate = learning_rate
        self.cost_critic_learning_rate = cost_critic_learning_rate
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.schedule = schedule
        self.desired_kl = desired_kl
        self.cost_limit = cost_limit
        self.kappa = kappa

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStoragePaper(
            num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device
        )

    def act(self, obs, critic_obs):
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.cost_values = self.actor_critic.cost_evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, costs, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.costs = costs.clone()
        self.transition.dones = dones

        if isinstance(infos, dict) and "time_outs" in infos:
            time_outs = infos["time_outs"].unsqueeze(1).to(self.device)
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * time_outs, 1)
            self.transition.costs += self.cost_gamma * torch.squeeze(self.transition.cost_values * time_outs, 1)

        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs)
        last_cost_values = self.actor_critic.cost_evaluate(last_critic_obs)
        self.storage.compute_returns(last_values.detach(), self.gamma, self.lam)
        self.storage.compute_cost_returns(last_cost_values.detach(), self.cost_gamma, self.cost_lam)

    def update(self):
        mean_value_loss = 0.0
        mean_cost_value_loss = 0.0
        mean_penalty_loss = 0.0
        mean_constraint_violation = 0.0
        mean_rollout_cost = 0.0

        episode_costs = self.storage.costs.sum(dim=0).squeeze(-1)
        avg_episode_cost = episode_costs.mean()
        raw_cost_adv_mean = self.storage.raw_cost_adv_mean
        raw_cost_adv_std = self.storage.raw_cost_adv_std
        normalized_bias = ((1.0 - self.cost_gamma) * (avg_episode_cost - self.cost_limit) + raw_cost_adv_mean) / raw_cost_adv_std

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
        ) in generator:
            self.actor_critic.act(obs_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

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
                    elif 0.0 < kl_mean < self.desired_kl / 2.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        self.cost_critic_learning_rate = min(1e-3, self.cost_critic_learning_rate * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate
                    for param_group in self.cost_value_optimizer.param_groups:
                        param_group["lr"] = self.cost_critic_learning_rate

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))

            reward_surr = -torch.squeeze(advantages_batch) * ratio
            reward_surr_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(reward_surr, reward_surr_clipped).mean()

            cost_surr = torch.squeeze(cost_advantages_batch) * ratio
            cost_surr_clipped = torch.squeeze(cost_advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            clipped_cost_surrogate = torch.max(cost_surr, cost_surr_clipped).mean()
            constraint_violation = clipped_cost_surrogate + normalized_bias
            penalty_loss = self.kappa * torch.relu(constraint_violation)

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean() + penalty_loss
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

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

            mean_value_loss += value_loss.item()
            mean_cost_value_loss += cost_value_loss.item()
            mean_penalty_loss += penalty_loss.item()
            mean_constraint_violation += float(constraint_violation.item())
            mean_rollout_cost += float(avg_episode_cost.item())

        num_updates = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()
        return {
            "value_function": mean_value_loss / num_updates,
            "cost_value": mean_cost_value_loss / num_updates,
            "penalty": mean_penalty_loss / num_updates,
            "constraint_violation": mean_constraint_violation / num_updates,
            "rollout_episode_cost": mean_rollout_cost / num_updates,
            "raw_cost_adv_mean": raw_cost_adv_mean,
            "raw_cost_adv_std": raw_cost_adv_std,
        }
