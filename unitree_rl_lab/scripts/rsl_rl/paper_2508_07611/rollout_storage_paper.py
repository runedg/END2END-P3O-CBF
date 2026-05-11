from __future__ import annotations

import torch


class RolloutStoragePaper:
    """Rollout storage with normalized and raw cost advantages."""

    class Transition:
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.costs = None
            self.dones = None
            self.values = None
            self.cost_values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        privileged_obs_shape,
        actions_shape,
        device="cpu",
    ):
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=device)
        if privileged_obs_shape[0] is not None:
            self.privileged_observations = torch.zeros(
                num_transitions_per_env, num_envs, *privileged_obs_shape, device=device
            )
        else:
            self.privileged_observations = None
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.costs = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=device).byte()
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.cost_values = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.cost_returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.cost_advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.raw_cost_advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.raw_cost_adv_mean = 0.0
        self.raw_cost_adv_std = 1.0
        self.step = 0

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")

        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None:
            self.privileged_observations[self.step].copy_(transition.critic_observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.costs[self.step].copy_(transition.costs.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.cost_values[self.step].copy_(transition.cost_values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            next_values = last_values if step == self.num_transitions_per_env - 1 else self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def compute_cost_returns(self, last_cost_values, cost_gamma, cost_lam):
        cost_advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            next_cost_values = last_cost_values if step == self.num_transitions_per_env - 1 else self.cost_values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            cost_delta = self.costs[step] + next_is_not_terminal * cost_gamma * next_cost_values - self.cost_values[step]
            cost_advantage = cost_delta + next_is_not_terminal * cost_gamma * cost_lam * cost_advantage
            self.cost_returns[step] = cost_advantage + self.cost_values[step]

        self.raw_cost_advantages = self.cost_returns - self.cost_values
        self.raw_cost_adv_mean = float(self.raw_cost_advantages.mean().item())
        self.raw_cost_adv_std = float(self.raw_cost_advantages.std().item() + 1e-8)
        self.cost_advantages = (self.raw_cost_advantages - self.raw_cost_adv_mean) / self.raw_cost_adv_std

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        critic_observations = (
            self.privileged_observations.flatten(0, 1) if self.privileged_observations is not None else observations
        )

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)
        cost_values = self.cost_values.flatten(0, 1)
        cost_returns = self.cost_returns.flatten(0, 1)
        cost_advantages = self.cost_advantages.flatten(0, 1)

        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]
                yield (
                    observations[batch_idx],
                    critic_observations[batch_idx],
                    actions[batch_idx],
                    values[batch_idx],
                    cost_values[batch_idx],
                    advantages[batch_idx],
                    cost_advantages[batch_idx],
                    returns[batch_idx],
                    cost_returns[batch_idx],
                    old_actions_log_prob[batch_idx],
                    old_mu[batch_idx],
                    old_sigma[batch_idx],
                )
