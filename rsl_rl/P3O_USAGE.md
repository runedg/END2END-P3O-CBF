# P3O (Penalized Proximal Policy Optimization) in RSL-RL

P3O is a Safe RL algorithm that extends PPO by adding a cost critic and a penalty term to enforce safety constraints.

## Overview

P3O modifies PPO's policy loss with a penalty term:

```
loss = loss_ppo + kappa * ReLU(surr_cadv + Jc)
```

Where:
- `surr_cadv`: Surrogate cost advantage
- `Jc = E[cost] - cost_limit`: Constraint violation
- `kappa`: Penalty coefficient

## Changes Made to RSL-RL

### 1. `storage/rollout_storage.py`
- Added `costs`, `cost_values`, `cost_returns`, `cost_advantages` storage
- Modified `Transition` class to support cost data
- Updated `Batch` class with cost fields
- Modified `mini_batch_generator` and `recurrent_mini_batch_generator` to return cost data

### 2. `algorithms/p3o.py` (New)
- New `P3O` class inheriting from `PPO`
- Adds `cost_critic` network (similar to value critic)
- Implements cost advantage computation using GAE
- Adds penalty loss to policy update
- Separate optimizer for cost critic

### 3. `algorithms/__init__.py`
- Exports `P3O` class

## Usage in Isaac Lab

### 1. Environment Setup

Your Isaac Lab environment needs to return costs in the `step()` method:

```python
def step(self, actions):
    # ... existing code ...
    
    # Compute rewards
    rewards = self._compute_rewards()
    
    # Compute costs (safety constraints)
    costs = self._compute_costs()
    # Examples of costs:
    # - Joint torque limits exceeded
    # - Base tilt angle too large
    # - Feet slipping
    # - Collision
    
    dones = self._compute_dones()
    
    return obs, rewards, costs, dones, extras
```

### 2. Runner Modification

Modify `OnPolicyRunner` to handle costs:

```python
# In learn() method, modify the rollout loop:
for _ in range(self.cfg["num_steps_per_env"]):
    actions = self.alg.act(obs)
    obs, rewards, costs, dones, extras = self.env.step(actions.to(self.env.device))
    
    # Check for NaN
    if self.cfg.get("check_for_nan", True):
        check_nan(obs, rewards, dones)
    
    # Process step with costs
    obs, rewards, costs, dones = (
        obs.to(self.device), 
        rewards.to(self.device), 
        costs.to(self.device),
        dones.to(self.device)
    )
    self.alg.process_env_step(obs, rewards, costs, dones, extras)
```

### 3. Configuration

Use the P3O config file:

```yaml
algorithm:
  class_name: rsl_rl.algorithms.P3O
  
  # P3O specific
  cost_gamma: 0.99
  cost_lam: 0.95
  cost_limit: 25.0      # Your safety threshold
  kappa: 1.0            # Penalty strength
```

### 4. Training

```bash
python scripts/rsl_rl/train.py --task YourRobotTask --headless --algo_cfg configs/p3o_config.yaml
```

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `cost_limit` | Maximum allowed cost per episode | 25.0 |
| `kappa` | Penalty coefficient (higher = more conservative) | 1.0 |
| `cost_gamma` | Discount factor for cost | 0.99 |
| `cost_lam` | GAE lambda for cost | 0.95 |
| `standardized_cost_adv` | Whether to standardize cost advantages | false |

## Cost Design Tips

1. **Start Conservative**: Set `cost_limit` lower initially, increase as needed
2. **Smooth Costs**: Use smooth cost functions (e.g., ReLU) rather than binary
3. **Multiple Constraints**: Sum multiple cost terms if needed
4. **Normalization**: Consider normalizing costs to similar scale as rewards

Example cost computation:

```python
def _compute_costs(self):
    costs = torch.zeros(self.num_envs, device=self.device)
    
    # Joint torque limit
    torque_cost = torch.sum(torch.relu(torch.abs(self.torques) - self.torque_limit), dim=1)
    
    # Base tilt angle
    tilt_cost = torch.relu(torch.abs(self.base_tilt) - self.tilt_limit)
    
    costs = torque_cost + tilt_cost * 10.0  # Weight tilt more
    return costs
```

## Debugging

Monitor these metrics in tensorboard/wandb:
- `value/cost`: Average predicted cost
- `loss/cost_value`: Cost critic loss
- `loss/penalty`: P3O penalty loss
- `Metrics/EpCost`: Episode cost (add to logger)

If cost is too high:
- Increase `kappa`
- Decrease `cost_limit`
- Make cost critic larger/more expressive

If learning is too conservative:
- Decrease `kappa`
- Increase `cost_limit`
- Check if cost estimates are accurate
