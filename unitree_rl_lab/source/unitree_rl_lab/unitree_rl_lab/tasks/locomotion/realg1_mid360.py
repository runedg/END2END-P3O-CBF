from __future__ import annotations

from dataclasses import MISSING
from functools import lru_cache

import numpy as np
import torch
from isaaclab.sensors.ray_caster.patterns.patterns_cfg import PatternBaseCfg
from isaaclab.utils import configclass


@lru_cache(maxsize=8)
def _load_mid360_pattern(pattern_file: str, samples: int) -> tuple[np.ndarray, np.ndarray]:
    pattern = np.load(pattern_file)
    if pattern.ndim != 2 or pattern.shape[1] != 2:
        raise ValueError(f"Expected Mid360 pattern with shape (N, 2), got {pattern.shape}")
    if samples <= 0:
        raise ValueError(f"samples must be positive, got {samples}")
    if samples >= pattern.shape[0]:
        selected = pattern
    else:
        idx = np.linspace(0, pattern.shape[0] - 1, num=samples, dtype=np.int64)
        selected = pattern[idx]
    return selected[:, 0].astype(np.float32), selected[:, 1].astype(np.float32)


def mid360_pattern(cfg: "Mid360PatternCfg", device: str) -> tuple[torch.Tensor, torch.Tensor]:
    theta_np, phi_np = _load_mid360_pattern(cfg.pattern_file, cfg.samples)
    theta = torch.as_tensor(theta_np, device=device)
    phi = torch.as_tensor(phi_np, device=device)

    ray_directions = torch.stack(
        (
            torch.cos(theta) * torch.cos(phi),
            torch.sin(theta) * torch.cos(phi),
            torch.sin(phi),
        ),
        dim=-1,
    )
    ray_starts = torch.zeros_like(ray_directions)
    return ray_starts, ray_directions


@configclass
class Mid360PatternCfg(PatternBaseCfg):
    """Mid360 ray pattern sampled from OmniPerception's scan-mode file."""

    func = mid360_pattern

    pattern_file: str = MISSING
    samples: int = 1024
