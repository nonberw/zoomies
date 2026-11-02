"""Load a trained policy and drive a ZoomieEnv with it."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .env import ACT_DIM, OBS_DIM, QV
from .ppo import ActorCritic, PPOConfig


class Policy:
    def __init__(self, path: str | Path, device: str = "cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        self.net = ActorCritic(OBS_DIM, ACT_DIM, PPOConfig()).to(device)
        self.net.load_state_dict(ckpt["model"])
        self.net.eval()
        self.device = device
        self.iter = ckpt.get("iter")

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.net.act(torch.as_tensor(obs, device=self.device), deterministic=True).cpu().numpy()


def kick(env, idx: np.ndarray, velocity_xy: np.ndarray):
    """Shove the torso of envs `idx` by an instant change of its world-frame velocity."""
    env.state[idx, QV:QV + 2] += velocity_xy
