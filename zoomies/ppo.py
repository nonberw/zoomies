"""Proximal Policy Optimization, from scratch in PyTorch.

Gaussian actor + value critic, GAE(lambda), clipped surrogate and clipped value loss,
running observation normalization, and a learning rate that adapts to keep the KL near a target.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class PPOConfig:
    horizon: int = 24            # env steps per env per update
    epochs: int = 5
    minibatches: int = 4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    value_coef: float = 1.0
    entropy_coef: float = 0.005
    lr: float = 1e-3
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    init_std: float = 1.0
    hidden: tuple[int, ...] = (512, 256, 128)


def mlp(sizes, out_scale=1.0):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ELU())
    with torch.no_grad():
        layers[-1].weight.mul_(out_scale)
        layers[-1].bias.zero_()
    return nn.Sequential(*layers)


class RunningNorm(nn.Module):
    """Welford-style running mean/variance, merged batch by batch."""

    def __init__(self, dim, clip=5.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))
        self.clip = clip

    @torch.no_grad()
    def update(self, x):
        b_mean, b_var, b_n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = b_mean - self.mean
        tot = self.count + b_n
        self.mean += delta * b_n / tot
        self.var = (self.var * self.count + b_var * b_n + delta**2 * self.count * b_n / tot) / tot
        self.count = tot

    def forward(self, x):
        return ((x - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-self.clip, self.clip)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, cfg: PPOConfig):
        super().__init__()
        self.norm = RunningNorm(obs_dim)
        self.actor = mlp([obs_dim, *cfg.hidden, act_dim], out_scale=0.01)
        self.critic = mlp([obs_dim, *cfg.hidden, 1])
        self.log_std = nn.Parameter(torch.full((act_dim,), float(np.log(cfg.init_std))))

    def dist(self, obs_n):
        return torch.distributions.Normal(self.actor(obs_n), self.log_std.exp().expand(obs_n.shape[0], -1))

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        obs_n = self.norm(obs)
        if deterministic:
            return self.actor(obs_n)
        d = self.dist(obs_n)
        a = d.sample()
        return a, d.log_prob(a).sum(-1), self.critic(obs_n).squeeze(-1)


class PPO:
    def __init__(self, obs_dim, act_dim, num_envs, cfg: PPOConfig | None = None, device="cuda"):
        self.cfg = cfg = cfg or PPOConfig()
        self.device = device
        self.ac = ActorCritic(obs_dim, act_dim, cfg).to(device)
        self.opt = torch.optim.Adam(self.ac.parameters(), lr=cfg.lr)
        self.lr = cfg.lr
        T, N = cfg.horizon, num_envs
        z = lambda *s: torch.zeros(*s, device=device)  # noqa: E731
        self.obs, self.act_buf = z(T, N, obs_dim), z(T, N, act_dim)
        self.logp, self.val, self.rew, self.done = z(T, N), z(T, N), z(T, N), z(T, N)
        self.t = 0

    def store(self, obs, act, logp, val, rew, done):
        t = self.t
        self.obs[t], self.act_buf[t], self.logp[t], self.val[t] = obs, act, logp, val
        self.rew[t], self.done[t] = rew, done
        self.t += 1

    @torch.no_grad()
    def bootstrap_timeouts(self, rew, timeout, next_obs_before_reset):
        """A time limit isn't a real ending: add gamma * V(s') so the critic doesn't learn 'the world ends at 20 s'."""
        if timeout.any():
            v = self.ac.critic(self.ac.norm(next_obs_before_reset)).squeeze(-1)
            rew = rew + self.cfg.gamma * v * timeout
        return rew

    @torch.no_grad()
    def _advantages(self, last_obs):
        c = self.cfg
        last_val = self.ac.critic(self.ac.norm(last_obs)).squeeze(-1)
        adv = torch.zeros_like(self.rew)
        gae = torch.zeros_like(last_val)
        for t in reversed(range(c.horizon)):
            next_val = last_val if t == c.horizon - 1 else self.val[t + 1]
            not_done = 1.0 - self.done[t]
            delta = self.rew[t] + c.gamma * next_val * not_done - self.val[t]
            gae = delta + c.gamma * c.lam * not_done * gae
            adv[t] = gae
        return adv, adv + self.val

    def update(self, last_obs):
        c = self.cfg
        self.ac.norm.update(self.obs.flatten(0, 1))
        adv, ret = self._advantages(last_obs)
        flat = lambda x: x.flatten(0, 1)  # noqa: E731
        obs, act, logp_old, val_old = flat(self.obs), flat(self.act_buf), flat(self.logp), flat(self.val)
        adv, ret = flat(adv), flat(ret)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = obs.shape[0]
        mb = n // c.minibatches
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "kl": 0.0}
        updates = 0
        for _ in range(c.epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(c.minibatches):
                idx = perm[i * mb:(i + 1) * mb]
                obs_n = self.ac.norm(obs[idx])
                d = self.ac.dist(obs_n)
                logp = d.log_prob(act[idx]).sum(-1)
                ratio = (logp - logp_old[idx]).exp()
                with torch.no_grad():
                    kl = ((ratio - 1) - (logp - logp_old[idx])).mean()  # low-variance KL estimator
                # Adaptive LR: shrink it when one step moves the policy too far, grow it when too timid.
                if kl > 2 * c.desired_kl:
                    self.lr = max(1e-5, self.lr / 1.5)
                elif kl < c.desired_kl / 2:
                    self.lr = min(1e-2, self.lr * 1.5)
                for g in self.opt.param_groups:
                    g["lr"] = self.lr

                surr = torch.min(ratio * adv[idx], ratio.clamp(1 - c.clip, 1 + c.clip) * adv[idx])
                v = self.ac.critic(obs_n).squeeze(-1)
                v_clip = val_old[idx] + (v - val_old[idx]).clamp(-c.clip, c.clip)
                v_loss = torch.max((v - ret[idx]) ** 2, (v_clip - ret[idx]) ** 2).mean()
                ent = d.entropy().sum(-1).mean()
                loss = -surr.mean() + c.value_coef * v_loss - c.entropy_coef * ent

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), c.max_grad_norm)
                self.opt.step()
                stats["policy_loss"] += -surr.mean().item()
                stats["value_loss"] += v_loss.item()
                stats["entropy"] += ent.item()
                stats["kl"] += kl.item()
                updates += 1
        self.t = 0
        return {k: v / updates for k, v in stats.items()} | {"lr": self.lr, "std": self.ac.log_std.exp().mean().item()}
