import numpy as np
import torch

from zoomies.ppo import PPO, PPOConfig, RunningNorm


def test_running_norm_matches_numpy():
    rng = np.random.default_rng(0)
    data = rng.normal(3.0, 2.0, size=(1000, 5)).astype(np.float32)
    norm = RunningNorm(5)
    for chunk in np.split(data, 10):
        norm.update(torch.as_tensor(chunk))
    np.testing.assert_allclose(norm.mean.numpy(), data.mean(0), atol=1e-4)
    np.testing.assert_allclose(norm.var.numpy(), data.var(0), rtol=1e-3)


def test_gae_matches_hand_computation():
    cfg = PPOConfig(horizon=3, gamma=0.9, lam=0.8)
    algo = PPO(2, 1, num_envs=1, cfg=cfg, device="cpu")
    algo.rew[:, 0] = torch.tensor([1.0, 0.0, 2.0])
    algo.val[:, 0] = torch.tensor([0.5, 0.4, 0.3])
    algo.done[:, 0] = torch.tensor([0.0, 1.0, 0.0])
    with torch.no_grad():
        for p in algo.ac.critic.parameters():
            p.zero_()  # V(last_obs) = 0
    adv, _ = algo._advantages(torch.zeros(1, 2))
    # t=2: delta = 2 - 0.3; t=1: episode ends, delta = 0 - 0.4; t=0: delta = 1 + 0.9*0.4 - 0.5
    d2, d1, d0 = 2 - 0.3, -0.4, 1 + 0.9 * 0.4 - 0.5
    expected = [d0 + 0.9 * 0.8 * d1, d1, d2]
    np.testing.assert_allclose(adv[:, 0].numpy(), expected, atol=1e-6)


def test_update_learns_a_bandit():
    """Reward = -|a - 0.7|^2 with a constant obs: the mean action should move toward 0.7."""
    torch.manual_seed(0)
    cfg = PPOConfig(horizon=16, epochs=4, minibatches=2, hidden=(32, 32))
    algo = PPO(3, 1, num_envs=64, cfg=cfg, device="cpu")
    obs = torch.ones(64, 3)
    for _ in range(40):
        for _ in range(cfg.horizon):
            a, logp, v = algo.ac.act(obs)
            r = -((a[:, 0] - 0.7) ** 2)
            algo.store(obs, a, logp, v, r, torch.ones(64))
        stats = algo.update(obs)
    assert all(np.isfinite(list(stats.values())))
    mean_action = algo.ac.act(obs[:1], deterministic=True).item()
    assert abs(mean_action - 0.7) < 0.15
