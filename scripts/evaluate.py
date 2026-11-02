"""Measure what the policies can actually do.

1. Command tracking: hold a velocity command for 10 s, compare commanded vs achieved velocity.
2. Kick recovery: walk at 0.5 m/s, get kicked sideways at a given strength, survive 4 s or not.

    python scripts/evaluate.py --policies main no_push
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zoomies.env import DT, EnvConfig, ZoomieEnv, quat_to_mat, QP, QV  # noqa: E402
from zoomies.policy import Policy, kick  # noqa: E402

N = 256
COMMANDS = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0), (-0.4, 0.0, 0.0),
            (0.0, 0.4, 0.0), (0.0, 0.0, 0.8), (0.6, 0.0, 0.6)]
KICKS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


def make_env(seed):
    # Fresh friction/mass randomization (new seed), commands held fixed, no random pushes.
    cfg = EnvConfig(num_envs=N, push=False, command_seconds=1e9, still_fraction=0.0)
    env = ZoomieEnv(cfg, seed=seed)
    env._reset(np.arange(N))
    return env


def set_command(env, cmd):
    env.commands[:] = cmd


def run(env, policy, seconds, record_vel=False):
    obs, fell_any, vels = env.observe(), np.zeros(N, bool), []
    for _ in range(int(seconds / DT)):
        obs, _, done, _, info = env.step(policy(obs))
        fell_any |= info["fell"]
        if record_vel:
            R = quat_to_mat(env.state[:, QP + 3:QP + 7])
            lin = np.einsum("nji,nj->ni", R, env.state[:, QV:QV + 3])
            vels.append(np.concatenate([lin[:, :2], env.state[:, QV + 5:QV + 6]], 1))
    return fell_any, (np.array(vels) if record_vel else None)


def tracking(policy, seed=100):
    rows = []
    for cmd in COMMANDS:
        env = make_env(seed)
        set_command(env, cmd)
        run(env, policy, 2.0)                       # get going
        fell, vel = run(env, policy, 8.0, record_vel=True)
        ok = ~fell
        achieved = vel[:, ok].mean(axis=(0, 1)) if ok.any() else np.full(3, np.nan)
        rows.append((cmd, achieved, fell.mean()))
        env.close()
    return rows


def kick_survival(policy, seed=200):
    out = []
    for k in KICKS:
        env = make_env(seed)
        set_command(env, (0.5, 0.0, 0.0))
        run(env, policy, 2.0)
        ang = env.rng.uniform(0, 2 * np.pi, N)  # random direction, mostly not straight ahead
        kick(env, np.arange(N), k * np.stack([np.cos(ang), np.sin(ang)], 1))
        fell, _ = run(env, policy, 4.0)
        out.append(1.0 - fell.mean())
        env.close()
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="+", default=["main"])
    args = ap.parse_args()

    survival = {}
    for name in args.policies:
        pol = Policy(ROOT / "checkpoints" / name / "policy.pt")
        print(f"\n## {name} (iteration {pol.iter})\n")
        print("| Command (vx, vy, yaw) | Achieved (vx, vy, yaw) | Fell |\n|---|---|---|")
        for cmd, ach, fell in tracking(pol):
            print(f"| {cmd[0]:+.1f} m/s, {cmd[1]:+.1f} m/s, {cmd[2]:+.1f} rad/s "
                  f"| {ach[0]:+.2f}, {ach[1]:+.2f}, {ach[2]:+.2f} | {fell:.1%} |")
        survival[name] = kick_survival(pol)
        print("\n| Kick | " + " | ".join(f"{k} m/s" for k in KICKS) + " |")
        print("|---|" + "---|" * len(KICKS))
        print("| Survived | " + " | ".join(f"{s:.0%}" for s in survival[name]) + " |")

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=120)
    ax.set_facecolor("#0d0b1a")
    colors = {"main": "#2de2e6", "no_push": "#ff5fa2"}
    labels = {"main": "trained with kicks", "no_push": "trained without kicks"}
    for name, s in survival.items():
        ax.plot(KICKS, 100 * s, "o-", lw=2.5, color=colors.get(name, "#ffd166"), label=labels.get(name, name))
    ax.set_xlabel("kick strength (instant velocity change, m/s)")
    ax.set_ylabel("still standing after 4 s (%)")
    ax.set_ylim(-5, 105)
    ax.set_title(f"Kick recovery, {N} robots per point, random kick direction")
    ax.grid(alpha=0.15)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ROOT / "assets" / "kick_survival.png", facecolor="#0d0b1a")
    print("\nsaved assets/kick_survival.png")


if __name__ == "__main__":
    main()
