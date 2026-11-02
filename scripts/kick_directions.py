"""Kick survival by direction relative to the robot's heading, while walking at 0.6 m/s.

    python scripts/kick_directions.py --policy main
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zoomies.env import QP, EnvConfig, ZoomieEnv  # noqa: E402
from zoomies.policy import Policy, kick  # noqa: E402

N = 256
DIRECTIONS = {"from behind (pushed forward)": (1, 0), "from the front": (-1, 0), "from the side": (0, 1)}
STRENGTHS = (1.0, 1.25, 1.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="main")
    args = ap.parse_args()
    pol = Policy(ROOT / "checkpoints" / args.policy / "policy.pt")

    print("| Kick direction | " + " | ".join(f"{k} m/s" for k in STRENGTHS) + " |")
    print("|---|" + "---|" * len(STRENGTHS))
    for name, (dx, dy) in DIRECTIONS.items():
        row = []
        for k in STRENGTHS:
            env = ZoomieEnv(EnvConfig(num_envs=N, push=False, command_seconds=1e9, still_fraction=0.0), seed=300)
            env._reset(np.arange(N))
            env.commands[:] = (0.6, 0.0, 0.0)
            obs = env.observe()
            for _ in range(100):
                obs, *_ = env.step(pol(obs))
            q = env.state[:, QP + 3:QP + 7]
            yaw = 2 * np.arctan2(q[:, 3], q[:, 0])
            c, s = np.cos(yaw), np.sin(yaw)
            kick(env, np.arange(N), k * np.stack([c * dx - s * dy, s * dx + c * dy], 1))
            fell = np.zeros(N, bool)
            for _ in range(200):
                obs, _, _, _, info = env.step(pol(obs))
                fell |= info["fell"]
            row.append(f"{1 - fell.mean():.0%}")
            env.close()
        print(f"| {name} | " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
