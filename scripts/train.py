"""Train a Zoomie to walk with PPO.

    python scripts/train.py --iters 1500 --run main
    python scripts/train.py --iters 1500 --run no_push --no-push
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zoomies.env import ACT_DIM, OBS_DIM, EnvConfig, ZoomieEnv  # noqa: E402
from zoomies.ppo import PPO, PPOConfig  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--run", default="main")
    ap.add_argument("--no-push", action="store_true", help="train without kicks (ablation)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = ZoomieEnv(EnvConfig(num_envs=args.envs, push=not args.no_push), seed=args.seed)
    algo = PPO(OBS_DIM, ACT_DIM, args.envs, PPOConfig(), device=device)

    out = ROOT / "checkpoints" / args.run
    out.mkdir(parents=True, exist_ok=True)
    log_f = open(out / "log.csv", "w", newline="")
    log = csv.writer(log_f)
    log.writerow(["iter", "env_steps", "seconds", "ep_return", "ep_seconds", "fall_rate", "track_lin_vel", "kl", "std"])

    obs = torch.as_tensor(env.observe(), device=device)
    done_returns, done_steps, done_fell, done_track = [], [], [], []
    t0 = time.perf_counter()
    for it in range(1, args.iters + 1):
        for _ in range(algo.cfg.horizon):
            act, logp, val = algo.ac.act(obs)
            next_obs, rew, done, timeout, info = env.step(act.cpu().numpy())
            rew_t = torch.as_tensor(rew, device=device)
            if info["terminal_obs"] is not None:
                rew_t = algo.bootstrap_timeouts(rew_t, torch.as_tensor(timeout, device=device, dtype=torch.float32),
                                                torch.as_tensor(info["terminal_obs"], device=device))
            algo.store(obs, act, logp, val, rew_t, torch.as_tensor(done, device=device, dtype=torch.float32))
            obs = torch.as_tensor(next_obs, device=device)
            if done.any():
                done_returns += info["ep_return"][done].tolist()
                done_steps += info["ep_steps"][done].tolist()
                done_fell += info["fell"][done].tolist()
                secs = info["ep_steps"][done] * 0.02
                done_track += (info["terms"]["track_lin_vel"][done] / np.maximum(secs, 1e-6)).tolist()
        stats = algo.update(obs)

        if it % 10 == 0 or it == 1:
            k = 500
            ret = np.mean(done_returns[-k:]) if done_returns else float("nan")
            secs = np.mean(done_steps[-k:]) * 0.02 if done_steps else float("nan")
            fall = np.mean(done_fell[-k:]) if done_fell else float("nan")
            track = np.mean(done_track[-k:]) if done_track else float("nan")
            elapsed = time.perf_counter() - t0
            steps = it * algo.cfg.horizon * args.envs
            log.writerow([it, steps, round(elapsed, 1), ret, secs, fall, track, stats["kl"], stats["std"]])
            log_f.flush()
            print(f"it {it:5d} | {steps / 1e6:6.1f}M steps | {elapsed / 60:5.1f} min | return {ret:7.2f} | "
                  f"episode {secs:5.1f} s | falls {fall:5.1%} | track {track:.2f} | std {stats['std']:.2f} | "
                  f"{steps / elapsed:,.0f} steps/s", flush=True)
        if it % 100 == 0 or it == args.iters:
            torch.save({"model": algo.ac.state_dict(), "iter": it, "args": vars(args)}, out / "policy.pt")
    env.close()


if __name__ == "__main__":
    main()
