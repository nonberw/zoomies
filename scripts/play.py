"""Drive a Zoomie yourself.

    python scripts/play.py --policy main

Arrow keys: up/down = forward/back, left/right = turn, , and . = strafe, space = KICK, 0 = stop.
"""

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zoomies.env import DT, EnvConfig, ZoomieEnv  # noqa: E402
from zoomies.policy import Policy, kick  # noqa: E402

UP, DOWN, LEFT, RIGHT, SPACE = 265, 264, 263, 262, 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="main")
    args = ap.parse_args()

    policy = Policy(Path(__file__).resolve().parents[1] / "checkpoints" / args.policy / "policy.pt")
    env = ZoomieEnv(EnvConfig(num_envs=1, push=False, command_seconds=1e9, randomize=False, episode_seconds=1e9))
    env._reset(np.arange(1))
    env.commands[0] = 0.0
    cmd = env.commands[0]
    rng = np.random.default_rng()

    def on_key(key):
        if key == UP:
            cmd[0] = min(cmd[0] + 0.2, 1.2)
        elif key == DOWN:
            cmd[0] = max(cmd[0] - 0.2, -0.6)
        elif key == LEFT:
            cmd[2] = min(cmd[2] + 0.3, 1.0)
        elif key == RIGHT:
            cmd[2] = max(cmd[2] - 0.3, -1.0)
        elif key == ord(","):
            cmd[1] = min(cmd[1] + 0.2, 0.5)
        elif key == ord("."):
            cmd[1] = max(cmd[1] - 0.2, -0.5)
        elif key == ord("0"):
            cmd[:] = 0.0
        elif key == SPACE:
            a = rng.uniform(0, 2 * np.pi)
            kick(env, np.array([0]), 1.5 * np.array([np.cos(a), np.sin(a)]))
        print(f"\rcommand: vx {cmd[0]:+.1f}  vy {cmd[1]:+.1f}  yaw {cmd[2]:+.1f}   ", end="", flush=True)

    model, data = env.base_model, mujoco.MjData(env.base_model)
    obs = env.observe()
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        viewer.cam.distance, viewer.cam.elevation = 2.0, -20
        clock = time.perf_counter()
        while viewer.is_running():
            obs, *_ = env.step(policy(obs))
            mujoco.mj_setState(model, data, env.state[0], mujoco.mjtState.mjSTATE_FULLPHYSICS)
            mujoco.mj_forward(model, data)
            viewer.cam.lookat[:] = data.body("torso").xpos
            viewer.sync()
            clock += DT
            time.sleep(max(0.0, clock - time.perf_counter()))
    env.close()


if __name__ == "__main__":
    main()
