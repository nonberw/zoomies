"""Render assets/demo.gif: walk, turn, strafe, take a kick, keep going.

    python scripts/render_gif.py --policy main
"""

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zoomies.env import DT, QP, EnvConfig, ZoomieEnv  # noqa: E402
from zoomies.policy import Policy, kick  # noqa: E402

# (start second, command vx, vy, yaw, caption)
SCRIPT = [
    (0.0, 0.0, 0.0, 0.0, "standing by"),
    (1.0, 0.9, 0.0, 0.0, "forward 0.9 m/s"),
    (4.0, 0.5, 0.0, 0.9, "forward + turn"),
    (7.0, 0.0, -0.4, 0.0, "strafe right"),
    (9.0, 0.6, 0.0, 0.0, "forward"),
    (13.0, 0.0, 0.0, 0.0, "stop"),
]
# Strengths every robot survived in the per-direction test (README), shown at their true size.
KICKS = [(10.0, np.array([0.0, 1.25])), (11.8, np.array([-0.9, -0.6]))]
SECONDS = 14.5


def font(size):
    try:
        return ImageFont.truetype("consola.ttf", size)
    except OSError:
        return ImageFont.load_default()


def save_gif(frames, path, fps):
    """Shared palette, no dithering: keeps the GIF small (see the YOINK repo for the long story)."""
    picked = frames[:: max(1, len(frames) // 12)]
    key = [(255, 90, 90), (158, 92, 255), (38, 230, 242), (255, 90, 166), (235, 230, 255)]
    rows = len(picked) * frames[0].shape[0] // 5
    swatch = np.repeat(np.repeat(np.array(key, np.uint8)[None], rows, 0), frames[0].shape[1] // len(key) + 1, 1)
    sample = Image.fromarray(np.concatenate(picked + [swatch[:, : frames[0].shape[1]]], axis=0))
    palette = sample.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    imgs = [Image.fromarray(f).quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=round(1000 / fps), loop=0, disposal=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="main")
    ap.add_argument("--out", default=str(ROOT / "assets" / "demo.gif"))
    args = ap.parse_args()

    W, H, FPS = 520, 320, 20
    policy = Policy(ROOT / "checkpoints" / args.policy / "policy.pt")
    env = ZoomieEnv(EnvConfig(num_envs=1, push=False, command_seconds=1e9, randomize=False), seed=3)
    env._reset(np.arange(1))
    env.state[0, QP + 3:QP + 7] = (1, 0, 0, 0)  # face +x for the camera

    model = env.base_model
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, H, W)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 1.9, 120, -20
    big, small = font(16), font(13)

    frames, caption, kick_flash = [], "", None
    every = round(1 / (DT * FPS)) or 1
    obs = env.observe()
    pending = list(KICKS)
    for k in range(int(SECONDS / DT)):
        t = k * DT
        for start, vx, vy, yaw, text in SCRIPT:
            if t >= start:
                env.commands[0] = (vx, vy, yaw)
                caption = text
        if pending and t >= pending[0][0]:
            _, v = pending.pop(0)
            kick(env, np.array([0]), v)
            kick_flash = (t, v)
        obs, _, done, _, info = env.step(policy(obs))
        if info["fell"][0]:
            print(f"fell at t={t:.1f}s")

        if k % every:
            continue
        mujoco.mj_setState(model, data, env.state[0], mujoco.mjtState.mjSTATE_FULLPHYSICS)
        mujoco.mj_forward(model, data)
        torso = data.body("torso").xpos
        cam.lookat[:] = 0.85 * cam.lookat + 0.15 * torso if k else torso
        renderer.update_scene(data, cam)
        flashing = kick_flash and t - kick_flash[0] < 0.6
        if flashing:  # red arrow showing where the kick came from
            v = kick_flash[1] / np.linalg.norm(kick_flash[1])
            start = torso - np.array([v[0], v[1], 0.0]) * 0.7
            scn = renderer.scene
            if scn.ngeom < scn.maxgeom:
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                                    np.array([1, 0.25, 0.3, 1], np.float32))
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.045, start, torso - np.array([v[0], v[1], 0]) * 0.15)
                scn.ngeom += 1
        img = Image.fromarray(renderer.render())
        d = ImageDraw.Draw(img)
        cmd = env.commands[0]
        d.text((12, 10), f"cmd: {caption}", fill=(235, 230, 255), font=big)
        d.text((12, 32), f"vx {cmd[0]:+.1f}  vy {cmd[1]:+.1f}  yaw {cmd[2]:+.1f}   t={t:4.1f}s", fill=(200, 190, 240), font=small)
        if flashing:
            d.text((12, 54), f"KICK {np.linalg.norm(kick_flash[1]):.1f} m/s", fill=(255, 90, 90), font=big)
        frames.append(np.asarray(img))

    renderer.close()
    env.close()
    save_gif(frames, args.out, FPS)
    print(f"wrote {args.out} ({len(frames)} frames, {Path(args.out).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
