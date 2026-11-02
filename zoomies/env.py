"""Thousands of Zoomies in parallel: physics on MuJoCo's C thread pool, everything else batched in NumPy.

Each policy step: actions -> joint targets -> `Rollout` advances every env by DECIMATION physics
steps on all cores -> observations, rewards and resets are computed for the whole batch at once.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np
from mujoco import rollout

from .robot import DECIMATION, DEFAULT_Q, JOINT_RANGES, KP, STAND_HEIGHT, TIMESTEP, TORQUE_LIMIT, build_xml

DT = TIMESTEP * DECIMATION
ACTION_SCALE = 0.25
OBS_DIM, ACT_DIM = 48, 12
Q_LOW = np.tile([r[0] for r in JOINT_RANGES], 4)
Q_HIGH = np.tile([r[1] for r in JOINT_RANGES], 4)

# Full-physics state layout for this model (no actuator activations, no plugins).
NQ, NV = 19, 18
T_, QP, QV = 0, 1, 1 + NQ


@dataclass
class EnvConfig:
    num_envs: int = 1024
    episode_seconds: float = 20.0
    command_seconds: float = 5.0
    lin_vel_x: tuple[float, float] = (-0.6, 1.2)
    lin_vel_y: tuple[float, float] = (-0.5, 0.5)
    yaw_rate: tuple[float, float] = (-1.0, 1.0)
    still_fraction: float = 0.1          # share of commands that are "stand still"
    push: bool = True
    push_interval: tuple[float, float] = (4.0, 8.0)
    push_speed: float = 1.0              # max velocity change of a kick, m/s
    randomize: bool = True
    friction: tuple[float, float] = (0.4, 1.25)
    added_mass: tuple[float, float] = (-1.0, 2.0)
    obs_noise: float = 1.0               # multiplier on per-channel sensor noise
    reward_scales: dict = field(default_factory=lambda: {
        "track_lin_vel": 1.0, "track_yaw": 0.5, "lin_vel_z": -2.0, "ang_vel_xy": -0.05,
        "orientation": -1.0, "torque": -2e-4, "action_rate": -0.01, "feet_air_time": 1.0,
        "base_height": -10.0, "joint_limits": -2.0,
    })


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """(N, 4) wxyz -> (N, 3, 3)."""
    w, x, y, z = q.T
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], 1).reshape(-1, 3, 3)


class ZoomieEnv:
    def __init__(self, cfg: EnvConfig | None = None, seed: int = 0, nthread: int | None = None):
        self.cfg = cfg = cfg or EnvConfig()
        self.n = cfg.num_envs
        self.rng = np.random.default_rng(seed)
        base = mujoco.MjModel.from_xml_string(build_xml())
        self.base_model = base
        self.models = self._model_variants(base, 32 if cfg.randomize else 1)
        self.env_model = [self.models[i % len(self.models)] for i in range(self.n)]
        self.nthread = nthread or 16
        self.datas = [mujoco.MjData(base) for _ in range(self.nthread)]
        self.runner = rollout.Rollout(nthread=self.nthread)

        d = mujoco.MjData(base)
        mujoco.mj_resetDataKeyframe(base, d, 0)
        self.nstate = mujoco.mj_stateSize(base, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        self.s0 = np.zeros(self.nstate)
        mujoco.mj_getState(base, d, self.s0, mujoco.mjtState.mjSTATE_FULLPHYSICS)

        self.state = np.tile(self.s0, (self.n, 1))
        self._out_state = np.zeros((self.n, DECIMATION, self.nstate))
        self._out_sens = np.zeros((self.n, DECIMATION, base.nsensordata))
        self.actions = np.zeros((self.n, ACT_DIM))
        self.last_actions = np.zeros((self.n, ACT_DIM))
        self.ctrl = np.tile(DEFAULT_Q, (self.n, 1))
        self.commands = np.zeros((self.n, 3))
        self.contact = np.ones((self.n, 4), dtype=bool)
        self.air_time = np.zeros((self.n, 4))
        self.steps = np.zeros(self.n, dtype=np.int64)
        self.next_push = np.zeros(self.n, dtype=np.int64)
        self.max_steps = int(cfg.episode_seconds / DT)
        self.cmd_steps = int(cfg.command_seconds / DT)
        self.ep_return = np.zeros(self.n)
        self.reward_sums = {k: np.zeros(self.n) for k in cfg.reward_scales}

        self._reset(np.arange(self.n))
        # Desynchronize episodes so resets don't all land on the same step.
        self.steps[:] = self.rng.integers(0, self.max_steps, self.n)

    # ---------------------------------------------------------------- setup --

    def _model_variants(self, base, k):
        if k == 1:
            return [base]
        out = []
        torso = base.body("torso").id
        for _ in range(k):
            m = mujoco.MjModel.from_xml_string(build_xml())
            m.geom_friction[:, 0] = self.rng.uniform(*self.cfg.friction)
            m.body_mass[torso] += self.rng.uniform(*self.cfg.added_mass)
            out.append(m)
        return out

    def _sample_commands(self, idx):
        c = self.cfg
        n = len(idx)
        cmd = np.stack([self.rng.uniform(*c.lin_vel_x, n), self.rng.uniform(*c.lin_vel_y, n),
                        self.rng.uniform(*c.yaw_rate, n)], 1)
        cmd[self.rng.random(n) < c.still_fraction] = 0.0
        cmd[np.linalg.norm(cmd[:, :2], axis=1) < 0.15, :2] = 0.0  # tiny commands are just noise
        self.commands[idx] = cmd

    def _reset(self, idx):
        if len(idx) == 0:
            return
        n = len(idx)
        s = np.tile(self.s0, (n, 1))
        s[:, QP + 7:QP + 19] = DEFAULT_Q * self.rng.uniform(0.7, 1.3, (n, 12))
        yaw = self.rng.uniform(-np.pi, np.pi, n)
        s[:, QP + 3] = np.cos(yaw / 2)
        s[:, QP + 6] = np.sin(yaw / 2)
        s[:, QV:QV + 6] = self.rng.uniform(-0.3, 0.3, (n, 6))
        self.state[idx] = s
        self.actions[idx] = 0.0
        self.last_actions[idx] = 0.0
        self.ctrl[idx] = DEFAULT_Q
        self.air_time[idx] = 0.0
        self.contact[idx] = True
        self.steps[idx] = 0
        self.ep_return[idx] = 0.0
        for v in self.reward_sums.values():
            v[idx] = 0.0
        lo, hi = self.cfg.push_interval
        self.next_push[idx] = self.rng.integers(int(lo / DT), int(hi / DT), n)
        self._sample_commands(idx)

    # ------------------------------------------------------------ physics --

    def _body_frame(self):
        qpos, qvel = self.state[:, QP:QP + NQ], self.state[:, QV:QV + NV]
        R = quat_to_mat(qpos[:, 3:7])
        lin_vel = np.einsum("nji,nj->ni", R, qvel[:, 0:3])   # world -> body
        ang_vel = qvel[:, 3:6]                                # free joint angvel is already local
        gravity = -R[:, 2, :]                                 # world -z expressed in body frame
        return qpos, qvel, lin_vel, ang_vel, gravity

    def observe(self) -> np.ndarray:
        qpos, qvel, lin_vel, ang_vel, gravity = self._body_frame()
        k = self.cfg.obs_noise
        obs = np.concatenate([
            lin_vel * 2.0 + k * self.rng.uniform(-0.2, 0.2, (self.n, 3)),
            ang_vel * 0.25 + k * self.rng.uniform(-0.05, 0.05, (self.n, 3)),
            gravity + k * self.rng.uniform(-0.05, 0.05, (self.n, 3)),
            self.commands * np.array([2.0, 2.0, 0.25]),
            (qpos[:, 7:] - DEFAULT_Q) + k * self.rng.uniform(-0.01, 0.01, (self.n, 12)),
            qvel[:, 6:] * 0.05 + k * self.rng.uniform(-0.075, 0.075, (self.n, 12)),
            self.actions,
        ], 1)
        return obs.astype(np.float32)

    def step(self, actions: np.ndarray):
        self.last_actions = self.actions
        self.actions = np.clip(actions, -5.0, 5.0)
        self.ctrl = np.clip(DEFAULT_Q + ACTION_SCALE * self.actions, Q_LOW, Q_HIGH)
        q_before = self.state[:, QP + 7:QP + 19].copy()

        # skip_checks avoids per-call validation but then the output buffers must be ours.
        control = np.ascontiguousarray(np.repeat(self.ctrl[:, None, :], DECIMATION, axis=1))
        self.runner.rollout(self.env_model, self.datas, np.ascontiguousarray(self.state), control,
                            nstep=DECIMATION, skip_checks=True, state=self._out_state, sensordata=self._out_sens)
        self.state = self._out_state[:, -1].copy()
        sens = self._out_sens
        self.steps += 1

        # Kicks: an instant velocity change of the torso, the sim version of a shove.
        if self.cfg.push:
            due = self.steps >= self.next_push
            if due.any():
                idx = np.nonzero(due)[0]
                v = self.cfg.push_speed
                self.state[idx, QV:QV + 2] += self.rng.uniform(-v, v, (len(idx), 2))
                lo, hi = self.cfg.push_interval
                self.next_push[idx] = self.steps[idx] + self.rng.integers(int(lo / DT), int(hi / DT), len(idx))

        resample = (self.steps % self.cmd_steps) == 0
        self._sample_commands(np.nonzero(resample)[0])

        contact = sens.max(axis=1) > 1.0  # any touch force during the 4 substeps
        rew, terms = self._rewards(contact, q_before)
        self.contact = contact

        _, _, _, _, gravity = self._body_frame()
        height = self.state[:, QP + 2]
        fell = (gravity[:, 2] > -0.5) | (height < 0.12)
        timeout = self.steps >= self.max_steps
        done = fell | timeout

        self.ep_return += rew
        for k, v in terms.items():
            self.reward_sums[k] += v
        info = {"fell": fell, "timeout": timeout, "ep_return": self.ep_return.copy(),
                "ep_steps": self.steps.copy(), "terms": {k: v.copy() for k, v in self.reward_sums.items()},
                # The critic needs the last observation *before* the reset to bootstrap time-outs.
                "terminal_obs": self.observe() if timeout.any() else None}
        self._reset(np.nonzero(done)[0])
        return self.observe(), rew.astype(np.float32), done, timeout, info

    # ------------------------------------------------------------ rewards --

    def _rewards(self, contact, q_before):
        qpos, qvel, lin_vel, ang_vel, gravity = self._body_frame()
        cmd = self.commands
        q = qpos[:, 7:]
        torque = np.clip(KP * (self.ctrl - q), -TORQUE_LIMIT, TORQUE_LIMIT)

        first_contact = contact & ~self.contact
        self.air_time += DT
        air = ((self.air_time - 0.25) * first_contact).sum(1) * (np.linalg.norm(cmd[:, :2], axis=1) > 0.1)
        self.air_time[contact] = 0.0

        over = np.maximum(q - (Q_HIGH - 0.05), 0) + np.maximum((Q_LOW + 0.05) - q, 0)
        raw = {
            "track_lin_vel": np.exp(-((cmd[:, :2] - lin_vel[:, :2]) ** 2).sum(1) / 0.25),
            "track_yaw": np.exp(-((cmd[:, 2] - ang_vel[:, 2]) ** 2) / 0.25),
            "lin_vel_z": lin_vel[:, 2] ** 2,
            "ang_vel_xy": (ang_vel[:, :2] ** 2).sum(1),
            "orientation": (gravity[:, :2] ** 2).sum(1),
            "torque": (torque ** 2).sum(1),
            "action_rate": ((self.actions - self.last_actions) ** 2).sum(1),
            "feet_air_time": air,
            "base_height": (qpos[:, 2] - (STAND_HEIGHT - 0.03)) ** 2,
            "joint_limits": over.sum(1),
        }
        terms = {k: raw[k] * s * DT for k, s in self.cfg.reward_scales.items()}
        return sum(terms.values()), terms

    def close(self):
        self.runner.close()
