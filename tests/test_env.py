import mujoco
import numpy as np
import pytest

from zoomies.env import NQ, NV, OBS_DIM, QP, QV, EnvConfig, ZoomieEnv, quat_to_mat
from zoomies.policy import kick
from zoomies.robot import DEFAULT_Q, load_model


@pytest.fixture(scope="module")
def env():
    e = ZoomieEnv(EnvConfig(num_envs=16, push=False, randomize=False), seed=0)
    yield e
    e.close()


def test_quat_to_mat_matches_mujoco():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(20, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    ours = quat_to_mat(q)
    for i in range(20):
        ref = np.zeros(9)
        mujoco.mju_quat2Mat(ref, q[i])
        np.testing.assert_allclose(ours[i].ravel(), ref, atol=1e-12)


def test_state_layout_matches_mujoco():
    m = load_model()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.qvel[:] = np.arange(NV)
    s = np.zeros(mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_FULLPHYSICS))
    mujoco.mj_getState(m, d, s, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    np.testing.assert_array_equal(s[QP:QP + NQ], d.qpos)
    np.testing.assert_array_equal(s[QV:QV + NV], d.qvel)


def test_standing_pose_holds_without_actions():
    e = ZoomieEnv(EnvConfig(num_envs=8, push=False, randomize=False), seed=1)
    e._reset(np.arange(8))
    e.state[:, QV:QV + NV] = 0.0
    e.state[:, QP + 7:QP + 19] = DEFAULT_Q
    for _ in range(100):  # 2 seconds of zero actions = PD holding the default pose
        _, _, _, _, info = e.step(np.zeros((8, 12)))
        assert not info["fell"].any()
    e.close()


def test_body_frame_velocity(env):
    env._reset(np.arange(env.n))
    yaw = np.pi / 2
    env.state[:, QP + 3:QP + 7] = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
    env.state[:, QV:QV + 3] = (0.0, 1.0, 0.0)  # moving along world +y while facing +y
    _, _, lin, _, grav = env._body_frame()
    np.testing.assert_allclose(lin[0], [1.0, 0.0, 0.0], atol=1e-9)  # that's straight ahead
    np.testing.assert_allclose(grav[0], [0.0, 0.0, -1.0], atol=1e-9)


def test_observation_shape_and_values(env):
    obs = env.observe()
    assert obs.shape == (env.n, OBS_DIM) and obs.dtype == np.float32
    assert np.isfinite(obs).all()


def test_perfect_tracking_gets_full_reward(env):
    env._reset(np.arange(env.n))
    env.commands[:] = (0.5, 0.0, 0.0)
    env.state[:, QP + 3:QP + 7] = (1, 0, 0, 0)
    env.state[:, QV:QV + 6] = (0.5, 0, 0, 0, 0, 0)
    _, terms = env._rewards(np.ones((env.n, 4), bool), env.state[:, QP + 7:QP + 19])
    scale = env.cfg.reward_scales["track_lin_vel"] * 0.02
    np.testing.assert_allclose(terms["track_lin_vel"], scale)


def test_kick_changes_velocity(env):
    before = env.state[:, QV:QV + 2].copy()
    kick(env, np.arange(env.n), np.array([1.0, -0.5]))
    np.testing.assert_allclose(env.state[:, QV:QV + 2] - before, np.tile([1.0, -0.5], (env.n, 1)))


def test_same_seed_same_rollout():
    outs = []
    for _ in range(2):
        e = ZoomieEnv(EnvConfig(num_envs=4), seed=7)
        a = np.random.default_rng(0).normal(size=(30, 4, 12))
        for t in range(30):
            obs, *_ = e.step(a[t])
        outs.append(e.state.copy())
        e.close()
    np.testing.assert_array_equal(outs[0], outs[1])
