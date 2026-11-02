"""Zoomie: a 12-DoF quadruped designed for this project. 3 motors per leg: hip roll, hip pitch, knee."""

from __future__ import annotations

import numpy as np

LEGS = ("FL", "FR", "HL", "HR")  # front/hind, left/right
HIP_X, HIP_Y = 0.17, 0.085
THIGH, CALF, FOOT_R = 0.18, 0.18, 0.02

# Crouched standing pose the policy acts around: thigh tilted back, knee bent forward.
DEFAULT_Q = np.tile([0.0, 0.75, -1.5], 4)
STAND_HEIGHT = 0.30
JOINT_RANGES = [(-0.6, 0.6), (-1.0, 2.5), (-2.6, -0.4)]
KP, TORQUE_LIMIT = 30.0, 25.0

TIMESTEP = 0.005
DECIMATION = 4  # physics steps per policy step -> 50 Hz control


def _v(a) -> str:
    return " ".join(f"{x:.6g}" for x in np.asarray(a, dtype=float).ravel())


def _leg(name: str, sx: int, sy: int) -> str:
    joints = ("roll", "pitch", "knee")
    r = [f'range="{lo} {hi}"' for lo, hi in JOINT_RANGES]
    mat = "shell_a" if sx > 0 else "shell_b"
    return f"""      <body name="{name}_hip" pos="{sx * HIP_X} {sy * HIP_Y} 0">
        <joint name="{name}_{joints[0]}" axis="1 0 0" {r[0]}/>
        <geom type="cylinder" fromto="-0.035 0 0 0.035 0 0" size="0.035" material="joint" mass="0.4"/>
        <body name="{name}_thigh" pos="0 {sy * 0.03} 0">
          <joint name="{name}_{joints[1]}" axis="0 1 0" {r[1]}/>
          <geom type="capsule" fromto="0 0 0 0 0 {-THIGH}" size="0.024" material="{mat}" mass="0.6"/>
          <body name="{name}_calf" pos="0 0 {-THIGH}">
            <joint name="{name}_{joints[2]}" axis="0 1 0" {r[2]}/>
            <geom type="capsule" fromto="0 0 0 0 0 {-CALF + FOOT_R}" size="0.016" material="joint" mass="0.2"/>
            <geom name="{name}_foot" type="sphere" pos="0 0 {-CALF}" size="{FOOT_R}" material="foot" mass="0.05"
                  condim="3" friction="1.0 0.02 0.002"/>
            <site name="{name}_foot" pos="0 0 {-CALF}" size="{FOOT_R + 0.004}" rgba="0 0 0 0"/>
          </body>
        </body>
      </body>"""


def build_xml() -> str:
    legs = "\n".join(_leg(n, sx, sy) for n, sx, sy in zip(LEGS, (1, 1, -1, -1), (1, -1, 1, -1)))
    joint_names = [f"{n}_{j}" for n in LEGS for j in ("roll", "pitch", "knee")]
    actuators = "\n".join(
        f'    <position name="{j}" joint="{j}" kp="{KP}" forcerange="{-TORQUE_LIMIT} {TORQUE_LIMIT}"/>'
        for j in joint_names
    )
    touch = "\n".join(f'    <touch name="{n}_touch" site="{n}_foot"/>' for n in LEGS)
    return f"""<mujoco model="zoomie">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{TIMESTEP}" integrator="implicitfast"/>

  <visual>
    <global offwidth="1280" offheight="720"/>
    <headlight ambient="0.35 0.35 0.4" diffuse="0.5 0.5 0.55" specular="0.1 0.1 0.1"/>
    <quality shadowsize="4096"/>
  </visual>

  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.16 0.09 0.32" rgb2="0.01 0.01 0.04" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.13 0.12 0.2" rgb2="0.08 0.08 0.13" width="512" height="512" mark="edge" markrgb="0.45 0.3 0.9"/>
    <material name="floor" texture="grid" texrepeat="40 40" reflectance="0.15"/>
    <material name="shell_a" rgba="0.62 0.36 1 1" specular="0.7" shininess="0.7"/>
    <material name="shell_b" rgba="0.15 0.9 0.95 1" specular="0.7" shininess="0.7"/>
    <material name="joint" rgba="0.1 0.1 0.13 1" specular="0.4"/>
    <material name="foot" rgba="1 0.35 0.65 1" specular="0.5"/>
  </asset>

  <default>
    <joint damping="0.8" armature="0.01"/>
    <geom contype="1" conaffinity="0"/>
  </default>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.9 0.85 1" castshadow="true" mode="trackcom"/>
    <geom name="floor" type="plane" size="50 50 0.1" material="floor" contype="0" conaffinity="1" condim="3"/>
    <body name="torso" pos="0 0 {STAND_HEIGHT}">
      <freejoint/>
      <geom type="box" size="0.2 0.08 0.045" material="shell_a" mass="5.0"/>
      <geom type="box" pos="0.2 0 0.01" size="0.025 0.06 0.035" material="joint" mass="0.2"/>
      <geom type="sphere" pos="0.215 0.03 0.02" size="0.012" rgba="0.3 1 1 1" contype="0" mass="0"/>
      <geom type="sphere" pos="0.215 -0.03 0.02" size="0.012" rgba="0.3 1 1 1" contype="0" mass="0"/>
      <site name="imu" pos="0 0 0"/>
{legs}
    </body>
  </worldbody>

  <actuator>
{actuators}
  </actuator>

  <sensor>
{touch}
  </sensor>

  <keyframe>
    <key name="stand" qpos="0 0 {STAND_HEIGHT} 1 0 0 0 {_v(DEFAULT_Q)}" ctrl="{_v(DEFAULT_Q)}"/>
  </keyframe>
</mujoco>
"""


def load_model():
    import mujoco

    return mujoco.MjModel.from_xml_string(build_xml())
