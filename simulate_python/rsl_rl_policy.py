"""RSL-RL locomotion policy with MuJoCo height scanning."""

from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml

from locomotion_policy import LocomotionPolicy

# Joint order mapping between Isaac Lab and MuJoCo.
#
# Isaac Lab (per joint type):  FL_hip FR_hip RL_hip RR_hip | FL_thigh … | FL_calf …
# MuJoCo ctrl (per leg):      FR_hip FR_thigh FR_calf | FL… | RR… | RL…
_RSL_TO_MUJOCO = np.array([3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8])
_MUJOCO_TO_RSL = np.array([1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10])

# Default standing pose in Isaac Lab joint order.
_DEFAULT_JOINT_POS = np.array(
    [0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5],
    dtype=np.float32,
)


def _load_env_yaml(path):
    """Load an Isaac Lab env.yaml, handling Python-specific tags."""

    class _Loader(yaml.SafeLoader):
        pass

    _Loader.add_constructor(
        "tag:yaml.org,2002:python/tuple",
        lambda loader, node: tuple(loader.construct_sequence(node)),
    )
    _Loader.add_constructor(None, lambda loader, node: None)
    with open(path) as f:
        return yaml.load(f, Loader=_Loader)


class _HeightScanner:
    """Cast downward rays on a grid centered on the robot.

    Matches the Isaac Lab RayCaster with grid_pattern(ordering='xy')
    and ray_alignment='yaw'.
    """

    def __init__(self, mj_model, mj_data, resolution, size, clip):
        self._m = mj_model
        self._d = mj_data
        self._down = np.array([0.0, 0.0, -1.0])
        # Exclude geom groups 2, 3 (robot visual/collision in go2.xml)
        self._geomgroup = np.array([1, 1, 0, 0, 1, 1], dtype=np.uint8)

        self.resolution = resolution
        self.size = size
        self.offset = 0.5  # Isaac Lab height_scan default offset
        self.source_offset = 1.0
        self.clip = clip

        self.x_steps = round(size[0] / resolution) + 1  # 14 for 0.8 / 0.06
        self.y_steps = round(size[1] / resolution) + 1  # 11 for 0.6 / 0.06

    def scan(self, robot_pos, yaw):
        """Return flattened height data (x-major, matching Isaac Lab xy ordering)."""
        data = np.empty(self.x_steps * self.y_steps, dtype=np.float32)
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        ray_z = robot_pos[2] + self.source_offset
        pnt = np.array([0.0, 0.0, ray_z])

        idx = 0
        for ix in range(self.x_steps):
            ox = -self.size[0] / 2 + ix * self.resolution
            for iy in range(self.y_steps):
                oy = -self.size[1] / 2 + iy * self.resolution
                pnt[0] = robot_pos[0] + ox * cos_y - oy * sin_y
                pnt[1] = robot_pos[1] + ox * sin_y + oy * cos_y
                dist = mujoco.mj_ray(
                    self._m, self._d, pnt, self._down,
                    self._geomgroup, 1, -1, None, None,
                )
                if dist >= 0:
                    data[idx] = dist - self.source_offset - self.offset
                else:
                    data[idx] = self.clip[1]
                idx += 1

        return np.clip(data, *self.clip)


def print_observation_debug(obs: np.ndarray, default_joint_pos: np.ndarray = None):
    """Print observation in a clear, readable format with statistics."""
    print("\n" + "="*70)
    print("OBSERVATION DEBUG OUTPUT")
    print("="*70)
    
    # Base motion
    print("\n[Base Motion]")
    print(f"  Linear Velocity:  [{obs[0]:7.4f}, {obs[1]:7.4f}, {obs[2]:7.4f}] m/s")
    print(f"  Angular Velocity: [{obs[3]:7.4f}, {obs[4]:7.4f}, {obs[5]:7.4f}] rad/s")
    
    # Gravity
    print("\n[Gravity]")
    print(f"  [{obs[6]:7.4f}, {obs[7]:7.4f}, {obs[8]:7.4f}] m/s²")
    
    # Command
    print("\n[Command]")
    print(f"  Linear X:  {obs[9]:7.4f} m/s")
    print(f"  Linear Y:  {obs[10]:7.4f} m/s")
    print(f"  Angular Z: {obs[11]:7.4f} rad/s")
    
    # Joint states
    print("\n[Joint States (12 joints)]")
    print(f"  Positions (normalized):  {np.array2string(obs[12:24], precision=4, separator=', ')}")
    print(f"  Velocities:              {np.array2string(obs[24:36], precision=4, separator=', ')}")
    
    # Previous action
    print("\n[Previous Action]")
    print(f"  {np.array2string(obs[36:48], precision=4, separator=', ')}")
    
    # Height scan (heightmap)
    height_scan = obs[48:202]
    print("\n[Height Scan (Heightmap - 154 values)]")
    print(f"  Min:    {np.min(height_scan):8.4f} m")
    print(f"  Max:    {np.max(height_scan):8.4f} m")
    print(f"  Mean:   {np.mean(height_scan):8.4f} m")
    print(f"  Median: {np.median(height_scan):8.4f} m")
    print(f"  Std:    {np.std(height_scan):8.4f} m")
    print("="*70 + "\n")


_POLICY_DIR = Path(__file__).resolve().parent / "policies" / "baseline"


class RslRlPolicy(LocomotionPolicy):
    """RSL-RL policy that builds observations from MuJoCo sensordata."""

    def __init__(self, mj_model, mj_data, num_motor: int = 12):
        policy_path = _POLICY_DIR
        cfg = _load_env_yaml(policy_path / "env.yaml")

        # Control parameters from training config
        dt = cfg["sim"]["dt"]
        self.hz = 1.0 / (dt * cfg["decimation"])
        actuator = cfg["scene"]["robot"]["actuators"]["base_legs"]
        self.kp = actuator["stiffness"]
        self.kd = actuator["damping"]
        self._action_scale = cfg["actions"]["joint_pos"]["scale"]

        # Height scanner matching training grid
        hs = cfg["scene"]["height_scanner"]["pattern_cfg"]
        hs_clip = cfg["observations"]["policy"]["height_scan"]["clip"]
        self._scanner = _HeightScanner(
            mj_model, mj_data,
            resolution=hs["resolution"],
            size=hs["size"],
            clip=hs_clip,
        )

        # Load TorchScript model
        self._model = torch.jit.load(str(policy_path / "policy.pt"))
        self._model.eval()
        self._num_motor = num_motor
        self._dim_motor_sensor = 3 * num_motor
        self._prev_action = np.zeros(12, dtype=np.float32)

        # Standing pose in MuJoCo ctrl order
        self.standing_pos = np.zeros(12, dtype=np.float64)
        self.standing_pos[_RSL_TO_MUJOCO] = _DEFAULT_JOINT_POS

        # Debug flag for observation logging
        self.debug_observations = False

        n = self._scanner.x_steps * self._scanner.y_steps
        print(
            f"Loaded RSL-RL policy: hz={self.hz}, kp={self.kp}, kd={self.kd}, "
            f"action_scale={self._action_scale}, height_scan={n}"
        )

    def step(self, sensordata, vx, vy, vyaw):
        d = self._dim_motor_sensor

        # Joint state: ctrl order → Isaac Lab order
        # _RSL_TO_MUJOCO[rsl_i] = mujoco_i, which is exactly what numpy fancy
        # indexing needs: result[i] = sensordata[_RSL_TO_MUJOCO[i]]
        joint_pos = sensordata[: self._num_motor][_RSL_TO_MUJOCO].astype(np.float32)
        joint_vel = sensordata[self._num_motor : 2 * self._num_motor][_RSL_TO_MUJOCO].astype(np.float32)

        # IMU: quaternion (w,x,y,z), gyroscope (body-frame), frame velocity (world)
        quat = sensordata[d : d + 4]
        gyro = sensordata[d + 4 : d + 7]
        robot_pos = sensordata[d + 10 : d + 13]
        frame_vel = sensordata[d + 13 : d + 16]

        # Rotation matrix transpose (world → body)
        w, x, y, z = quat
        R_T = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)],
            [2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)],
            [2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)],
        ], dtype=np.float32)

        lin_vel_body = R_T @ frame_vel
        gravity_body = R_T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)

        # Height scan (grid rotated by yaw only)
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        height_scan = self._scanner.scan(robot_pos, yaw)

        # Build 202-dim observation matching Isaac Lab training layout
        obs = np.zeros(202, dtype=np.float32)
        obs[0:3] = lin_vel_body
        obs[3:6] = gyro
        obs[6:9] = gravity_body
        obs[9:12] = [vx, vy, vyaw]
        obs[12:24] = joint_pos - _DEFAULT_JOINT_POS
        obs[24:36] = joint_vel
        obs[36:48] = self._prev_action
        obs[48:202] = height_scan

        # Debug logging if enabled
        if self.debug_observations:
            print_observation_debug(obs)

        with torch.no_grad():
            action = (
                self._model(torch.from_numpy(obs).unsqueeze(0).float())
                .squeeze(0)
                .numpy()
            )

        self._prev_action = action.copy()

        # Target positions: Isaac Lab order → ctrl order
        target_rsl = _DEFAULT_JOINT_POS + action * self._action_scale
        target_ctrl = np.zeros(12, dtype=np.float64)
        target_ctrl[_RSL_TO_MUJOCO] = target_rsl
        return target_ctrl

    def reset(self):
        self._prev_action = np.zeros(12, dtype=np.float32)
