"""
Walk-These-Ways locomotion policy controller for the Go2 robot.

Loads pre-trained TorchScript checkpoints and exposes a step() interface
compatible with both the MuJoCo direct-integration (sport_mujoco.py) and
the standalone demo (go2_wtw_demo.py).
"""

import io
import pickle

import mujoco
import numpy as np
import torch


class CPUUnpickler(pickle.Unpickler):
    """Unpickler that maps CUDA tensors to CPU."""
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
        return super().find_class(module, name)


# Maps WTW joint indices → MuJoCo ctrl indices (different ordering)
# 4 legs × 3 joints: Hip, Thigh, Calf
WTW_TO_MUJOCO_CTRL = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]

# Default joint angles in WTW order (standing pose)
DEFAULT_JOINT_ANGLES_WTW = np.array([
    0.1,  0.8, -1.5,   # FL
   -0.1,  0.8, -1.5,   # FR
    0.1,  1.0, -1.5,   # RL
   -0.1,  1.0, -1.5,   # RR
], dtype=np.float32)


class WalkTheseWaysController:
    def __init__(self, model_dir: str, cfg_path: str):
        with open(cfg_path, "rb") as f:
            pkl_cfg = CPUUnpickler(f).load()
        self.cfg = pkl_cfg["Cfg"]

        self.body              = torch.jit.load(f"{model_dir}/body_latest.jit",              map_location="cpu")
        self.adaptation_module = torch.jit.load(f"{model_dir}/adaptation_module_latest.jit", map_location="cpu")
        self.body.eval()
        self.adaptation_module.eval()

        self.num_obs      = self.cfg["env"]["num_observations"]       # 70
        self.num_hist     = self.cfg["env"]["num_observation_history"] # 30
        self.num_actions  = self.cfg["env"]["num_actions"]             # 12
        self.num_commands = self.cfg["commands"]["num_commands"]        # 15

        self.obs_scales        = self.cfg["obs_scales"]
        self.clip_actions      = self.cfg["normalization"]["clip_actions"]
        self.action_scale      = self.cfg["control"]["action_scale"]
        self.hip_scale_reduction = self.cfg["control"]["hip_scale_reduction"]
        self.stiffness         = self.cfg["control"]["stiffness"]["joint"]
        self.damping           = self.cfg["control"]["damping"]["joint"]

        self.commands_scale = np.array([
            self.obs_scales["lin_vel"],           # 0: lin_vel_x
            self.obs_scales["lin_vel"],           # 1: lin_vel_y
            self.obs_scales["ang_vel"],           # 2: ang_vel_yaw
            self.obs_scales["body_height_cmd"],   # 3: body height
            1.0,                                  # 4: gait freq
            1.0,                                  # 5: gait phase
            1.0,                                  # 6: gait offset
            1.0,                                  # 7: gait bound
            1.0,                                  # 8: gait duration
            self.obs_scales["footswing_height_cmd"], # 9
            self.obs_scales["body_pitch_cmd"],    # 10
            self.obs_scales["body_roll_cmd"],     # 11
            self.obs_scales["stance_width_cmd"],  # 12
            self.obs_scales["stance_length_cmd"], # 13
            self.obs_scales["aux_reward_cmd"],    # 14
        ], dtype=np.float32)

        self.obs_history  = torch.zeros(1, self.num_obs * self.num_hist, dtype=torch.float32)
        self.actions      = torch.zeros(self.num_actions, dtype=torch.float32)
        self.last_actions = torch.zeros(self.num_actions, dtype=torch.float32)
        self.gait_index   = 0.0
        self.dt           = 0.02  # 50 Hz control

        print(f"Loaded Walk-These-Ways policy:")
        print(f"  num_obs: {self.num_obs}, num_hist: {self.num_hist}")
        print(f"  action_scale: {self.action_scale}, hip_scale: {self.hip_scale_reduction}")
        print(f"  stiffness: {self.stiffness}, damping: {self.damping}")

    # ------------------------------------------------------------------
    def get_gravity_vector(self, quat: np.ndarray) -> np.ndarray:
        """Project world gravity [0,0,-1] into body frame. quat: [w,x,y,z]."""
        w, x, y, z = quat
        R = np.array([
            [1 - 2*(y*y + z*z),  2*(x*y - z*w),      2*(x*z + y*w)],
            [2*(x*y + z*w),      1 - 2*(x*x + z*z),  2*(y*z - x*w)],
            [2*(x*z - y*w),      2*(y*z + x*w),      1 - 2*(x*x + y*y)],
        ], dtype=np.float32)
        return R.T @ np.array([0, 0, -1], dtype=np.float32)

    def get_clock_inputs(self, commands: np.ndarray) -> np.ndarray:
        phases, offsets, bounds = commands[5], commands[6], commands[7]
        foot_indices = [
            self.gait_index + phases + offsets + bounds,
            self.gait_index + offsets,
            self.gait_index + bounds,
            self.gait_index + phases,
        ]
        return np.array([np.sin(2 * np.pi * fi) for fi in foot_indices], dtype=np.float32)

    def get_commands(self, vx: float = 0.0, vy: float = 0.0, vyaw: float = 0.0) -> np.ndarray:
        commands = np.zeros(self.num_commands, dtype=np.float32)
        commands[0] = vx
        commands[1] = vy
        commands[2] = vyaw
        commands[4] = 3.0   # gait frequency
        commands[5] = 0.5   # phase
        commands[8] = 0.5   # duration
        commands[9] = 0.08  # footswing height
        return commands

    def update_history(self, obs: torch.Tensor):
        self.obs_history = torch.roll(self.obs_history, -self.num_obs, dims=1)
        self.obs_history[0, -self.num_obs:] = obs[0]

    def build_observation(self, data: mujoco.MjData, commands: np.ndarray) -> torch.Tensor:
        """Build 70-dim observation from MjData (uses qpos/qvel directly)."""
        obs = np.zeros(self.num_obs, dtype=np.float32)
        quat      = data.qpos[3:7]
        joint_pos = data.qpos[7:19]
        joint_vel = data.qvel[6:18]
        obs[0:3]   = self.get_gravity_vector(quat)
        obs[3:18]  = commands * self.commands_scale
        obs[18:30] = (joint_pos - DEFAULT_JOINT_ANGLES_WTW) * self.obs_scales["dof_pos"]
        obs[30:42] = joint_vel * self.obs_scales["dof_vel"]
        obs[42:54] = torch.clip(self.actions, -self.clip_actions, self.clip_actions).numpy()
        obs[54:66] = self.last_actions.numpy()
        obs[66:70] = self.get_clock_inputs(commands)
        return torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

    def step(self, data: mujoco.MjData, commands: np.ndarray) -> np.ndarray:
        """Run one policy step; returns target joint positions in WTW order."""
        obs = self.build_observation(data, commands)
        self.update_history(obs)
        with torch.no_grad():
            latent = self.adaptation_module(self.obs_history)
            action = self.body(torch.cat([self.obs_history, latent], dim=1))
        self.last_actions = self.actions.clone()
        self.actions = action[0].clone()
        scaled = action[0].numpy() * self.action_scale
        scaled[[0, 3, 6, 9]] *= self.hip_scale_reduction
        target_pos_wtw = scaled + DEFAULT_JOINT_ANGLES_WTW
        self.gait_index = (self.gait_index + self.dt * commands[4]) % 1.0
        return target_pos_wtw

    def reset(self):
        self.obs_history.zero_()
        self.actions.zero_()
        self.last_actions.zero_()
        self.gait_index = 0.0
