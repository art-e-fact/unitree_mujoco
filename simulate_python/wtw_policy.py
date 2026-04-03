"""Walk-These-Ways locomotion policy for MuJoCo."""

import os

import numpy as np
import torch

from locomotion_policy import LocomotionPolicy
from wtw_controller import (
    DEFAULT_JOINT_ANGLES_WTW,
    WTW_TO_MUJOCO_CTRL,
    WalkTheseWaysController,
)


class _DirectController(WalkTheseWaysController):
    """WTW controller that reads directly from MuJoCo sensordata."""

    def step_from_mujoco(
        self,
        sensordata: np.ndarray,
        num_motor: int,
        dim_motor_sensor: int,
        commands: np.ndarray,
    ) -> np.ndarray:
        """Run one WTW policy step from MuJoCo sensordata.

        Returns target joint positions in ctrl order (FR, FL, RR, RL).
        """
        joint_pos_ctrl = sensordata[:num_motor]
        joint_vel_ctrl = sensordata[num_motor : 2 * num_motor]
        quat = sensordata[dim_motor_sensor : dim_motor_sensor + 4].astype(np.float32)

        # Reorder from ctrl order (FR,FL,RR,RL) → WTW order (FL,FR,RL,RR)
        joint_pos_wtw = np.array(
            [joint_pos_ctrl[WTW_TO_MUJOCO_CTRL[i]] for i in range(12)],
            dtype=np.float32,
        )
        joint_vel_wtw = np.array(
            [joint_vel_ctrl[WTW_TO_MUJOCO_CTRL[i]] for i in range(12)],
            dtype=np.float32,
        )

        obs = self._build_obs(quat, joint_pos_wtw, joint_vel_wtw, commands)
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

        target_ctrl = np.zeros(12, dtype=np.float64)
        for i in range(12):
            target_ctrl[WTW_TO_MUJOCO_CTRL[i]] = target_pos_wtw[i]
        return target_ctrl

    def _build_obs(self, quat, joint_pos_wtw, joint_vel_wtw, commands):
        obs = np.zeros(self.num_obs, dtype=np.float32)
        obs[0:3] = self.get_gravity_vector(quat)
        obs[3:18] = commands * self.commands_scale
        obs[18:30] = (joint_pos_wtw - DEFAULT_JOINT_ANGLES_WTW) * self.obs_scales["dof_pos"]
        obs[30:42] = joint_vel_wtw * self.obs_scales["dof_vel"]
        obs[42:54] = torch.clip(self.actions, -self.clip_actions, self.clip_actions).numpy()
        obs[54:66] = self.last_actions.numpy()
        obs[66:70] = self.get_clock_inputs(commands)
        return torch.tensor(obs, dtype=torch.float32).unsqueeze(0)


_WTW_DIR = os.path.join(os.path.dirname(__file__), "wtw")


class WtwPolicy(LocomotionPolicy):
    """Walk-These-Ways policy wrapper implementing LocomotionPolicy."""

    def __init__(self, num_motor: int = 12):
        self._ctrl = _DirectController(_WTW_DIR, os.path.join(_WTW_DIR, "parameters_cpu.pkl"))
        self._num_motor = num_motor
        self._dim_motor_sensor = 3 * num_motor

        self.kp = self._ctrl.stiffness
        self.kd = self._ctrl.damping
        self.hz = 50.0

        self.standing_pos = np.zeros(12, dtype=np.float64)
        for i in range(12):
            self.standing_pos[WTW_TO_MUJOCO_CTRL[i]] = DEFAULT_JOINT_ANGLES_WTW[i]

    def step(self, sensordata, vx, vy, vyaw):
        commands = self._ctrl.get_commands(vx, vy, vyaw)
        return self._ctrl.step_from_mujoco(
            sensordata, self._num_motor, self._dim_motor_sensor, commands
        )

    def reset(self):
        self._ctrl.reset()
