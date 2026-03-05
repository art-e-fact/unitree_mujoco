#!/usr/bin/env python3
"""
Simulated SportClient RPC server for Go2 + MuJoCo.

Runs alongside unitree_mujoco.py (the bridge). Subscribes to rt/lowstate,
translates high-level SportClient calls into rt/lowcmd via the WTW locomotion
policy.

Usage:
  Terminal 1: cd src/unitree_mujoco/simulate_python && python unitree_mujoco.py
  Terminal 2: cd src/unitree_mujoco/simulate_python && python sport_sim_server.py
  Terminal 3: cd src/unitree_sdk2_python && python example/go2/high_level/go2_sport_client.py lo
"""

import sys
import os
import json
import time
import threading
import argparse
import numpy as np
import torch

# Project root is 3 levels up from simulate_python/
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src", "unitree_sdk2_python"))
sys.path.insert(0, _PROJECT_ROOT)  # for go2_wtw_demo

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber, ChannelPublisher
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.rpc.server import Server
from unitree_sdk2py.rpc.internal import RPC_ERR_SERVER_API_NOT_IMPL
from unitree_sdk2py.go2.sport.sport_api import (
    SPORT_SERVICE_NAME, SPORT_API_VERSION,
    SPORT_API_ID_DAMP, SPORT_API_ID_BALANCESTAND, SPORT_API_ID_STOPMOVE,
    SPORT_API_ID_STANDUP, SPORT_API_ID_STANDDOWN, SPORT_API_ID_RECOVERYSTAND,
    SPORT_API_ID_EULER, SPORT_API_ID_MOVE, SPORT_API_ID_SIT, SPORT_API_ID_RISESIT,
    SPORT_API_ID_SPEEDLEVEL, SPORT_API_ID_HELLO, SPORT_API_ID_STRETCH,
    SPORT_API_ID_CONTENT, SPORT_API_ID_DANCE1, SPORT_API_ID_DANCE2,
    SPORT_API_ID_SWITCHJOYSTICK, SPORT_API_ID_POSE, SPORT_API_ID_SCRAPE,
    SPORT_API_ID_FRONTFLIP, SPORT_API_ID_FRONTJUMP, SPORT_API_ID_FRONTPOUNCE,
    SPORT_API_ID_HEART, SPORT_API_ID_STATICWALK, SPORT_API_ID_TROTRUN,
    SPORT_API_ID_ECONOMICGAIT, SPORT_API_ID_LEFTFLIP, SPORT_API_ID_BACKFLIP,
    SPORT_API_ID_HANDSTAND, SPORT_API_ID_FREEWALK, SPORT_API_ID_FREEBOUND,
    SPORT_API_ID_FREEJUMP, SPORT_API_ID_FREEAVOID, SPORT_API_ID_CLASSICWALK,
    SPORT_API_ID_WALKUPRIGHT, SPORT_API_ID_CROSSSTEP,
    SPORT_API_ID_AUTORECOVERY_SET, SPORT_API_ID_AUTORECOVERY_GET,
    SPORT_API_ID_SWITCHAVOIDMODE,
)
from unitree_sdk2py.utils.crc import CRC

import config

# WTW controller lives in go2_wtw_demo.py at the project root
from go2_wtw_demo import WalkTheseWaysController, DEFAULT_JOINT_ANGLES_WTW, WTW_TO_MUJOCO_CTRL

# ---------------------------------------------------------------------------
# Stand poses from src/unitree_mujoco/example/python/stand_go2.py
# Motor (ctrl) order: FR, FL, RR, RL — same as LowCmd motor_cmd indices
# ---------------------------------------------------------------------------
STAND_UP_POS = np.array([
     0.00571868,  0.608813, -1.21763,   # FR
    -0.00571868,  0.608813, -1.21763,   # FL
     0.00571868,  0.608813, -1.21763,   # RR
    -0.00571868,  0.608813, -1.21763,   # RL
], dtype=np.float64)

STAND_DOWN_POS = np.array([
     0.0473455,  1.22187, -2.44375,     # FR
    -0.0473455,  1.22187, -2.44375,     # FL
     0.0473455,  1.22187, -2.44375,     # RR
    -0.0473455,  1.22187, -2.44375,     # RL
], dtype=np.float64)

TRANSITION_DURATION = 2.0   # seconds (tanh ramp)
CONTROL_DT = 0.01           # 100 Hz (WTW policy still steps at 50 Hz)
WTW_STEP_EVERY = 2          # step WTW policy every N control ticks → 50 Hz


# ---------------------------------------------------------------------------
# WTW controller extended to consume rt/lowstate arrays
# ---------------------------------------------------------------------------
class SportWTWController(WalkTheseWaysController):
    """WTW controller that reads state from a LowState_ message."""

    def step_from_lowstate(self, lowstate: LowState_, commands: np.ndarray) -> np.ndarray:
        """
        Run one WTW policy step from a LowState_ message.

        Returns target joint positions in ctrl order (FR, FL, RR, RL).
        """
        quat = np.array(lowstate.imu_state.quaternion, dtype=np.float32)  # [w,x,y,z]
        joint_pos_wtw = np.array(
            [lowstate.motor_state[WTW_TO_MUJOCO_CTRL[i]].q for i in range(12)],
            dtype=np.float32,
        )
        joint_vel_wtw = np.array(
            [lowstate.motor_state[WTW_TO_MUJOCO_CTRL[i]].dq for i in range(12)],
            dtype=np.float32,
        )

        obs = self._build_obs_arrays(quat, joint_pos_wtw, joint_vel_wtw, commands)
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

        # Reorder from WTW order (FL,FR,RL,RR) to ctrl order (FR,FL,RR,RL)
        target_ctrl = np.zeros(12, dtype=np.float64)
        for i in range(12):
            target_ctrl[WTW_TO_MUJOCO_CTRL[i]] = target_pos_wtw[i]
        return target_ctrl

    def _build_obs_arrays(
        self,
        quat: np.ndarray,
        joint_pos_wtw: np.ndarray,
        joint_vel_wtw: np.ndarray,
        commands: np.ndarray,
    ) -> torch.Tensor:
        obs = np.zeros(self.num_obs, dtype=np.float32)
        obs[0:3]   = self.get_gravity_vector(quat)
        obs[3:18]  = commands * self.commands_scale
        obs[18:30] = (joint_pos_wtw - DEFAULT_JOINT_ANGLES_WTW) * self.obs_scales["dof_pos"]
        obs[30:42] = joint_vel_wtw * self.obs_scales["dof_vel"]
        obs[42:54] = torch.clip(self.actions, -self.clip_actions, self.clip_actions).numpy()
        obs[54:66] = self.last_actions.numpy()
        obs[66:70] = self.get_clock_inputs(commands)
        return torch.tensor(obs, dtype=torch.float32).unsqueeze(0)


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------
class State:
    IDLE           = "idle"            # waiting for first rt/lowstate
    IDLE_CONNECTED = "idle_connected"  # bridge seen, settling for 0.5 s
    STANDING       = "standing"        # holding STAND_UP_POS
    STANDING_UP    = "standing_up"     # tanh transition → STAND_UP_POS
    STANDING_DOWN  = "standing_down"   # tanh transition → STAND_DOWN_POS
    WALKING        = "walking"         # WTW policy active
    DAMP           = "damp"            # motors braked (kp=0)


# ---------------------------------------------------------------------------
# RPC server
# ---------------------------------------------------------------------------
class SportSimServer(Server):

    def __init__(self, model_dir: str, cfg_path: str):
        super().__init__(SPORT_SERVICE_NAME)

        self._controller = SportWTWController(model_dir, cfg_path)
        self._crc = CRC()

        self._lock = threading.Lock()
        self._lowstate: LowState_ | None = None

        self._state = State.IDLE
        self._idle_connected_at = 0.0
        self._vx = 0.0
        self._vy = 0.0
        self._vyaw = 0.0
        self._transition_start = 0.0
        self._transition_from = np.zeros(12)
        self._transition_to   = np.zeros(12)

        self._lowstate_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._lowstate_sub.Init(self._on_lowstate, 10)

        self._lowcmd_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self._lowcmd_pub.Init()

    # ------------------------------------------------------------------ DDS
    def _on_lowstate(self, msg: LowState_):
        with self._lock:
            self._lowstate = msg
            if self._state == State.IDLE:
                self._state = State.IDLE_CONNECTED
                self._idle_connected_at = time.perf_counter()
                print("[sport_sim_server] Bridge connected. Settling…")

            elif self._state == State.IDLE_CONNECTED:
                if time.perf_counter() - self._idle_connected_at >= 0.5:
                    self._transition_from = np.array(
                        [msg.motor_state[i].q for i in range(12)]
                    )
                    self._transition_to = STAND_UP_POS.copy()
                    self._transition_start = time.perf_counter()
                    self._state = State.STANDING_UP
                    print("[sport_sim_server] Standing up gradually…")

    # ------------------------------------------------------------ RPC init
    def Init(self):
        self._SetApiVersion(SPORT_API_VERSION)

        self._RegistHandler(SPORT_API_ID_STANDUP,       self._handle_stand_up,   False)
        self._RegistHandler(SPORT_API_ID_STANDDOWN,     self._handle_stand_down, False)
        self._RegistHandler(SPORT_API_ID_MOVE,          self._handle_move,       False)
        self._RegistHandler(SPORT_API_ID_STOPMOVE,      self._handle_stop_move,  False)
        self._RegistHandler(SPORT_API_ID_DAMP,          self._handle_damp,       False)
        self._RegistHandler(SPORT_API_ID_BALANCESTAND,  self._handle_stub,       False)
        self._RegistHandler(SPORT_API_ID_RECOVERYSTAND, self._handle_stub,       False)

        for api_id in [
            SPORT_API_ID_EULER, SPORT_API_ID_SIT, SPORT_API_ID_RISESIT,
            SPORT_API_ID_SPEEDLEVEL, SPORT_API_ID_HELLO, SPORT_API_ID_STRETCH,
            SPORT_API_ID_CONTENT, SPORT_API_ID_DANCE1, SPORT_API_ID_DANCE2,
            SPORT_API_ID_SWITCHJOYSTICK, SPORT_API_ID_POSE, SPORT_API_ID_SCRAPE,
            SPORT_API_ID_FRONTFLIP, SPORT_API_ID_FRONTJUMP, SPORT_API_ID_FRONTPOUNCE,
            SPORT_API_ID_HEART, SPORT_API_ID_STATICWALK, SPORT_API_ID_TROTRUN,
            SPORT_API_ID_ECONOMICGAIT, SPORT_API_ID_LEFTFLIP, SPORT_API_ID_BACKFLIP,
            SPORT_API_ID_HANDSTAND, SPORT_API_ID_FREEWALK, SPORT_API_ID_FREEBOUND,
            SPORT_API_ID_FREEJUMP, SPORT_API_ID_FREEAVOID, SPORT_API_ID_CLASSICWALK,
            SPORT_API_ID_WALKUPRIGHT, SPORT_API_ID_CROSSSTEP,
            SPORT_API_ID_AUTORECOVERY_SET, SPORT_API_ID_AUTORECOVERY_GET,
            SPORT_API_ID_SWITCHAVOIDMODE,
        ]:
            self._RegistHandler(api_id, self._handle_not_impl, False)

    # --------------------------------------------------------- RPC handlers
    def _handle_stand_up(self, parameter: str):
        with self._lock:
            if self._lowstate is None:
                return 0, ""
            self._transition_from = np.array(
                [self._lowstate.motor_state[i].q for i in range(12)]
            )
            self._transition_to = STAND_UP_POS.copy()
            self._transition_start = time.perf_counter()
            self._state = State.STANDING_UP
            print("[sport_sim_server] StandUp")
        return 0, ""

    def _handle_stand_down(self, parameter: str):
        with self._lock:
            if self._lowstate is None:
                return 0, ""
            self._transition_from = np.array(
                [self._lowstate.motor_state[i].q for i in range(12)]
            )
            self._transition_to = STAND_DOWN_POS.copy()
            self._transition_start = time.perf_counter()
            self._state = State.STANDING_DOWN
            print("[sport_sim_server] StandDown")
        return 0, ""

    def _handle_move(self, parameter: str):
        p = json.loads(parameter)
        with self._lock:
            self._vx   = float(p.get("x", 0.0))
            self._vy   = float(p.get("y", 0.0))
            self._vyaw = float(p.get("z", 0.0))
            self._state = State.WALKING
            print(f"[sport_sim_server] Move vx={self._vx:.2f} vy={self._vy:.2f} vyaw={self._vyaw:.2f}")
        return 0, ""

    def _handle_stop_move(self, parameter: str):
        with self._lock:
            self._vx = self._vy = self._vyaw = 0.0
            print("[sport_sim_server] StopMove")
        return 0, ""

    def _handle_damp(self, parameter: str):
        with self._lock:
            self._state = State.DAMP
            print("[sport_sim_server] Damp")
        return 0, ""

    def _handle_stub(self, parameter: str):
        return 0, ""

    def _handle_not_impl(self, parameter: str):
        return RPC_ERR_SERVER_API_NOT_IMPL, ""

    # ------------------------------------------------------- control loop
    def _make_lowcmd(self, target_ctrl: np.ndarray, kp: float, kd: float) -> LowCmd_:
        cmd = unitree_go_msg_dds__LowCmd_()
        cmd.head[0] = 0xFE
        cmd.head[1] = 0xEF
        cmd.level_flag = 0xFF
        for i in range(12):
            cmd.motor_cmd[i].mode = 0x01
            cmd.motor_cmd[i].q   = float(target_ctrl[i])
            cmd.motor_cmd[i].kp  = float(kp)
            cmd.motor_cmd[i].dq  = 0.0
            cmd.motor_cmd[i].kd  = float(kd)
            cmd.motor_cmd[i].tau = 0.0
        cmd.crc = self._crc.Crc(cmd)
        return cmd

    def run_control_loop(self):
        print("[sport_sim_server] Control loop running at 100 Hz (WTW at 50 Hz)")
        tick = 0
        last_walking_cmd = None
        while True:
            t0 = time.perf_counter()

            with self._lock:
                state      = self._state
                lowstate   = self._lowstate
                vx, vy, vyaw = self._vx, self._vy, self._vyaw
                t_start    = self._transition_start
                t_from     = self._transition_from.copy()
                t_to       = self._transition_to.copy()

            if lowstate is None:
                time.sleep(CONTROL_DT)
                continue

            cmd = None

            if state == State.DAMP:
                hold = np.array([lowstate.motor_state[i].q for i in range(12)])
                cmd = self._make_lowcmd(hold, kp=0.0, kd=2.0)

            elif state in (State.STANDING_UP, State.STANDING_DOWN):
                elapsed = time.perf_counter() - t_start
                phase   = float(np.tanh(elapsed / TRANSITION_DURATION))
                target  = (1.0 - phase) * t_from + phase * t_to
                kp      = phase * 50.0 + (1.0 - phase) * 10.0
                cmd = self._make_lowcmd(target, kp=kp, kd=3.5)
                if phase >= 0.99:
                    with self._lock:
                        self._state = (
                            State.STANDING if state == State.STANDING_UP else State.IDLE
                        )
                    print(f"[sport_sim_server] Transition done → {self._state}")

            elif state == State.STANDING:
                cmd = self._make_lowcmd(STAND_UP_POS, kp=50.0, kd=3.5)

            elif state == State.WALKING:
                # Step WTW policy at 50 Hz; republish last cmd on odd ticks
                if tick % WTW_STEP_EVERY == 0:
                    commands    = self._controller.get_commands(vx, vy, vyaw)
                    target_ctrl = self._controller.step_from_lowstate(lowstate, commands)
                    last_walking_cmd = self._make_lowcmd(
                        target_ctrl,
                        kp=self._controller.stiffness,
                        kd=self._controller.damping,
                    )
                cmd = last_walking_cmd

            if cmd is not None:
                self._lowcmd_pub.Write(cmd)

            tick += 1
            sleep = CONTROL_DT - (time.perf_counter() - t0)
            if sleep > 0:
                time.sleep(sleep)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Simulated SportClient RPC server")
    parser.add_argument("--interface", default=config.INTERFACE, help="Network interface for DDS")
    parser.add_argument("--domain",    default=config.DOMAIN_ID, type=int, help="DDS domain ID")
    parser.add_argument(
        "--model-dir",
        default=os.path.join(
            _PROJECT_ROOT,
            "src/walk-these-ways-go2/runs/gait-conditioned-agility/"
            "pretrain-go2/train/142238.667503/checkpoints",
        ),
    )
    parser.add_argument(
        "--cfg-path",
        default=os.path.join(
            _PROJECT_ROOT,
            "src/walk-these-ways-go2/runs/gait-conditioned-agility/"
            "pretrain-go2/train/142238.667503/parameters_cpu.pkl",
        ),
    )
    args = parser.parse_args()

    print(f"[sport_sim_server] DDS domain={args.domain} interface={args.interface}")
    ChannelFactoryInitialize(args.domain, args.interface)

    server = SportSimServer(args.model_dir, args.cfg_path)
    server.Init()
    server.Start()
    print("[sport_sim_server] Serving sport RPC. Waiting for bridge…")

    server.run_control_loop()


if __name__ == "__main__":
    main()
