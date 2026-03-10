#!/usr/bin/env python3
"""
Unified MuJoCo sim + WTW sport server (direct integration).

Replaces running unitree_mujoco.py + sport_sim_server.py as two separate
processes.  WTW runs directly inside the physics step — no DDS hop for
lowstate/lowcmd, no sync issues at any real-time factor.

The sport RPC is still served over DDS so go2_sport_client.py works unchanged.

Usage (two terminals):
  Terminal 1: cd src/unitree_mujoco/simulate_python
              python sport_mujoco.py
  Terminal 2: cd src/unitree_sdk2_python
              python example/go2/high_level/go2_sport_client.py lo
"""

import sys
import os
import json
import time
import threading
import argparse
import numpy as np
import torch
import mujoco
import mujoco.viewer
from threading import Thread

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src", "unitree_sdk2_python"))
sys.path.insert(0, _PROJECT_ROOT)  # for go2_wtw_demo

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_ as LowState_default
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

import config
from go2_wtw_demo import WalkTheseWaysController, DEFAULT_JOINT_ANGLES_WTW, WTW_TO_MUJOCO_CTRL

# ---------------------------------------------------------------------------
# Stand poses (ctrl order: FR, FL, RR, RL)
# ---------------------------------------------------------------------------
_WTW_STAND_POS = np.zeros(12, dtype=np.float64)
for _i in range(12):
    _WTW_STAND_POS[WTW_TO_MUJOCO_CTRL[_i]] = DEFAULT_JOINT_ANGLES_WTW[_i]
STAND_UP_POS = _WTW_STAND_POS

STAND_DOWN_POS = np.array([
     0.0473455,  1.22187, -2.44375,   # FR
    -0.0473455,  1.22187, -2.44375,   # FL
     0.0473455,  1.22187, -2.44375,   # RR
    -0.0473455,  1.22187, -2.44375,   # RL
], dtype=np.float64)

TRANSITION_DURATION = 2.0   # seconds (tanh ramp)
WTW_HZ = 50                 # target WTW policy rate

# Derived at import time — adapts if config.SIMULATE_DT changes
WTW_STEP_EVERY    = max(1, round(1.0 / (WTW_HZ * config.SIMULATE_DT)))
IDLE_SETTLE_TICKS = round(0.5 / config.SIMULATE_DT)


# ---------------------------------------------------------------------------
# WTW controller — reads directly from MuJoCo sensordata
# ---------------------------------------------------------------------------
class SportDirectController(WalkTheseWaysController):

    def step_from_mujoco(
        self,
        sensordata: np.ndarray,
        num_motor: int,
        dim_motor_sensor: int,
        commands: np.ndarray,
    ) -> np.ndarray:
        """
        Run one WTW policy step from MuJoCo sensordata.

        sensordata layout (same as UnitreeSdk2Bridge.PublishLowState):
          [0 : num_motor]              joint q   (ctrl order)
          [num_motor : 2*num_motor]    joint dq  (ctrl order)
          [dim_motor_sensor + 0 : +4]  IMU quaternion [w, x, y, z]

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

        target_ctrl = np.zeros(12, dtype=np.float64)
        for i in range(12):
            target_ctrl[WTW_TO_MUJOCO_CTRL[i]] = target_pos_wtw[i]
        return target_ctrl

    def _build_obs_arrays(self, quat, joint_pos_wtw, joint_vel_wtw, commands):
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
    IDLE_CONNECTED = "idle_connected"  # settling after startup
    STANDING       = "standing"        # WTW at zero velocity
    STANDING_UP    = "standing_up"     # tanh transition → STAND_UP_POS
    STANDING_DOWN  = "standing_down"   # tanh transition → STAND_DOWN_POS
    WALKING        = "walking"         # WTW with velocity commands
    DAMP           = "damp"            # motors off


# ---------------------------------------------------------------------------
# RPC server (sport commands only — no DDS lowstate/lowcmd)
# ---------------------------------------------------------------------------
class SportMuJoCoServer(Server):
    """
    Serves the sport RPC API over DDS.  State is updated by RPC handlers
    and consumed every physics step by tick().
    """

    def __init__(self, controller: SportDirectController, num_motor: int):
        super().__init__(SPORT_SERVICE_NAME)
        self._controller = controller
        self._num_motor  = num_motor

        self._lock = threading.Lock()

        # Shared state — written by RPC handlers, read by sim thread via tick()
        self._state     = State.IDLE_CONNECTED
        self._vx = self._vy = self._vyaw = 0.0

        # Transition
        self._transition_start_step = 0
        self._transition_from = np.zeros(num_motor)
        self._transition_to   = np.zeros(num_motor)

        # Updated by tick() so RPC handlers can snapshot current joint pos
        self._current_q  = np.zeros(num_motor)
        self._sim_step   = 0
        self._idle_start_step = 0

        # Cached WTW output (ctrl order) — reused between WTW steps
        self._last_wtw_ctrl: np.ndarray | None = None

    # ------------------------------------------------------ RPC registration
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
            self._transition_from = self._current_q.copy()
            self._transition_to   = STAND_UP_POS.copy()
            self._transition_start_step = self._sim_step
            self._state = State.STANDING_UP
            self._controller.reset()
            print("[sport_mujoco] StandUp")
        return 0, ""

    def _handle_stand_down(self, parameter: str):
        with self._lock:
            self._transition_from = self._current_q.copy()
            self._transition_to   = STAND_DOWN_POS.copy()
            self._transition_start_step = self._sim_step
            self._state = State.STANDING_DOWN
            print("[sport_mujoco] StandDown")
        return 0, ""

    def _handle_move(self, parameter: str):
        p = json.loads(parameter)
        with self._lock:
            self._vx   = float(p.get("x", 0.0))
            self._vy   = float(p.get("y", 0.0))
            self._vyaw = float(p.get("z", 0.0))
            self._state = State.WALKING
            print(f"[sport_mujoco] Move vx={self._vx:.2f} vy={self._vy:.2f} vyaw={self._vyaw:.2f}")
        return 0, ""

    def _handle_stop_move(self, parameter: str):
        with self._lock:
            self._vx = self._vy = self._vyaw = 0.0
            print("[sport_mujoco] StopMove")
        return 0, ""

    def _handle_damp(self, parameter: str):
        with self._lock:
            self._state = State.DAMP
            print("[sport_mujoco] Damp")
        return 0, ""

    def _handle_stub(self, _):
        return 0, ""

    def _handle_not_impl(self, _):
        return RPC_ERR_SERVER_API_NOT_IMPL, ""

    # ----------------------------------------- called by sim thread each step
    def tick(self, sensordata: np.ndarray, num_motor: int, dim_motor_sensor: int) -> None:
        """
        Compute and write mj_data.ctrl for the current physics step.

        Called inside the sim loop (under the mujoco lock) on every step.
        Writes ctrl via the provided setter to keep mj_data out of this class.
        Returns (ctrl_target, kp, kd) so the caller can apply them.
        """
        with self._lock:
            step  = self._sim_step
            state = self._state
            vx, vy, vyaw = self._vx, self._vy, self._vyaw
            t_start = self._transition_start_step
            t_from  = self._transition_from
            t_to    = self._transition_to

            # Keep current_q fresh so RPC handlers can snapshot it
            self._current_q = sensordata[:num_motor].copy()
            self._sim_step += 1

        ctrl_target = self._current_q.copy()
        kp, kd = 50.0, 3.5

        if state == State.IDLE_CONNECTED:
            if step - self._idle_start_step >= IDLE_SETTLE_TICKS:
                with self._lock:
                    self._controller.reset()
                    self._state = State.STANDING
                print("[sport_mujoco] Standing complete.")
            # Hold keyframe during settle — ctrl_target already = current_q

        elif state == State.DAMP:
            kp, kd = 0.0, 2.0

        elif state in (State.STANDING_UP, State.STANDING_DOWN):
            elapsed = (step - t_start) * config.SIMULATE_DT
            phase   = float(np.tanh(elapsed / TRANSITION_DURATION))
            ctrl_target = (1.0 - phase) * t_from + phase * t_to
            kp = phase * 50.0 + (1.0 - phase) * 20.0
            if phase >= 0.99:
                with self._lock:
                    self._state = (
                        State.STANDING if state == State.STANDING_UP else State.IDLE_CONNECTED
                    )
                print(f"[sport_mujoco] Transition done → {self._state}")
                if state == State.STANDING_UP:
                    print("[sport_mujoco] Standing complete.")

        elif state in (State.STANDING, State.WALKING):
            v = (vx, vy, vyaw) if state == State.WALKING else (0.0, 0.0, 0.0)
            if step % WTW_STEP_EVERY == 0:
                commands = self._controller.get_commands(*v)
                self._last_wtw_ctrl = self._controller.step_from_mujoco(
                    sensordata, num_motor, dim_motor_sensor, commands
                )
            if self._last_wtw_ctrl is not None:
                ctrl_target = self._last_wtw_ctrl
            kp = self._controller.stiffness
            kd = self._controller.damping

        return ctrl_target, kp, kd


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Unified MuJoCo sim + WTW sport server")
    parser.add_argument("--interface", default=config.INTERFACE)
    parser.add_argument("--domain",    default=config.DOMAIN_ID, type=int)
    parser.add_argument("--headless",  action="store_true", help="Run without viewer")
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

    # --- MuJoCo setup -------------------------------------------------------
    mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
    mj_data  = mujoco.MjData(mj_model)
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    mj_model.opt.timestep = config.SIMULATE_DT

    num_motor        = mj_model.nu
    dim_motor_sensor = 3 * num_motor  # q, dq, tau_est per motor

    # --- Controller + RPC server --------------------------------------------
    controller = SportDirectController(args.model_dir, args.cfg_path)
    server = SportMuJoCoServer(controller, num_motor)

    print(f"[sport_mujoco] DDS domain={args.domain} interface={args.interface}")
    ChannelFactoryInitialize(args.domain, args.interface)
    server.Init()
    server.Start()

    # --- lowstate publisher (re-enables record_joints.py and other subscribers)
    low_state     = LowState_default()
    low_state_pub = ChannelPublisher("rt/lowstate", LowState_)
    low_state_pub.Init()
    print("[sport_mujoco] Serving sport RPC.")
    print(f"[sport_mujoco] WTW every {WTW_STEP_EVERY} steps → {WTW_HZ} Hz sim-time")

    # --- Sim loop -----------------------------------------------------------
    def _step():
        """One physics step: compute ctrl, step, publish lowstate."""
        ctrl_target, kp, kd = server.tick(mj_data.sensordata, num_motor, dim_motor_sensor)
        for i in range(num_motor):
            q  = mj_data.sensordata[i]
            dq = mj_data.sensordata[num_motor + i]
            mj_data.ctrl[i] = kp * (ctrl_target[i] - q) + kd * (-dq)
        mujoco.mj_step(mj_model, mj_data)
        for i in range(num_motor):
            low_state.motor_state[i].q       = mj_data.sensordata[i]
            low_state.motor_state[i].dq      = mj_data.sensordata[num_motor + i]
            low_state.motor_state[i].tau_est = mj_data.sensordata[2 * num_motor + i]
        low_state.imu_state.quaternion[0] = mj_data.sensordata[dim_motor_sensor]
        low_state.imu_state.quaternion[1] = mj_data.sensordata[dim_motor_sensor + 1]
        low_state.imu_state.quaternion[2] = mj_data.sensordata[dim_motor_sensor + 2]
        low_state.imu_state.quaternion[3] = mj_data.sensordata[dim_motor_sensor + 3]
        low_state_pub.Write(low_state)

    if args.headless:
        print("[sport_mujoco] Running headless.")
        while True:
            t0 = time.perf_counter()
            _step()
            dt_left = config.SIMULATE_DT - (time.perf_counter() - t0)
            if dt_left > 0:
                time.sleep(dt_left)
    else:
        viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
        time.sleep(0.2)
        locker = threading.Lock()

        def SimulationThread():
            while viewer.is_running():
                t0 = time.perf_counter()
                with locker:
                    _step()
                dt_left = config.SIMULATE_DT - (time.perf_counter() - t0)
                if dt_left > 0:
                    time.sleep(dt_left)

        def PhysicsViewerThread():
            while viewer.is_running():
                with locker:
                    viewer.sync()
                time.sleep(config.VIEWER_DT)

        sim_thread    = Thread(target=SimulationThread,    daemon=True)
        viewer_thread = Thread(target=PhysicsViewerThread, daemon=True)
        sim_thread.start()
        viewer_thread.start()
        sim_thread.join()


if __name__ == "__main__":
    main()
