"""
Integration test: sport_sim_server + SportClient.

The test publishes a fake rt/lowstate (static standing pose) to simulate the
bridge, so MuJoCo / unitree_mujoco.py is not required.

Topology:
  - subprocess:    sport_sim_server.py
  - this process:  fake bridge (rt/lowstate publisher) + SportClient

Run with: pytest test_sport_server.py -v -s
"""

import os
import sys
import time
import threading
import subprocess
import pytest

os.environ["PYTHONUNBUFFERED"] = "1"

# tests/ is inside src/unitree_mujoco/ — project root is 3 levels up
_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
SIM_DIR     = os.path.join(PROJECT_DIR, "src", "unitree_mujoco", "simulate_python")
SDK_PATH    = os.path.join(PROJECT_DIR, "src", "unitree_sdk2_python")
sys.path.insert(0, SDK_PATH)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
from unitree_sdk2py.go2.sport.sport_client import SportClient

SPORT_SERVER_SCRIPT    = os.path.join(SIM_DIR, "sport_sim_server.py")
HEADLESS_BRIDGE_SCRIPT = os.path.join(PROJECT_DIR, "headless_bridge.py")
SERVER_READY_TIMEOUT = 30   # seconds (WTW model load)
SEQUENCE_SLEEP       = 2    # seconds between commands

# Standing pose in ctrl order (FR, FL, RR, RL) — from stand_go2.py
STAND_UP_POS = [
     0.00571868,  0.608813, -1.21763,
    -0.00571868,  0.608813, -1.21763,
     0.00571868,  0.608813, -1.21763,
    -0.00571868,  0.608813, -1.21763,
]


def _make_lowstate() -> LowState_:
    """Fake rt/lowstate: robot standing, IMU upright."""
    msg = unitree_go_msg_dds__LowState_()
    # IMU: identity quaternion (w=1, x=y=z=0) → gravity straight down → valid for WTW
    msg.imu_state.quaternion[0] = 1.0
    msg.imu_state.quaternion[1] = 0.0
    msg.imu_state.quaternion[2] = 0.0
    msg.imu_state.quaternion[3] = 0.0
    for i in range(12):
        msg.motor_state[i].q  = STAND_UP_POS[i]
        msg.motor_state[i].dq = 0.0
    return msg


def _publish_lowstate(stop_event: threading.Event):
    """Publish fake rt/lowstate at 50 Hz until stop_event is set."""
    pub = ChannelPublisher("rt/lowstate", LowState_)
    pub.Init()
    msg = _make_lowstate()
    while not stop_event.is_set():
        pub.Write(msg)
        time.sleep(0.02)


def _drain_stdout(proc: subprocess.Popen, ready_event: threading.Event, marker: str):
    """Read subprocess stdout continuously; set ready_event when marker is seen."""
    for line in proc.stdout:
        print(f"  [server] {line.rstrip()}")
        if marker in line:
            ready_event.set()


def _start_stack(use_real_bridge: bool):
    """Shared setup: returns (procs, client, stop_event)."""
    stop_event  = threading.Event()
    ready_event = threading.Event()
    procs       = []

    ChannelFactoryInitialize(1, "lo")

    if use_real_bridge:
        bridge_proc = subprocess.Popen(
            [sys.executable, "-u", HEADLESS_BRIDGE_SCRIPT,
             "--interface", "lo", "--domain", "1"],
            cwd=PROJECT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        procs.append(bridge_proc)
        bridge_ready = threading.Event()
        threading.Thread(
            target=_drain_stdout,
            args=(bridge_proc, bridge_ready, "Running"),
            daemon=True,
        ).start()
        assert bridge_ready.wait(timeout=15), "headless_bridge did not start in time"
    else:
        pub_thread = threading.Thread(
            target=_publish_lowstate, args=(stop_event,), daemon=True
        )
        pub_thread.start()

    return stop_event, ready_event, procs


@pytest.fixture(scope="module")
def running_stack():
    """Start sport server + fake lowstate publisher, yield SportClient."""
    stop_event = ready_event = None
    procs = []
    server_proc = None
    try:
        stop_event, ready_event, procs = _start_stack(use_real_bridge=False)

        server_proc = subprocess.Popen(
            [sys.executable, "-u", SPORT_SERVER_SCRIPT,
             "--interface", "lo", "--domain", "1"],
            cwd=PROJECT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        procs.append(server_proc)
        threading.Thread(
            target=_drain_stdout,
            args=(server_proc, ready_event, "Bridge connected"),
            daemon=True,
        ).start()

        ready = ready_event.wait(timeout=SERVER_READY_TIMEOUT)
        assert ready, "sport_sim_server did not connect in time"

        client = SportClient()
        client.SetTimeout(5.0)
        client.Init()

        yield client

    finally:
        if stop_event:
            stop_event.set()
        for proc in reversed(procs):
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()


@pytest.fixture(scope="module")
def running_stack_real_bridge():
    """Start sport server + real headless MuJoCo bridge, yield SportClient."""
    stop_event = ready_event = None
    procs = []
    server_proc = None
    try:
        stop_event, ready_event, procs = _start_stack(use_real_bridge=True)

        server_proc = subprocess.Popen(
            [sys.executable, "-u", SPORT_SERVER_SCRIPT,
             "--interface", "lo", "--domain", "1"],
            cwd=PROJECT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        procs.append(server_proc)
        threading.Thread(
            target=_drain_stdout,
            args=(server_proc, ready_event, "Bridge connected"),
            daemon=True,
        ).start()

        ready = ready_event.wait(timeout=SERVER_READY_TIMEOUT)
        assert ready, "sport_sim_server did not connect in time"

        client = SportClient()
        client.SetTimeout(5.0)
        client.Init()

        yield client

    finally:
        if stop_event:
            stop_event.set()
        for proc in reversed(procs):
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()


# ---------------------------------------------------------------------------
# Tests using fake bridge (fast, no MuJoCo)
# ---------------------------------------------------------------------------

def test_stand_up(running_stack):
    code = running_stack.StandUp()
    print(f"StandUp → {code}")
    assert code == 0
    time.sleep(SEQUENCE_SLEEP)


def test_move_forward(running_stack):
    code = running_stack.Move(0.3, 0.0, 0.0)
    print(f"Move(0.3,0,0) → {code}")
    assert code == 0
    time.sleep(SEQUENCE_SLEEP)


def test_stop_move(running_stack):
    code = running_stack.StopMove()
    print(f"StopMove → {code}")
    assert code == 0
    time.sleep(1)


def test_stand_down(running_stack):
    code = running_stack.StandDown()
    print(f"StandDown → {code}")
    assert code == 0
    time.sleep(SEQUENCE_SLEEP)


def test_damp(running_stack):
    code = running_stack.Damp()
    print(f"Damp → {code}")
    assert code == 0
    time.sleep(1)


def test_stub_returns_ok(running_stack):
    code = running_stack.BalanceStand()
    print(f"BalanceStand (stub) → {code}")
    assert code == 0


def test_not_impl_returns_error(running_stack):
    code = running_stack.BackFlip()
    print(f"BackFlip (not impl) → {code}")
    assert code != 0


# ---------------------------------------------------------------------------
# Tests using real headless MuJoCo bridge
# ---------------------------------------------------------------------------

def test_real_bridge_stand_up(running_stack_real_bridge):
    code = running_stack_real_bridge.StandUp()
    print(f"[real bridge] StandUp → {code}")
    assert code == 0
    time.sleep(SEQUENCE_SLEEP)


def test_real_bridge_move(running_stack_real_bridge):
    code = running_stack_real_bridge.Move(0.3, 0.0, 0.0)
    print(f"[real bridge] Move(0.3,0,0) → {code}")
    assert code == 0
    time.sleep(SEQUENCE_SLEEP)


def test_real_bridge_stop_move(running_stack_real_bridge):
    code = running_stack_real_bridge.StopMove()
    print(f"[real bridge] StopMove → {code}")
    assert code == 0
    time.sleep(1)
