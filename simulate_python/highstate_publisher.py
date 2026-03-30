"""Publish SportModeState_ DDS messages from MuJoCo sensor data.

Partial implementation — only position, velocity, and IMU RPY are populated.
Call ChannelFactoryInitialize() before creating this object.
"""

import math

from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_

TOPIC = "rt/sportmodestate"


class HighStatePublisher:
    """Publish robot high-level state (position, velocity, IMU) from MuJoCo sensors.

    Reads from mj_data.sensordata using the same layout as LowState_:
      [0 .. num_motor-1]           joint positions
      [num_motor .. 2*num_motor-1] joint velocities
      [2*num_motor .. 3*num_motor] joint torques
      [3*num_motor + 0..3]         IMU quaternion (w, x, y, z)
      [3*num_motor + 4..6]         IMU gyroscope
      [3*num_motor + 7..9]         IMU accelerometer
      [3*num_motor + 10..12]       frame position (x, y, z)
      [3*num_motor + 13..15]       frame linear velocity (x, y, z)
    """

    def __init__(self, num_motor: int):
        self._num_motor = num_motor
        self._dim_motor = 3 * num_motor  # q, dq, tau_est
        self._msg = unitree_go_msg_dds__SportModeState_()
        self._pub = ChannelPublisher(TOPIC, SportModeState_)
        self._pub.Init()

    def update(self, sensordata):
        """Read sensors and publish one SportModeState_ message."""
        sd = sensordata
        d = self._dim_motor
        msg = self._msg

        # Position & velocity from framepos / framelinvel sensors
        for k in range(3):
            msg.position[k] = sd[d + 10 + k]
            msg.velocity[k] = sd[d + 13 + k]

        # IMU quaternion (w, x, y, z) → roll/pitch/yaw
        w, x, y, z = sd[d], sd[d + 1], sd[d + 2], sd[d + 3]
        msg.imu_state.rpy[0] = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        msg.imu_state.rpy[1] = math.asin(max(-1, min(1, 2 * (w * y - z * x))))
        msg.imu_state.rpy[2] = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

        # TODO: populate msg.imu_state.quaternion
        # TODO: populate msg.imu_state.gyroscope
        # TODO: populate msg.imu_state.accelerometer
        # TODO: populate msg.foot_position_body (foot positions in body frame)
        # TODO: populate msg.foot_speed_body (foot velocities in body frame)
        # TODO: populate msg.foot_force (estimated contact forces)
        # TODO: populate msg.body_height
        # TODO: populate msg.yaw_speed
        # TODO: populate msg.mode / msg.gait_type

        self._pub.Write(msg)
