"""Publish simulated UwbState_ DDS messages from a MuJoCo mocap body.

Maps a "human_marker" mocap body (the UWB tag/label) and the robot
"base_link" body (the UWB base station) into the UwbState_ spherical-
coordinate convention used by the real Go2 UWB module.

Also subscribes to ``rt/human_marker_pose`` so external code can
reposition the tag at runtime (e.g. a pursuit controller publishing
a Pose_ message).

Call ``ChannelFactoryInitialize()`` before creating this object.
"""

import math
import threading

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import UwbState_
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Pose_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__UwbState_

TOPIC_UWB = "rt/uwbstate"
TOPIC_POSE = "rt/human_marker_pose"


def _quat_to_euler(w, x, y, z):
    """Quaternion (w,x,y,z) → (roll, pitch, yaw) in radians."""
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


class UwbPublisher:
    """Simulate the Go2 UWB module using two MuJoCo bodies.

    * **base** = robot body (``base_link``) — the UWB base station.
    * **tag**  = mocap body (``human_marker``) — the UWB tag/label.

    Each ``update()`` call:
    1. Applies any pending ``Pose_`` from ``rt/human_marker_pose`` to the
       mocap body (so external code can move the tag).
    2. Reads both body poses from ``mj_data``.
    3. Computes UWB spherical coordinates and publishes ``UwbState_``.
    """

    def __init__(self, mj_model, mj_data, robot_body="base_link", marker_body="human_marker"):
        self._m = mj_model
        self._d = mj_data

        self._base_id = mj_model.body(robot_body).id
        self._tag_id = mj_model.body(marker_body).id
        self._mocap_id = mj_model.body(marker_body).mocapid[0]

        # Incoming pose from external publishers
        self._pose_lock = threading.Lock()
        self._pending_pose = None

        self._msg = unitree_go_msg_dds__UwbState_()

        self._pub = ChannelPublisher(TOPIC_UWB, UwbState_)
        self._pub.Init()

        self._sub = ChannelSubscriber(TOPIC_POSE, Pose_)
        self._sub.Init(self._on_pose, 10)

        print(f"[uwb] Publishing on {TOPIC_UWB}, subscribing to {TOPIC_POSE}")

    # -- DDS callback (called from DDS thread) -------------------------------

    def _on_pose(self, msg):
        with self._pose_lock:
            self._pending_pose = msg

    # -- Called every tick from the sim loop ----------------------------------

    def update(self):
        """Apply pending pose, compute UWB fields, publish."""
        # 1. Move the mocap body if a new pose arrived
        with self._pose_lock:
            pose = self._pending_pose
        if pose is not None:
            self._d.mocap_pos[self._mocap_id] = [
                pose.position.x, pose.position.y, pose.position.z,
            ]
            self._d.mocap_quat[self._mocap_id] = [
                pose.orientation.w, pose.orientation.x,
                pose.orientation.y, pose.orientation.z,
            ]

        # 2. Read body poses
        base_pos = self._d.xpos[self._base_id]
        base_quat = self._d.xquat[self._base_id]  # (w, x, y, z)
        tag_pos = self._d.xpos[self._tag_id]
        tag_quat = self._d.xquat[self._tag_id]

        base_roll, base_pitch, base_yaw = _quat_to_euler(*base_quat)
        tag_roll, tag_pitch, tag_yaw = _quat_to_euler(*tag_quat)

        # 3. Relative position: tag in base-local frame
        delta_world = tag_pos - base_pos
        cos_y, sin_y = math.cos(base_yaw), math.sin(base_yaw)
        # Rotate world-frame delta into base-local frame (yaw only)
        local_x = cos_y * delta_world[0] + sin_y * delta_world[1]
        local_y = -sin_y * delta_world[0] + cos_y * delta_world[1]
        local_z = delta_world[2]

        dist = math.sqrt(local_x ** 2 + local_y ** 2 + local_z ** 2)
        horiz = math.sqrt(local_x ** 2 + local_y ** 2)

        # Spherical coordinates in the base frame
        orientation_est = math.atan2(local_y, local_x)             # azimuth
        pitch_est = math.atan2(local_z, horiz) if horiz > 1e-6 else 0.0  # elevation
        yaw_est = tag_yaw - base_yaw  # tag heading in base frame
        # Wrap to [-pi, pi]
        yaw_est = (yaw_est + math.pi) % (2.0 * math.pi) - math.pi

        # 4. Fill message
        msg = self._msg
        msg.orientation_est = orientation_est
        msg.pitch_est = pitch_est
        msg.distance_est = dist
        msg.yaw_est = yaw_est
        msg.tag_roll = tag_roll
        msg.tag_pitch = tag_pitch
        msg.tag_yaw = tag_yaw
        msg.base_roll = base_roll
        msg.base_pitch = base_pitch
        msg.base_yaw = base_yaw

        self._pub.Write(msg)
