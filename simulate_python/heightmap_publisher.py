"""Publish Go2-compatible HeightMap_ DDS messages from MuJoCo ray casting."""

import numpy as np
import mujoco

from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.unitree_go.msg.dds_ import HeightMap_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__HeightMap_

TOPIC = "rt/utlidar/height_map_array"
EMPTY = 1.0e9


class HeightMapPublisher:
    """Publish a grid of heights around the robot using mj_ray.

    Casts downward rays on a grid centered on the robot, excluding robot geoms
    via geom group filtering (groups 2,3 = robot visual/collision in go2.xml).

    Call ChannelFactoryInitialize() before creating this object.

    The default parameters match the Go2 API based on: https://support.unitree.com/home/en/developer/LiDAR_service#heading-6
    """

    def __init__(
        self,
        mj_model,
        mj_data,
        robot_body="base_link",
        width=128,
        height=128,
        resolution=0.06,
        source_offset=1.0,
        debug=False,
    ):
        self._m = mj_model
        self._d = mj_data
        self._body_id = mj_model.body(robot_body).id
        self.width = width
        self.height = height
        self.resolution = resolution
        self._source_offset = source_offset
        self._down = np.array([0.0, 0.0, -1.0])
        self._debug = debug
        self._debug_data = None
        self._debug_xy = None
        self._debug_ray_z = 0.0
        self._debug_logged = False
        if debug:
            self._geomid_buf = np.array([-1], dtype=np.int32)
        # Include geom groups 0,1,4,5; exclude 2,3 (robot visual/collision)
        self._geomgroup = np.array([1, 1, 0, 0, 1, 1], dtype=np.uint8)

        self._msg = unitree_go_msg_dds__HeightMap_()
        self._msg.frame_id = "odom"
        self._msg.resolution = resolution
        self._msg.width = width
        self._msg.height = height

        self._pub = ChannelPublisher(TOPIC, HeightMap_)
        self._pub.Init()
        print(f"[heightmap] Publishing {width}x{height} @ {resolution}m on {TOPIC}")

    def update(self):
        """Cast rays and publish one height map centered on the robot."""
        robot_pos = self._d.xpos[self._body_id]
        robot_xy = robot_pos[:2]
        ray_z = robot_pos[2] + self._source_offset

        half_w = 0.5 * self.width * self.resolution
        half_h = 0.5 * self.height * self.resolution

        # From HeightMap_.idl:
        #   origin[2]  -- "Map frame origin xy-position [m]"
        #              -- "the xyz-axis direction of map frame is aligned with the world frame"
        #   "For a cell whose 2d-array-index is [ix, iy],
        #    its position in world frame is: [origin[0] + ix * resolution, origin[1] + iy * resolution]"
        #
        # So origin is the world-frame position of cell [0,0] (the grid corner),
        # and the grid axes are axis-aligned with the world frame (no yaw rotation).
        origin_x = robot_xy[0] - half_w
        origin_y = robot_xy[1] - half_h

        data = np.full(self.width * self.height, EMPTY, dtype=np.float32)
        pnt = np.array([0.0, 0.0, ray_z])
        geomid_buf = self._geomid_buf if self._debug else None
        elevated_geoms = set() if (self._debug and not self._debug_logged) else None
        if self._debug:
            xy_positions = np.empty((self.width * self.height, 2), dtype=np.float64)

        for iy in range(self.height):
            for ix in range(self.width):
                pnt[0] = origin_x + ix * self.resolution
                pnt[1] = origin_y + iy * self.resolution
                if geomid_buf is not None:
                    geomid_buf[0] = -1
                dist = mujoco.mj_ray(
                    self._m,
                    self._d,
                    pnt,
                    self._down,
                    self._geomgroup,
                    1,
                    -1,
                    geomid_buf,
                    None,
                )
                idx = self.width * iy + ix
                if self._debug:
                    xy_positions[idx] = [pnt[0], pnt[1]]
                if dist >= 0:
                    h = ray_z - dist
                    data[idx] = h
                    if elevated_geoms is not None and h > 0.01:
                        elevated_geoms.add(int(geomid_buf[0]))

        if self._debug:
            self._debug_data = data.copy()
            self._debug_xy = xy_positions
            self._debug_ray_z = ray_z

        if elevated_geoms:
            self._debug_logged = True
            print(f"[heightmap] DEBUG: {len(elevated_geoms)} geom(s) hit above ground:")
            for gid in sorted(elevated_geoms):
                name = (
                    mujoco.mj_id2name(self._m, mujoco.mjtObj.mjOBJ_GEOM, gid)
                    or f"<unnamed:{gid}>"
                )
                body_id = self._m.geom_bodyid[gid]
                body_name = (
                    mujoco.mj_id2name(self._m, mujoco.mjtObj.mjOBJ_BODY, body_id)
                    or f"<unnamed:{body_id}>"
                )
                group = self._m.geom_group[gid]
                print(f"  geom[{gid}] name={name!r} body={body_name!r} group={group}")

        self._msg.stamp = self._d.time
        self._msg.origin = [float(origin_x), float(origin_y)]
        self._msg.data = data.tolist()
        self._pub.Write(self._msg)

    def draw_debug(self, scn, stride=8):
        """Draw subsampled ray lines into scn (e.g. viewer.user_scn).

        Green = ground level, red = elevated hit.
        """
        scn.ngeom = 0
        if self._debug_data is None:
            return
        data = self._debug_data
        xy = self._debug_xy
        ray_z = self._debug_ray_z

        for iy in range(0, self.height, stride):
            for ix in range(0, self.width, stride):
                if scn.ngeom >= scn.maxgeom:
                    return
                idx = self.width * iy + ix
                h = data[idx]
                if h >= EMPTY:
                    continue
                x, y = xy[idx]
                t = min(max(h, 0.0) / 1.0, 1.0)
                mujoco.mjv_connector(
                    scn.geoms[scn.ngeom],
                    mujoco.mjtGeom.mjGEOM_LINE,
                    0.002,
                    np.array([x, y, ray_z], dtype=np.float64),
                    np.array([x, y, h], dtype=np.float64),
                )
                scn.geoms[scn.ngeom].rgba[:] = [t, 1.0 - t, 0.0, 0.5]
                scn.ngeom += 1
