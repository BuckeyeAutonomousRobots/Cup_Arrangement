import math
import random
import subprocess
import time
import xml.etree.ElementTree as ET

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_srvs.srv import Trigger


class Rubik2x2MechanismController(Node):
    """Kinematic Gazebo-side 2x2 Rubik mechanism.

    The world generator creates eight independent cubie models.  This node owns
    the puzzle state and moves those Gazebo models with /world/*/set_pose so the
    internal cube state is authored in the simulation backend, not Unity.
    """

    def __init__(self):
        super().__init__("rubik_task_controller")

        self.declare_parameter("world_name", "ur_hande_dual_arm_tabletop")
        self.declare_parameter("scene_layout_sdf_path", "/workspace/simulation/worlds/ur_hande_dual_arm_tabletop.sdf")
        self.declare_parameter("root_name", "Sync_Rubik2x2")
        self.declare_parameter("set_pose_timeout_ms", 2000)
        self.declare_parameter("shuffle_moves", 14)

        self.world_name = str(self.get_parameter("world_name").value)
        self.scene_layout_sdf_path = str(self.get_parameter("scene_layout_sdf_path").value)
        self.root_name = str(self.get_parameter("root_name").value)
        self.set_pose_timeout_ms = int(self.get_parameter("set_pose_timeout_ms").value)
        self.shuffle_moves = max(1, int(self.get_parameter("shuffle_moves").value))

        self._cubies = self._load_cubies_from_sdf()
        self._center = self._compute_center(self._cubies)
        self._state = {
            name: {
                "offset": np.array(pose["position"], dtype=float) - self._center,
                "quat": np.array(pose["quat"], dtype=float),
            }
            for name, pose in self._cubies.items()
        }

        self.create_service(Trigger, "/rubik2x2/reset", self._on_reset)
        self.create_service(Trigger, "/rubik2x2/shuffle", self._on_shuffle)
        self.create_service(Trigger, "/rubik2x2/twist_x", lambda req, resp: self._on_twist(req, resp, np.array([1.0, 0.0, 0.0])))
        self.create_service(Trigger, "/rubik2x2/twist_y", lambda req, resp: self._on_twist(req, resp, np.array([0.0, 1.0, 0.0])))
        self.create_service(Trigger, "/rubik2x2/twist_z", lambda req, resp: self._on_twist(req, resp, np.array([0.0, 0.0, 1.0])))

        self.get_logger().info(
            f"Rubik2x2 mechanism ready: cubies={len(self._cubies)}, root={self.root_name}, "
            f"center={self._center.tolist()}, world={self.world_name}"
        )

    def _load_cubies_from_sdf(self):
        root = ET.parse(self.scene_layout_sdf_path).getroot()
        world = root.find("world")
        if world is None:
            world = root.find(".//world")
        if world is None:
            raise RuntimeError(f"No <world> found in {self.scene_layout_sdf_path}")

        cubies = {}
        prefix = f"{self.root_name}_cubie_"
        for model in world.findall("model"):
            name = str(model.get("name", ""))
            if not name.startswith(prefix):
                continue
            xyz, quat = self._parse_pose(model.findtext("pose", default="0 0 0 0 0 0"))
            cubies[name] = {"position": xyz, "quat": quat}

        if len(cubies) != 8:
            raise RuntimeError(f"Expected 8 cubies named {prefix}*, found {len(cubies)}")
        return cubies

    @staticmethod
    def _parse_pose(text):
        parts = [float(v) for v in str(text).split()]
        while len(parts) < 6:
            parts.append(0.0)
        xyz = np.array(parts[:3], dtype=float)
        quat = Rubik2x2MechanismController._quat_from_rpy(parts[3], parts[4], parts[5])
        return xyz, quat

    @staticmethod
    def _compute_center(cubies):
        points = [pose["position"] for pose in cubies.values()]
        return np.mean(np.array(points, dtype=float), axis=0)

    def _on_reset(self, _req, resp):
        self._state = {
            name: {
                "offset": np.array(pose["position"], dtype=float) - self._center,
                "quat": np.array(pose["quat"], dtype=float),
            }
            for name, pose in self._cubies.items()
        }
        ok = self._apply_state()
        resp.success = ok
        resp.message = "rubik reset" if ok else "rubik reset failed"
        return resp

    def _on_shuffle(self, _req, resp):
        axes = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]
        for _ in range(self.shuffle_moves):
            self._twist_state(random.choice(axes), random.choice((-1.0, 1.0)), random.choice((-1, 1)))
        ok = self._apply_state()
        resp.success = ok
        resp.message = f"rubik shuffled {self.shuffle_moves} moves" if ok else "rubik shuffle failed"
        return resp

    def _on_twist(self, _req, resp, axis):
        self._twist_state(axis, layer_sign=1.0, quarter_turns=1)
        ok = self._apply_state()
        resp.success = ok
        resp.message = f"rubik twist axis={axis.tolist()}" if ok else "rubik twist failed"
        return resp

    def _twist_state(self, axis, layer_sign: float, quarter_turns: int):
        axis = axis / np.linalg.norm(axis)
        offsets = np.array([entry["offset"] for entry in self._state.values()])
        layer_center = float(np.max(np.abs(offsets @ axis))) * float(layer_sign)
        threshold = max(0.001, abs(layer_center) * 0.5)
        rot = self._quat_from_axis_angle(axis, math.radians(90.0 * quarter_turns))

        for entry in self._state.values():
            if float(entry["offset"] @ axis) * layer_sign < threshold:
                continue
            entry["offset"] = self._snap_offset(self._rotate_vec(entry["offset"], rot))
            entry["quat"] = self._normalize_quat(self._quat_multiply(rot, entry["quat"]))

    def _snap_offset(self, offset):
        solved_abs = np.max(np.abs(np.array([entry["offset"] for entry in self._state.values()])), axis=0)
        snapped = np.zeros(3, dtype=float)
        for i in range(3):
            snapped[i] = solved_abs[i] if offset[i] >= 0.0 else -solved_abs[i]
        return snapped

    def _apply_state(self):
        ok = True
        for name, entry in self._state.items():
            position = self._center + entry["offset"]
            if not self._set_model_pose(name, position, entry["quat"]):
                ok = False
        return ok

    def _set_model_pose(self, model_name: str, xyz, quat) -> bool:
        req = (
            f'name: "{model_name}" '
            f'position: {{x: {xyz[0]} y: {xyz[1]} z: {xyz[2]}}} '
            f'orientation: {{x: {quat[0]} y: {quat[1]} z: {quat[2]} w: {quat[3]}}}'
        )
        cmd = [
            "ign",
            "service",
            "-s",
            f"/world/{self.world_name}/set_pose",
            "--reqtype",
            "ignition.msgs.Pose",
            "--reptype",
            "ignition.msgs.Boolean",
            "--timeout",
            str(self.set_pose_timeout_ms),
            "--req",
            req,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=4.0, check=False)
        except Exception as exc:
            self.get_logger().warn(f"Set pose failed for {model_name}: {exc}")
            return False
        if result.returncode != 0:
            self.get_logger().warn(f"Set pose failed for {model_name}: {result.stderr.strip()}")
            return False
        time.sleep(0.005)
        return True

    @staticmethod
    def _quat_from_axis_angle(axis, angle):
        axis = np.array(axis, dtype=float)
        axis = axis / max(1e-9, np.linalg.norm(axis))
        s = math.sin(angle * 0.5)
        return Rubik2x2MechanismController._normalize_quat(
            np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(angle * 0.5)], dtype=float)
        )

    @staticmethod
    def _quat_from_rpy(roll, pitch, yaw):
        cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
        cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
        cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
        return Rubik2x2MechanismController._normalize_quat(
            np.array(
                [
                    sr * cp * cy - cr * sp * sy,
                    cr * sp * cy + sr * cp * sy,
                    cr * cp * sy - sr * sp * cy,
                    cr * cp * cy + sr * sp * sy,
                ],
                dtype=float,
            )
        )

    @staticmethod
    def _quat_multiply(q1, q2):
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return np.array(
            [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ],
            dtype=float,
        )

    @staticmethod
    def _quat_conjugate(q):
        return np.array([-q[0], -q[1], -q[2], q[3]], dtype=float)

    @staticmethod
    def _normalize_quat(q):
        q = np.array(q, dtype=float)
        n = np.linalg.norm(q)
        if n < 1e-9:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        return q / n

    @classmethod
    def _rotate_vec(cls, v, q):
        qv = np.array([v[0], v[1], v[2], 0.0], dtype=float)
        return cls._quat_multiply(cls._quat_multiply(q, qv), cls._quat_conjugate(q))[:3]


def main(args=None):
    rclpy.init(args=args)
    node = Rubik2x2MechanismController()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()
