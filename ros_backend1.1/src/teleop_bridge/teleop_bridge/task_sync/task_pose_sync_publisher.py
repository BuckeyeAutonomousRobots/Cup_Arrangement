import time
from dataclasses import dataclass
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformListener

from teleop_bridge.common.scene_layout import load_scene_object_poses_from_sdf


@dataclass(frozen=True)
class TrackedObjectSpec:
    name: str
    output_topic: str
    frames: tuple
    default_pose_xyz: tuple
    default_pose_xyzw: tuple
    tf_fallback_enabled: bool = False


class CubePoseSyncPublisher(Node):
    def __init__(self):
        super().__init__("task_pose_sync_publisher")

        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("publish_rate_hz", 60.0)
        self.declare_parameter("use_gz_dynamic_pose_topic", True)
        self.declare_parameter("gz_dynamic_pose_topic", "/world/ur_hande_tabletop/dynamic_pose/info")
        self.declare_parameter("gz_pose_timeout_sec", 0.6)
        self.declare_parameter("publish_default_when_unavailable", True)
        self.declare_parameter("warn_interval_sec", 2.0)
        self.declare_parameter("sync_scene_objects", True)
        self.declare_parameter("scene_layout_reload_interval_sec", 1.0)
        self.declare_parameter(
            "scene_layout_sdf_path",
            "/workspace/simulation/worlds/ur_hande_tabletop.sdf",
        )

        self.target_frame = str(self.get_parameter("target_frame").value)
        self.publish_rate_hz = max(1.0, float(self.get_parameter("publish_rate_hz").value))
        self.use_gz_dynamic_pose_topic = bool(self.get_parameter("use_gz_dynamic_pose_topic").value)
        self.gz_dynamic_pose_topic = str(self.get_parameter("gz_dynamic_pose_topic").value)
        self.gz_pose_timeout_sec = max(0.05, float(self.get_parameter("gz_pose_timeout_sec").value))
        self.publish_default_when_unavailable = bool(
            self.get_parameter("publish_default_when_unavailable").value
        )
        self.warn_interval_sec = max(0.1, float(self.get_parameter("warn_interval_sec").value))
        self.sync_scene_objects = bool(self.get_parameter("sync_scene_objects").value)
        self.scene_layout_reload_interval_sec = max(
            0.1,
            float(self.get_parameter("scene_layout_reload_interval_sec").value),
        )
        self.scene_layout_sdf_path = str(self.get_parameter("scene_layout_sdf_path").value)

        self._specs = []
        self._sync_publishers = {}
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._gz_latest = {}
        self._gz_last_rx_time = 0.0
        self._scene_layout_mtime = None
        self._last_scene_layout_reload_check = 0.0
        self._active_frame_by_object = {}
        self._last_warn_time_by_object = {}
        self._last_info_time_by_object = {}
        if self.sync_scene_objects:
            self._reload_scene_object_specs(force=True)
        if self.use_gz_dynamic_pose_topic:
            self.create_subscription(TFMessage, self.gz_dynamic_pose_topic, self._on_gz_dynamic_pose, 50)
        self.create_timer(1.0 / self.publish_rate_hz, self._publish_loop)

        self.get_logger().info(
            f"CubePoseSyncPublisher started: target_frame={self.target_frame}, "
            f"objects={[spec.name for spec in self._specs]}, "
            f"use_gz_dynamic_pose_topic={self.use_gz_dynamic_pose_topic}, "
            f"gz_dynamic_pose_topic={self.gz_dynamic_pose_topic}, "
            f"publish_default_when_unavailable={self.publish_default_when_unavailable}"
        )

    def _maybe_reload_scene_object_specs(self, now: float):
        if not self.sync_scene_objects:
            return
        if (now - self._last_scene_layout_reload_check) < self.scene_layout_reload_interval_sec:
            return
        self._last_scene_layout_reload_check = now
        path = Path(self.scene_layout_sdf_path)
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return
        if self._scene_layout_mtime is None or mtime != self._scene_layout_mtime:
            self._reload_scene_object_specs(force=True)

    def _reload_scene_object_specs(self, force: bool = False):
        path = Path(self.scene_layout_sdf_path)
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = None

        if not force and mtime == self._scene_layout_mtime:
            return

        new_specs = self._build_scene_object_specs()
        new_names = {spec.name for spec in new_specs}
        old_names = {spec.name for spec in self._specs}

        for removed_name in sorted(old_names - new_names):
            publisher = self._sync_publishers.pop(removed_name, None)
            if publisher is not None:
                self.destroy_publisher(publisher)
            self._active_frame_by_object.pop(removed_name, None)
            self._last_warn_time_by_object.pop(removed_name, None)
            self._last_info_time_by_object.pop(removed_name, None)

        for spec in new_specs:
            if spec.name not in self._sync_publishers:
                self._sync_publishers[spec.name] = self.create_publisher(PoseStamped, spec.output_topic, 10)

        self._specs = new_specs
        self._scene_layout_mtime = mtime
        self.get_logger().info(
            "Scene layout sync objects loaded: "
            f"{sorted(new_names)}"
        )

    def _build_scene_object_specs(self):
        scene_layout = load_scene_object_poses_from_sdf(self.scene_layout_sdf_path, name_prefix="Sync_")
        if not scene_layout:
            self.get_logger().warn(
                f"No Sync_* objects found in {self.scene_layout_sdf_path}; no scene-object sync topics will be published."
            )
            return []

        specs = []
        for name in sorted(scene_layout):
            scene_pose = scene_layout[name]
            link_name = scene_pose.link_name or "body_link"
            frames = (
                f"{name}/{link_name}",
                f"{name}::{link_name}",
                f"{name}/body_link",
                f"{name}::body_link",
                f"{name}/plate_link",
                f"{name}::plate_link",
                name,
            )
            specs.append(
                TrackedObjectSpec(
                    name=name,
                    output_topic=f"/unity_sync/{name}_pose",
                    frames=frames,
                    default_pose_xyz=scene_pose.pose_xyz,
                    default_pose_xyzw=scene_pose.pose_xyzw,
                )
            )
        return specs

    @staticmethod
    def _parse_vec3(value, fallback):
        try:
            out = [float(value[0]), float(value[1]), float(value[2])]
            return out
        except Exception:
            return [float(fallback[0]), float(fallback[1]), float(fallback[2])]

    @staticmethod
    def _parse_vec4(value, fallback):
        try:
            out = [float(value[0]), float(value[1]), float(value[2]), float(value[3])]
            return out
        except Exception:
            return [float(fallback[0]), float(fallback[1]), float(fallback[2]), float(fallback[3])]

    def _on_gz_dynamic_pose(self, msg: TFMessage):
        latest = {}
        for tf in msg.transforms:
            child = str(tf.child_frame_id).strip()
            if not child:
                continue
            latest[child] = (
                (
                    float(tf.transform.translation.x),
                    float(tf.transform.translation.y),
                    float(tf.transform.translation.z),
                ),
                self._normalize_quat(
                    (
                        float(tf.transform.rotation.x),
                        float(tf.transform.rotation.y),
                        float(tf.transform.rotation.z),
                        float(tf.transform.rotation.w),
                    )
                ),
            )

        if latest:
            self._gz_latest = latest
            self._gz_last_rx_time = time.monotonic()

    @staticmethod
    def _quat_multiply(q1, q2):
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        )

    @staticmethod
    def _quat_conjugate(q):
        return (-q[0], -q[1], -q[2], q[3])

    @staticmethod
    def _normalize_quat(q):
        x, y, z, w = q
        n = (x * x + y * y + z * z + w * w) ** 0.5
        if n < 1e-9:
            return (0.0, 0.0, 0.0, 1.0)
        inv = 1.0 / n
        return (x * inv, y * inv, z * inv, w * inv)

    def _rotate_vec_by_quat(self, v, q):
        qv = (v[0], v[1], v[2], 0.0)
        qr = self._quat_multiply(self._quat_multiply(q, qv), self._quat_conjugate(q))
        return (qr[0], qr[1], qr[2])

    def _lookup_from_gz_topic(self, spec: TrackedObjectSpec):
        if not self.use_gz_dynamic_pose_topic or not self._gz_latest:
            return False, None
        if (time.monotonic() - self._gz_last_rx_time) > self.gz_pose_timeout_sec:
            return False, None

        world_pose = None
        active_frame = None
        for frame in spec.frames:
            if frame in self._gz_latest:
                world_pose = self._gz_latest[frame]
                active_frame = frame
                break
        if world_pose is None:
            return False, None

        if active_frame != self._active_frame_by_object.get(spec.name):
            self._active_frame_by_object[spec.name] = active_frame
            self.get_logger().info(f"{spec.name} source active: {active_frame}")

        if self.target_frame in self._gz_latest:
            target_pose_world = self._gz_latest[self.target_frame]
        else:
            target_pose_world = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))

        obj_pos_w, obj_quat_w = world_pose
        tgt_pos_w, tgt_quat_w = target_pose_world
        inv_tgt_q = self._quat_conjugate(tgt_quat_w)
        delta_pos_w = (
            obj_pos_w[0] - tgt_pos_w[0],
            obj_pos_w[1] - tgt_pos_w[1],
            obj_pos_w[2] - tgt_pos_w[2],
        )
        rel_pos = self._rotate_vec_by_quat(delta_pos_w, inv_tgt_q)
        rel_quat = self._normalize_quat(self._quat_multiply(inv_tgt_q, obj_quat_w))
        return True, (rel_pos, rel_quat)

    def _lookup_from_tf(self, spec: TrackedObjectSpec):
        active = self._active_frame_by_object.get(spec.name, "")

        if active:
            try:
                tf = self._tf_buffer.lookup_transform(self.target_frame, active, rclpy.time.Time())
                return True, tf
            except Exception:
                self._active_frame_by_object[spec.name] = ""

        for frame in spec.frames:
            try:
                tf = self._tf_buffer.lookup_transform(self.target_frame, frame, rclpy.time.Time())
                self._active_frame_by_object[spec.name] = frame
                self.get_logger().info(f"{spec.name} TF source active: {self.target_frame} <- {frame}")
                return True, tf
            except Exception:
                continue

        return False, None

    def _publish_default(self, spec: TrackedObjectSpec):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.target_frame
        msg.pose.position.x = float(spec.default_pose_xyz[0])
        msg.pose.position.y = float(spec.default_pose_xyz[1])
        msg.pose.position.z = float(spec.default_pose_xyz[2])
        msg.pose.orientation.x = float(spec.default_pose_xyzw[0])
        msg.pose.orientation.y = float(spec.default_pose_xyzw[1])
        msg.pose.orientation.z = float(spec.default_pose_xyzw[2])
        msg.pose.orientation.w = float(spec.default_pose_xyzw[3])
        self._sync_publishers[spec.name].publish(msg)

    def _publish_loop(self):
        now = time.monotonic()
        self._maybe_reload_scene_object_specs(now)

        for spec in self._specs:
            publisher = self._sync_publishers.get(spec.name)
            if publisher is None:
                continue

            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.target_frame

            gz_ok, gz_pose = self._lookup_from_gz_topic(spec)
            if gz_ok and gz_pose is not None:
                pos, quat = gz_pose
                msg.pose.position.x = float(pos[0])
                msg.pose.position.y = float(pos[1])
                msg.pose.position.z = float(pos[2])
                msg.pose.orientation.x = float(quat[0])
                msg.pose.orientation.y = float(quat[1])
                msg.pose.orientation.z = float(quat[2])
                msg.pose.orientation.w = float(quat[3])
                publisher.publish(msg)
                continue

            if spec.tf_fallback_enabled:
                tf_ok, tf = self._lookup_from_tf(spec)
                if tf_ok and tf is not None:
                    msg.pose.position.x = float(tf.transform.translation.x)
                    msg.pose.position.y = float(tf.transform.translation.y)
                    msg.pose.position.z = float(tf.transform.translation.z)
                    msg.pose.orientation.x = float(tf.transform.rotation.x)
                    msg.pose.orientation.y = float(tf.transform.rotation.y)
                    msg.pose.orientation.z = float(tf.transform.rotation.z)
                    msg.pose.orientation.w = float(tf.transform.rotation.w)
                    publisher.publish(msg)
                    continue

            if not self.publish_default_when_unavailable:
                if now - self._last_warn_time_by_object.get(spec.name, 0.0) > self.warn_interval_sec:
                    self.get_logger().warn(f"No live pose found for {spec.name}; not publishing.")
                    self._last_warn_time_by_object[spec.name] = now
                continue

            self._publish_default(spec)
            if now - self._last_info_time_by_object.get(spec.name, 0.0) > 5.0:
                self.get_logger().info(
                    f"Publishing fallback pose for {spec.name}; update frames or provide live Gazebo pose."
                )
                self._last_info_time_by_object[spec.name] = now


def main(args=None):
    rclpy.init(args=args)
    node = CubePoseSyncPublisher()
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
