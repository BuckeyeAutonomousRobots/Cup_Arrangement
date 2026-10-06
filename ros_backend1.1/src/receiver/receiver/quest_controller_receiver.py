import json
import socket
import threading
import time

import rclpy
from geometry_msgs.msg import Pose
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from teleop_bridge_msgs.msg import ReceivedPoseStates


class QuestControllerReceiver(Node):
    def __init__(self):
        super().__init__("quest_controller_receiver")

        self.declare_parameter("listen_ip", "0.0.0.0")
        self.declare_parameter("listen_port", 5005)
        self.declare_parameter("output_topic", "/received_pose_states")
        self.declare_parameter("publish_dual_arm_topics", True)
        self.declare_parameter("left_output_topic", "/left_arm/received_pose_states")
        self.declare_parameter("right_output_topic", "/right_arm/received_pose_states")
        self.declare_parameter("frame_id", "unity_world")
        self.declare_parameter("publish_rate_hz", 60.0)
        self.declare_parameter("stale_timeout_sec", 0.25)
        self.declare_parameter("analog_press_threshold", 0.55)
        self.declare_parameter("tcp_backlog", 1)
        self.declare_parameter("client_idle_disconnect_sec", 1.5)

        listen_ip = str(self.get_parameter("listen_ip").value)
        listen_port = int(self.get_parameter("listen_port").value)
        output_topic = str(self.get_parameter("output_topic").value)
        self.publish_dual_arm_topics = bool(self.get_parameter("publish_dual_arm_topics").value)
        left_output_topic = str(self.get_parameter("left_output_topic").value)
        right_output_topic = str(self.get_parameter("right_output_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        publish_rate_hz = max(1.0, float(self.get_parameter("publish_rate_hz").value))
        self.stale_timeout_sec = max(0.05, float(self.get_parameter("stale_timeout_sec").value))
        self.analog_press_threshold = float(self.get_parameter("analog_press_threshold").value)
        self.tcp_backlog = max(1, int(self.get_parameter("tcp_backlog").value))
        self.client_idle_disconnect_sec = max(
            0.25, float(self.get_parameter("client_idle_disconnect_sec").value)
        )

        self.pub = self.create_publisher(ReceivedPoseStates, output_topic, 20)
        self.left_pub = self.create_publisher(ReceivedPoseStates, left_output_topic, 20)
        self.right_pub = self.create_publisher(ReceivedPoseStates, right_output_topic, 20)

        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind((listen_ip, listen_port))
        self.server_sock.listen(self.tcp_backlog)
        self.server_sock.settimeout(0.5)

        self._client_lock = threading.Lock()
        self._client_sock = None
        self._last_sender = "none"

        self._state_lock = threading.Lock()
        self._latest_state = self._neutral_state(source="startup")
        self._latest_left_state = self._neutral_state(source="left_startup")
        self._latest_right_state = self._neutral_state(source="right_startup")
        self._last_packet_time_monotonic = 0.0
        self._last_client_rx_time_monotonic = 0.0

        self._last_log_time = time.monotonic()
        self._packet_count_window = 0
        self._byte_count_window = 0
        self._decode_error_count_window = 0
        self._window_start = time.monotonic()

        self._json_decoder = json.JSONDecoder()

        self.rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self.rx_thread.start()

        self.create_timer(1.0 / publish_rate_hz, self._publish_loop)

        self.get_logger().info(
            f"QuestControllerReceiver listening on TCP {listen_ip}:{listen_port}, "
            f"publishing {output_topic} @ {publish_rate_hz:.1f} Hz"
        )
        if self.publish_dual_arm_topics:
            self.get_logger().info(
                f"Dual-arm fanout enabled: left={left_output_topic}, right={right_output_topic}"
            )
        self.get_logger().info(
            f"stale_timeout_sec={self.stale_timeout_sec:.3f}, "
            f"analog_press_threshold={self.analog_press_threshold:.2f}, "
            f"client_idle_disconnect_sec={self.client_idle_disconnect_sec:.2f}, "
            f"frame_id={self.frame_id}"
        )

    def _set_client(self, client_sock, sender_label):
        with self._client_lock:
            old = self._client_sock
            self._client_sock = client_sock
            self._last_sender = sender_label
            self._last_client_rx_time_monotonic = time.monotonic()

        if old is not None and old is not client_sock:
            try:
                old.close()
            except Exception:
                pass

    def _drop_client(self, reason):
        with self._client_lock:
            sock = self._client_sock
            sender = self._last_sender
            self._client_sock = None
            self._last_sender = "none"
            self._last_client_rx_time_monotonic = 0.0

        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass
            self.get_logger().info(f"TCP client disconnected ({reason}): {sender}")

    def _rx_loop(self):
        text_buffer = ""

        while rclpy.ok():
            with self._client_lock:
                client_sock = self._client_sock

            if client_sock is None:
                try:
                    new_client, addr = self.server_sock.accept()
                except socket.timeout:
                    continue
                except OSError:
                    return
                except Exception as exc:
                    self.get_logger().error(f"TCP accept error: {exc}")
                    continue

                try:
                    new_client.settimeout(0.5)
                    new_client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                except OSError:
                    pass

                sender = f"{addr[0]}:{addr[1]}"
                self._set_client(new_client, sender)
                text_buffer = ""
                self.get_logger().info(f"TCP client connected: {sender}")
                continue

            try:
                data = client_sock.recv(8192)
            except socket.timeout:
                if (
                    self._last_client_rx_time_monotonic > 0.0
                    and time.monotonic() - self._last_client_rx_time_monotonic
                    > self.client_idle_disconnect_sec
                ):
                    self._drop_client("idle no bytes")
                    text_buffer = ""
                continue
            except OSError:
                self._drop_client("socket error")
                text_buffer = ""
                continue
            except Exception as exc:
                self.get_logger().error(f"TCP recv error: {exc}")
                self._drop_client("exception")
                text_buffer = ""
                continue

            if not data:
                self._drop_client("peer closed")
                text_buffer = ""
                continue

            with self._state_lock:
                self._byte_count_window += len(data)
            self._last_client_rx_time_monotonic = time.monotonic()
            text_buffer += data.decode("utf-8", errors="ignore")
            text_buffer = self._consume_buffer(text_buffer)

    def _consume_buffer(self, text_buffer):
        while True:
            newline_idx = text_buffer.find("\n")
            if newline_idx < 0:
                break

            raw = text_buffer[:newline_idx].strip()
            text_buffer = text_buffer[newline_idx + 1 :]
            if not raw:
                continue

            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                with self._state_lock:
                    self._decode_error_count_window += 1
                continue

            self._store_payload(payload)

        # Fallback for streams that send raw JSON objects without newlines.
        while True:
            text_buffer = text_buffer.lstrip()
            if not text_buffer:
                break

            try:
                payload, consumed = self._json_decoder.raw_decode(text_buffer)
            except json.JSONDecodeError:
                with self._state_lock:
                    self._decode_error_count_window += 1
                if len(text_buffer) > 65536:
                    text_buffer = text_buffer[-65536:]
                break

            self._store_payload(payload)
            text_buffer = text_buffer[consumed:]

        return text_buffer

    def _store_payload(self, payload):
        parsed_state = self._parse_payload(payload, arm="right", compatibility=True)
        left_state = self._parse_payload(payload, arm="left", compatibility=False)
        right_state = self._parse_payload(payload, arm="right", compatibility=False)
        now = time.monotonic()

        with self._state_lock:
            self._latest_state = parsed_state
            self._latest_left_state = left_state
            self._latest_right_state = right_state
            self._last_packet_time_monotonic = now
            self._packet_count_window += 1

    def _parse_payload(self, payload, arm="right", compatibility=False):
        if not isinstance(payload, dict):
            return self._neutral_state(source="invalid_packet")

        controls = payload.get("controls")
        if not isinstance(controls, dict):
            controls = {}

        arm = "left" if arm == "left" else "right"
        hand_key = f"{arm}_hand"
        hand = payload.get(hand_key)
        if not isinstance(hand, dict):
            hand = {}

        tracked = self._as_bool(hand.get("isTracked", False))

        pose = Pose()
        pose.orientation.w = 1.0

        position = hand.get("pos")
        if isinstance(position, dict):
            pose.position.x = self._as_float(position.get("x"), 0.0)
            pose.position.y = self._as_float(position.get("y"), 0.0)
            pose.position.z = self._as_float(position.get("z"), 0.0)

        rotation = hand.get("rot")
        if isinstance(rotation, dict):
            pose.orientation.x = self._as_float(rotation.get("x"), 0.0)
            pose.orientation.y = self._as_float(rotation.get("y"), 0.0)
            pose.orientation.z = self._as_float(rotation.get("z"), 0.0)
            pose.orientation.w = self._as_float(rotation.get("w"), 1.0)

        workspace_pose = Pose()
        workspace_pose.orientation.w = 1.0
        workspace_pose_valid = self._as_bool(controls.get("workspace_pose_valid", False))
        workspace_position = controls.get("workspace_pos")
        workspace_rotation = controls.get("workspace_rot")
        if isinstance(workspace_position, dict):
            workspace_pose.position.x = self._as_float(workspace_position.get("x"), 0.0)
            workspace_pose.position.y = self._as_float(workspace_position.get("y"), 0.0)
            workspace_pose.position.z = self._as_float(workspace_position.get("z"), 0.0)
        else:
            workspace_pose_valid = False

        if isinstance(workspace_rotation, dict):
            workspace_pose.orientation.x = self._as_float(workspace_rotation.get("x"), 0.0)
            workspace_pose.orientation.y = self._as_float(workspace_rotation.get("y"), 0.0)
            workspace_pose.orientation.z = self._as_float(workspace_rotation.get("z"), 0.0)
            workspace_pose.orientation.w = self._as_float(workspace_rotation.get("w"), 1.0)
        else:
            workspace_pose_valid = False

        if arm == "left":
            close_flag = self._as_optional_bool(
                controls.get("left_close_enable", controls.get("left_close_held"))
            )
            open_flag = self._as_optional_bool(
                controls.get("left_open_enable", controls.get("left_open_held"))
            )
            rotate_enable = self._as_bool(
                controls.get("left_rotate_enable", controls.get("left_rotate_held", False))
            )
            teleop_flag = self._as_optional_bool(
                controls.get("left_teleop_enable", controls.get("left_teleop_held"))
            )
            grip_value = self._as_float(controls.get("left_grip_value"), 1.0 if teleop_flag else 0.0)
            trigger_value = self._as_float(controls.get("left_trigger_value"), 1.0 if open_flag else 0.0)
            attachment_mode = self._as_bool(controls.get("left_attachment_mode", False))
            attachment_toggle_enable = self._as_bool(controls.get("left_attachment_toggle_enable", False))
            attachment_position_offset = controls.get("left_attachment_position_offset")
            attachment_rotation_offset = controls.get("left_attachment_rotation_offset")
            source = "quest_left_controller"
        else:
            close_flag = self._as_optional_bool(
                controls.get("right_close_enable", controls.get("close_enable", controls.get("close_held")))
            )
            open_flag = self._as_optional_bool(
                controls.get("right_open_enable", controls.get("open_enable", controls.get("open_held")))
            )
            rotate_enable = self._as_bool(
                controls.get("right_rotate_enable", controls.get("rotate_enable", controls.get("rotate_held", False)))
            )
            teleop_flag = self._as_optional_bool(
                controls.get("right_teleop_enable", controls.get("teleop_enable", controls.get("teleop_held")))
            )
            grip_value = self._as_float(controls.get("grip_value"), 1.0 if teleop_flag else 0.0)
            trigger_value = self._as_float(controls.get("trigger_value"), 1.0 if open_flag else 0.0)
            attachment_mode = self._as_bool(controls.get("right_attachment_mode", False))
            attachment_toggle_enable = self._as_bool(controls.get("right_attachment_toggle_enable", False))
            attachment_position_offset = controls.get("right_attachment_position_offset")
            attachment_rotation_offset = controls.get("right_attachment_rotation_offset")
            source = "quest_right_controller"

        if compatibility:
            source = controls.get("source", source)
            source = str(source) if source is not None else "quest_right_controller"

        reset_flag = self._as_optional_bool(
            controls.get("reset_enable", controls.get("reset_held"))
        )
        arm_reset_robot_key = "left_reset_robot_enable" if arm == "left" else "right_reset_robot_enable"
        arm_reset_robot_held_key = "left_reset_robot_held" if arm == "left" else "right_reset_robot_held"
        arm_reset_robot_flag = self._as_optional_bool(
            controls.get(arm_reset_robot_key, controls.get(arm_reset_robot_held_key))
        )
        reset_robot_flag = self._as_optional_bool(
            controls.get("reset_robot_enable", controls.get("reset_robot_held"))
        )
        reset_scene_flag = self._as_optional_bool(
            controls.get("reset_scene_enable", controls.get("reset_scene_held"))
        )
        recenter_key = "left_recenter_enable" if arm == "left" else "recenter_enable"
        recenter_fallback = "left_recenter_held" if arm == "left" else "recenter_held"
        recenter_flag = self._as_optional_bool(
            controls.get(recenter_key, controls.get(recenter_fallback))
        )

        open_from_analog = trigger_value >= self.analog_press_threshold

        close_enable = close_flag if close_flag is not None else False
        open_enable = open_from_analog if open_flag is None else open_flag

        reset_enable = reset_flag if reset_flag is not None else False
        reset_robot_enable = (
            arm_reset_robot_flag
            if arm_reset_robot_flag is not None
            else (reset_robot_flag if reset_robot_flag is not None else reset_enable)
        )
        reset_scene_enable = reset_scene_flag if reset_scene_flag is not None else reset_enable
        recenter_enable = recenter_flag if recenter_flag is not None else False
        teleop_enable = teleop_flag if teleop_flag is not None else False
        mode_switch_enable = self._as_bool(
            controls.get("mode_switch_enable", controls.get("mode_switch_held", False))
        )

        left_stick_x = self._as_float(controls.get("left_thumbstick_x"), 0.0)
        left_stick_y = self._as_float(controls.get("left_thumbstick_y"), 0.0)
        right_stick_x = self._as_float(controls.get("right_thumbstick_x"), 0.0)
        right_stick_y = self._as_float(controls.get("right_thumbstick_y"), 0.0)
        left_grip_value = self._as_float(controls.get("left_grip_value"), 0.0)
        left_trigger_value = self._as_float(controls.get("left_trigger_value"), 0.0)

        control_mode = self._parse_control_mode(controls.get("control_mode"))
        mapping_mode = self._parse_mapping_mode(controls.get("mapping_mode"))
        attachment_adjustment_mode = self._as_bool(controls.get("attachment_adjustment_mode", False))
        attachment_offset_valid, attachment_offset_pose = self._parse_attachment_offset(
            attachment_position_offset,
            attachment_rotation_offset,
        )

        return {
            "tracked": tracked,
            "pose": pose,
            "workspace_pose_valid": workspace_pose_valid,
            "workspace_pose": workspace_pose,
            "grip_value": grip_value,
            "trigger_value": trigger_value,
            "rotate_enable": rotate_enable,
            "close_enable": close_enable,
            "open_enable": open_enable,
            "reset_enable": reset_enable,
            "reset_robot_enable": reset_robot_enable,
            "reset_scene_enable": reset_scene_enable,
            "recenter_enable": recenter_enable,
            "teleop_enable": teleop_enable,
            "mode_switch_enable": mode_switch_enable,
            "attachment_mode": attachment_mode,
            "attachment_toggle_enable": attachment_toggle_enable,
            "attachment_adjustment_mode": attachment_adjustment_mode,
            "attachment_offset_valid": attachment_offset_valid,
            "attachment_offset": attachment_offset_pose,
            "left_stick_x": left_stick_x,
            "left_stick_y": left_stick_y,
            "right_stick_x": right_stick_x,
            "right_stick_y": right_stick_y,
            "left_grip_value": left_grip_value,
            "left_trigger_value": left_trigger_value,
            "source": source,
            "control_mode": control_mode,
            "mapping_mode": mapping_mode,
        }

    def _publish_loop(self):
        now_monotonic = time.monotonic()
        now_msg = self.get_clock().now().to_msg()

        with self._state_lock:
            age = (
                now_monotonic - self._last_packet_time_monotonic
                if self._last_packet_time_monotonic > 0.0
                else float("inf")
            )
            stale = age > self.stale_timeout_sec
            state = self._latest_state if not stale else self._neutral_state(source="stale_timeout")
            left_state = self._latest_left_state if not stale else self._neutral_state(source="left_stale_timeout")
            right_state = self._latest_right_state if not stale else self._neutral_state(source="right_stale_timeout")
            packet_count = self._packet_count_window
            byte_count = self._byte_count_window
            decode_error_count = self._decode_error_count_window
            window_dt = max(now_monotonic - self._window_start, 1e-6)

            if now_monotonic - self._last_log_time > 2.0:
                with self._client_lock:
                    sender = self._last_sender
                rx_hz = packet_count / window_dt
                self.get_logger().info(
                    f"RX {rx_hz:.1f} Hz from {sender}, bytes={byte_count}, decode_errors={decode_error_count}, "
                    f"stale={stale}, right_tracked={right_state['tracked']}, left_tracked={left_state['tracked']}, "
                    f"right teleop/rot/close/open/attach=({right_state['teleop_enable']},{right_state['rotate_enable']},{right_state['close_enable']},{right_state['open_enable']},{right_state['attachment_mode']}), "
                    f"left teleop/rot/close/open/attach=({left_state['teleop_enable']},{left_state['rotate_enable']},{left_state['close_enable']},{left_state['open_enable']},{left_state['attachment_mode']}), "
                    f"arm_reset(L/R)=({left_state['reset_robot_enable']},{right_state['reset_robot_enable']}), "
                    f"reset={state['reset_enable']}, reset_robot={state['reset_robot_enable']}, reset_scene={state['reset_scene_enable']}, recenter={state['recenter_enable']}, mode_switch={state['mode_switch_enable']}, "
                    f"control_mode={state['control_mode'] or 'unspecified'}, "
                    f"grip={state['grip_value']:.2f}, trigger={state['trigger_value']:.2f}, "
                    f"left_stick=({state['left_stick_x']:.2f},{state['left_stick_y']:.2f}), "
                    f"right_stick=({state['right_stick_x']:.2f},{state['right_stick_y']:.2f}), "
                    f"left_grip={state['left_grip_value']:.2f}, left_trigger={state['left_trigger_value']:.2f}"
                )
                self._packet_count_window = 0
                self._byte_count_window = 0
                self._decode_error_count_window = 0
                self._window_start = now_monotonic
                self._last_log_time = now_monotonic

        msg = self._make_msg(state, now_msg)
        self.pub.publish(msg)
        if self.publish_dual_arm_topics:
            self.left_pub.publish(self._make_msg(left_state, now_msg))
            self.right_pub.publish(self._make_msg(right_state, now_msg))

    def _make_msg(self, state, stamp):
        msg = ReceivedPoseStates()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.tracked = bool(state["tracked"])
        msg.pose = state["pose"]
        if hasattr(msg, "workspace_pose_valid"):
            msg.workspace_pose_valid = bool(state["workspace_pose_valid"])
        if hasattr(msg, "workspace_pose"):
            msg.workspace_pose = state["workspace_pose"]
        msg.grip_value = float(state["grip_value"])
        msg.trigger_value = float(state["trigger_value"])
        msg.rotate_enable = bool(state["rotate_enable"])
        msg.close_enable = bool(state["close_enable"])
        msg.open_enable = bool(state["open_enable"])
        msg.reset_enable = bool(state["reset_enable"])
        if hasattr(msg, "reset_robot_enable"):
            msg.reset_robot_enable = bool(state["reset_robot_enable"])
        if hasattr(msg, "reset_scene_enable"):
            msg.reset_scene_enable = bool(state["reset_scene_enable"])
        msg.recenter_enable = bool(state["recenter_enable"])
        msg.teleop_enable = bool(state["teleop_enable"])
        msg.attachment_mode = bool(state["attachment_mode"])
        msg.mode_switch_enable = bool(state["mode_switch_enable"])
        msg.left_stick_x = float(state["left_stick_x"])
        msg.left_stick_y = float(state["left_stick_y"])
        msg.right_stick_x = float(state["right_stick_x"])
        msg.right_stick_y = float(state["right_stick_y"])
        msg.left_grip_value = float(state["left_grip_value"])
        msg.left_trigger_value = float(state["left_trigger_value"])
        msg.source = str(state["source"])
        msg.control_mode = str(state["control_mode"])
        if hasattr(msg, "mapping_mode"):
            msg.mapping_mode = str(state["mapping_mode"])
        if hasattr(msg, "attachment_adjustment_mode"):
            msg.attachment_adjustment_mode = bool(state["attachment_adjustment_mode"])
        if hasattr(msg, "attachment_offset_valid"):
            msg.attachment_offset_valid = bool(state["attachment_offset_valid"])
        if hasattr(msg, "attachment_offset"):
            msg.attachment_offset = state["attachment_offset"]
        return msg

    def _parse_attachment_offset(self, position, rotation):
        offset = Pose()
        offset.orientation.w = 1.0
        valid = False

        if isinstance(position, dict):
            offset.position.x = self._as_float(position.get("x"), 0.0)
            offset.position.y = self._as_float(position.get("y"), 0.0)
            offset.position.z = self._as_float(position.get("z"), 0.0)
            valid = True

        if isinstance(rotation, dict):
            offset.orientation.x = self._as_float(rotation.get("x"), 0.0)
            offset.orientation.y = self._as_float(rotation.get("y"), 0.0)
            offset.orientation.z = self._as_float(rotation.get("z"), 0.0)
            offset.orientation.w = self._as_float(rotation.get("w"), 1.0)
            valid = True

        return valid, offset

    @staticmethod
    def _as_float(value, default):
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _as_bool(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return False

    @staticmethod
    def _as_optional_bool(value):
        if value is None:
            return None
        return QuestControllerReceiver._as_bool(value)

    @staticmethod
    def _parse_control_mode(value):
        if value is None:
            return ""
        mode = str(value).strip().lower()
        aliases = {
            "hand": "hand_pose",
            "pose": "hand_pose",
            "controller": "hand_pose",
            "controller_pose": "hand_pose",
            "game": "gamepad",
            "thumbstick": "gamepad",
        }
        mode = aliases.get(mode, mode)
        return mode if mode in ("hand_pose", "gamepad") else ""

    @staticmethod
    def _parse_mapping_mode(value):
        if value is None:
            return ""
        mode = str(value).strip().lower()
        aliases = {
            "mapping": "unity_world_delta",
            "world_mapping": "unity_world_delta",
            "world": "unity_world_delta",
            "delta": "world_delta",
            "legacy": "absolute",
        }
        mode = aliases.get(mode, mode)
        return mode if mode in ("world_delta", "absolute", "unity_world_delta", "unity_world_absolute") else ""

    @staticmethod
    def _neutral_state(source):
        pose = Pose()
        pose.orientation.w = 1.0
        workspace_pose = Pose()
        workspace_pose.orientation.w = 1.0
        attachment_offset = Pose()
        attachment_offset.orientation.w = 1.0
        return {
            "tracked": False,
            "pose": pose,
            "workspace_pose_valid": False,
            "workspace_pose": workspace_pose,
            "grip_value": 0.0,
            "trigger_value": 0.0,
            "rotate_enable": False,
            "close_enable": False,
            "open_enable": False,
            "reset_enable": False,
            "reset_robot_enable": False,
            "reset_scene_enable": False,
            "recenter_enable": False,
            "teleop_enable": False,
            "mode_switch_enable": False,
            "attachment_mode": False,
            "attachment_toggle_enable": False,
            "attachment_adjustment_mode": False,
            "attachment_offset_valid": False,
            "attachment_offset": attachment_offset,
            "left_stick_x": 0.0,
            "left_stick_y": 0.0,
            "right_stick_x": 0.0,
            "right_stick_y": 0.0,
            "left_grip_value": 0.0,
            "left_trigger_value": 0.0,
            "source": source,
            "control_mode": "",
            "mapping_mode": "",
        }

    def destroy_node(self):
        self._drop_client("shutdown")
        try:
            self.server_sock.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = QuestControllerReceiver()
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
