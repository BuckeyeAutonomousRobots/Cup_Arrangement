import json
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import rclpy
import yaml
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger


class TaskManager(Node):
    """Runtime task selector for profile-backed Gazebo scenes.

    The task profile remains the source of truth. On selection this node:
    1. Generates the world SDF and Unity-compatible task manifest.
    2. Removes old Sync_* task models from the live Gazebo world.
    3. Spawns the new Sync_* task models through Ignition transport services.
    4. Publishes the manifest so Unity can rebuild only its task group.
    """

    def __init__(self):
        super().__init__("runtime_task_manager")

        self.declare_parameter("repo_root", "/workspace")
        self.declare_parameter("default_scene_profile", "profiles/scenes/dual_arm_tabletop/scene.yaml")
        self.declare_parameter("active_scene_profile_file", "profiles/active_scene_profile.txt")
        self.declare_parameter("world_output", "simulation/worlds/ur_hande_dual_arm_tabletop.sdf")
        self.declare_parameter("generator_path", "simulation/tools/generate_dual_arm_world_from_profiles.py")
        self.declare_parameter("unity_manifest_output", "/tmp/task_manager_active_task.json")
        self.declare_parameter("world_name", "ur_hande_dual_arm_tabletop")
        self.declare_parameter("status_topic", "/task_manager/status")
        self.declare_parameter("manifest_topic", "/task_manager/active_task_manifest")
        self.declare_parameter("select_topic", "/task_manager/select_task")
        self.declare_parameter("hot_swap_on_select", True)
        self.declare_parameter("gazebo_service_timeout_ms", 5000)
        self.declare_parameter("model_tmp_dir", "/tmp/task_manager_models")
        self.declare_parameter("publish_rate_hz", 1.0)

        self.repo_root = Path(str(self.get_parameter("repo_root").value)).resolve()
        self.default_scene_profile = self._resolve_repo_path(
            str(self.get_parameter("default_scene_profile").value)
        )
        self.active_scene_profile_file = self._resolve_repo_path(
            str(self.get_parameter("active_scene_profile_file").value)
        )
        self.world_output = self._resolve_repo_path(str(self.get_parameter("world_output").value))
        self.generator_path = self._resolve_repo_path(str(self.get_parameter("generator_path").value))
        self.unity_manifest_output = Path(str(self.get_parameter("unity_manifest_output").value)).resolve()
        self.world_name = str(self.get_parameter("world_name").value)
        status_topic = str(self.get_parameter("status_topic").value)
        manifest_topic = str(self.get_parameter("manifest_topic").value)
        select_topic = str(self.get_parameter("select_topic").value)
        self.hot_swap_on_select = bool(self.get_parameter("hot_swap_on_select").value)
        self.gazebo_service_timeout_ms = int(self.get_parameter("gazebo_service_timeout_ms").value)
        self.model_tmp_dir = Path(str(self.get_parameter("model_tmp_dir").value)).resolve()
        publish_rate_hz = max(0.2, float(self.get_parameter("publish_rate_hz").value))

        self.status_pub = self.create_publisher(String, status_topic, 10)
        self.manifest_pub = self.create_publisher(String, manifest_topic, 10)
        self.create_subscription(String, select_topic, self._on_select_task_msg, 10)
        self.create_service(Trigger, "/task_manager/list_tasks", self._on_list_tasks)
        self.create_service(Trigger, "/task_manager/current_task", self._on_current_task)
        self.create_service(Trigger, "/task_manager/regenerate_world", self._on_regenerate_world)

        self.task_scene_map: dict[str, Path] = {}
        self.last_result = "started"
        self.last_error = ""
        self.last_switch_time = 0.0
        self.active_manifest: dict[str, Any] = {}
        self.current_task_model_names: set[str] = set()
        self.active_scene_profile = self._read_active_scene_profile()
        self._discover_task_scenes()
        self.active_task = self._task_name_for_scene(self.active_scene_profile) or "unknown"
        self._refresh_outputs_for_active_scene()

        self.create_timer(1.0 / publish_rate_hz, self._publish_state)
        self.get_logger().info(
            "TaskManager started: "
            f"active_task={self.active_task}, tasks={sorted(self.task_scene_map)}, "
            f"hot_swap_on_select={self.hot_swap_on_select}, select_topic={select_topic}"
        )

    def _resolve_repo_path(self, raw: str) -> Path:
        path = Path(raw)
        if path.is_absolute():
            return path
        return (self.repo_root / path).resolve()

    @staticmethod
    def _read_yaml(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}

    def _read_active_scene_profile(self) -> Path:
        try:
            raw = self.active_scene_profile_file.read_text(encoding="utf-8").strip()
        except OSError:
            raw = ""
        if raw:
            return self._resolve_repo_path(raw)
        return self.default_scene_profile

    def _write_active_scene_profile(self, scene_profile: Path):
        self.active_scene_profile_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            relative = scene_profile.resolve().relative_to(self.repo_root)
            text = str(relative)
        except ValueError:
            text = str(scene_profile.resolve())
        self.active_scene_profile_file.write_text(text + "\n", encoding="utf-8")

    def _resolve_profile_path(self, scene_profile_path: Path, raw_path: str) -> Path:
        path = Path(raw_path)
        if path.is_absolute():
            return path
        return (scene_profile_path.parent / path).resolve()

    def _discover_task_scenes(self):
        self.task_scene_map.clear()
        scene_root = self.repo_root / "profiles" / "scenes"
        for scene_path in sorted(scene_root.glob("*/scene.yaml")):
            task_name = self._task_name_for_scene(scene_path)
            if not task_name:
                continue
            self.task_scene_map.setdefault(task_name, scene_path.resolve())

    def _task_name_for_scene(self, scene_profile_path: Path) -> str:
        try:
            scene = self._read_yaml(scene_profile_path)
            task_profile = self._resolve_profile_path(scene_profile_path, str(scene["task_profile"]))
            task = self._read_yaml(task_profile)
            return str(task.get("task_profile") or task_profile.parent.name)
        except Exception:
            return ""

    def _refresh_outputs_for_active_scene(self):
        ok, manifest = self._generate_outputs(self.active_scene_profile)
        if ok:
            self.active_manifest = manifest
            self.current_task_model_names = self._task_model_names_from_sdf(self.world_output)
        else:
            self.active_manifest = self._fallback_manifest(self.active_scene_profile)
            self.current_task_model_names = self._task_model_names_from_sdf(self.world_output)

    def _fallback_manifest(self, scene_profile: Path) -> dict[str, Any]:
        try:
            scene = self._read_yaml(scene_profile)
            task_profile = self._resolve_profile_path(scene_profile, str(scene["task_profile"]))
            task = self._read_yaml(task_profile)
            return {
                "taskProfile": str(task.get("task_profile", self.active_task)),
                "taskGroupName": str(task.get("task_group_name", "TaskGroup_Main")),
                "taskGroupLocalPosition": {"x": 0.0, "y": 0.0, "z": 0.0},
                "taskGroupLocalEuler": {"x": 0.0, "y": 0.0, "z": 0.0},
                "objects": [],
                "error": self.last_error,
            }
        except Exception as exc:
            return {"taskProfile": self.active_task, "objects": [], "error": str(exc)}

    def _status(self) -> dict[str, Any]:
        return {
            "active_task": self.active_task,
            "available_tasks": sorted(self.task_scene_map.keys()),
            "scene_profile": str(self.active_scene_profile),
            "requires_restart_after_select": False,
            "hot_swap_supported": True,
            "hot_swap_on_select": self.hot_swap_on_select,
            "tracked_models": sorted(self.current_task_model_names),
            "last_result": self.last_result,
            "last_error": self.last_error,
            "last_switch_time": self.last_switch_time,
            "unix_time": time.time(),
        }

    def _publish_json(self, publisher, payload: dict[str, Any]):
        msg = String()
        msg.data = json.dumps(payload, sort_keys=True)
        publisher.publish(msg)

    def _publish_state(self):
        self._publish_json(self.status_pub, self._status())
        manifest = dict(self.active_manifest or {})
        manifest["runtimeSwitch"] = {
            "activeTask": self.active_task,
            "hotSwapSupported": True,
            "lastResult": self.last_result,
            "lastError": self.last_error,
        }
        self._publish_json(self.manifest_pub, manifest)

    def _on_select_task_msg(self, msg: String):
        task_name = str(msg.data or "").strip()
        self._select_task(task_name)

    def _select_task(self, task_name: str) -> bool:
        self._discover_task_scenes()
        scene_profile = self.task_scene_map.get(task_name)
        if scene_profile is None:
            self.last_error = f"Unknown task '{task_name}'. Available: {sorted(self.task_scene_map)}"
            self.last_result = "select failed"
            self.get_logger().warn(self.last_error)
            self._publish_state()
            return False

        old_models = self._task_model_elements_from_sdf(self.world_output)
        old_model_names = set(old_models.keys())
        candidate_world = Path("/tmp/task_manager_candidate_world.sdf")
        candidate_manifest = Path("/tmp/task_manager_candidate_task.json")
        ok, manifest = self._generate_outputs(
            scene_profile,
            world_output=candidate_world,
            unity_manifest_output=candidate_manifest,
        )
        if not ok:
            self.last_result = f"select {task_name} failed during generation"
            self._publish_state()
            return False

        new_models = self._task_model_elements_from_sdf(candidate_world)
        new_model_names = set(new_models.keys())
        if self.hot_swap_on_select:
            ok = self._replace_gazebo_task_models(old_models, new_models)
            if not ok:
                self.last_result = f"select {task_name} failed during Gazebo hot-swap"
                self._publish_state()
                return False

        self._apply_candidate_outputs(candidate_world, candidate_manifest)
        self.active_task = task_name
        self.active_scene_profile = scene_profile
        self.active_manifest = manifest
        self.current_task_model_names = new_model_names
        self._write_active_scene_profile(scene_profile)
        self.last_error = ""
        self.last_result = (
            f"selected {task_name}; hot-swapped {len(old_model_names)} old models "
            f"for {len(new_model_names)} new models"
        )
        self.last_switch_time = time.time()
        self.get_logger().info(self.last_result)
        self._publish_state()
        return True

    def _generate_outputs(
        self,
        scene_profile: Path,
        world_output: Path | None = None,
        unity_manifest_output: Path | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        world_output = world_output or self.world_output
        unity_manifest_output = unity_manifest_output or self.unity_manifest_output
        world_output.parent.mkdir(parents=True, exist_ok=True)
        unity_manifest_output.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "python3",
            str(self.generator_path),
            "--repo-root",
            str(self.repo_root),
            "--scene-profile",
            str(scene_profile),
            "--output",
            str(world_output),
            "--unity-task-json",
            str(unity_manifest_output),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10.0, check=False)
        except Exception as exc:
            self.last_error = f"world generation failed to run: {exc}"
            self.get_logger().warn(self.last_error)
            return False, {}

        if result.returncode != 0:
            stderr = result.stderr.strip() or result.stdout.strip() or "unknown error"
            self.last_error = f"world generation failed: {stderr}"
            self.get_logger().warn(self.last_error)
            return False, {}

        try:
            manifest = json.loads(unity_manifest_output.read_text(encoding="utf-8"))
        except Exception as exc:
            self.last_error = f"generated Unity task manifest could not be read: {exc}"
            self.get_logger().warn(self.last_error)
            return False, {}

        self.last_error = ""
        return True, manifest

    def _apply_candidate_outputs(self, candidate_world: Path, candidate_manifest: Path):
        self.world_output.parent.mkdir(parents=True, exist_ok=True)
        self.unity_manifest_output.parent.mkdir(parents=True, exist_ok=True)
        self.world_output.write_bytes(candidate_world.read_bytes())
        self.unity_manifest_output.write_bytes(candidate_manifest.read_bytes())

    def _task_model_names_from_sdf(self, sdf_path: Path) -> set[str]:
        return set(self._task_model_elements_from_sdf(sdf_path).keys())

    def _task_model_elements_from_sdf(self, sdf_path: Path) -> dict[str, ET.Element]:
        if not sdf_path.exists():
            return {}
        try:
            root = ET.parse(sdf_path).getroot()
        except Exception as exc:
            self.get_logger().warn(f"Could not parse task models from {sdf_path}: {exc}")
            return {}
        world = root.find("world") or root.find(".//world")
        if world is None:
            return {}
        models: dict[str, ET.Element] = {}
        for model in world.findall("model"):
            name = str(model.get("name", "")).strip()
            if name.startswith("Sync_"):
                models[name] = model
        return models

    def _replace_gazebo_task_models(
        self,
        old_models: dict[str, ET.Element],
        new_models: dict[str, ET.Element],
    ) -> bool:
        old_model_names = set(old_models.keys())
        if not old_model_names and not new_models:
            return True

        removed_failures = []
        for name in sorted(old_model_names):
            if not self._remove_gazebo_model(name):
                removed_failures.append(name)

        spawned_failures = []
        for name, model in sorted(new_models.items()):
            model_file = self._write_model_sdf(name, model)
            if model_file is None or not self._spawn_gazebo_model(name, model_file):
                spawned_failures.append(name)

        if removed_failures or spawned_failures:
            self.get_logger().warn("Gazebo hot-swap failed; attempting to restore previous task models.")
            for name in sorted(new_models.keys()):
                self._remove_gazebo_model(name)
            for name, model in sorted(old_models.items()):
                model_file = self._write_model_sdf(name, model)
                if model_file is not None:
                    self._spawn_gazebo_model(name, model_file)
            self.last_error = (
                f"Gazebo task hot-swap incomplete; remove_failed={removed_failures}, "
                f"spawn_failed={spawned_failures}"
            )
            self.get_logger().warn(self.last_error)
            return False
        return True

    def _write_model_sdf(self, name: str, model: ET.Element) -> Path | None:
        self.model_tmp_dir.mkdir(parents=True, exist_ok=True)
        path = self.model_tmp_dir / f"{name}.sdf"
        try:
            root = ET.Element("sdf", attrib={"version": "1.8"})
            root.append(ET.fromstring(ET.tostring(model, encoding="unicode")))
            ET.indent(root)
            ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
            return path
        except Exception as exc:
            self.get_logger().warn(f"Could not write model SDF for {name}: {exc}")
            return None

    def _remove_gazebo_model(self, name: str) -> bool:
        request = f'name: "{name}"\ntype: 2'
        return self._call_ign_service(
            service=f"/world/{self.world_name}/remove",
            reqtype="ignition.msgs.Entity",
            request=request,
            action=f"remove {name}",
            tolerate_false=True,
        )

    def _spawn_gazebo_model(self, name: str, model_file: Path) -> bool:
        request = f'sdf_filename: "{model_file}"\nname: "{name}"\nallow_renaming: false'
        return self._call_ign_service(
            service=f"/world/{self.world_name}/create",
            reqtype="ignition.msgs.EntityFactory",
            request=request,
            action=f"spawn {name}",
            tolerate_false=False,
        )

    def _call_ign_service(
        self,
        service: str,
        reqtype: str,
        request: str,
        action: str,
        tolerate_false: bool,
    ) -> bool:
        cmd = [
            "ign",
            "service",
            "-s",
            service,
            "--reqtype",
            reqtype,
            "--reptype",
            "ignition.msgs.Boolean",
            "--timeout",
            str(self.gazebo_service_timeout_ms),
            "--req",
            request,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=(self.gazebo_service_timeout_ms / 1000.0) + 2.0, check=False)
        except Exception as exc:
            self.get_logger().warn(f"Gazebo {action} service call failed to run: {exc}")
            return False

        output = (result.stdout + "\n" + result.stderr).strip()
        if result.returncode != 0:
            self.get_logger().warn(f"Gazebo {action} service call failed: {output}")
            return False
        lowered = output.lower()
        if "no service providers" in lowered or "timed out" in lowered or "timeout" in lowered:
            self.get_logger().warn(f"Gazebo {action} service unavailable: {output}")
            return False
        if "data: false" in output and not tolerate_false:
            self.get_logger().warn(f"Gazebo {action} returned false: {output}")
            return False
        return True

    def _on_list_tasks(self, _request, response):
        self._discover_task_scenes()
        response.success = True
        response.message = json.dumps({"available_tasks": sorted(self.task_scene_map.keys())})
        return response

    def _on_current_task(self, _request, response):
        response.success = True
        response.message = json.dumps(self._status())
        return response

    def _on_regenerate_world(self, _request, response):
        ok, manifest = self._generate_outputs(self.active_scene_profile)
        if ok:
            self.active_manifest = manifest
            self.current_task_model_names = self._task_model_names_from_sdf(self.world_output)
            self.last_result = f"regenerated world for {self.active_task}; no restart required for manifest reload"
        response.success = ok
        response.message = self.last_result if ok else self.last_error
        self._publish_state()
        return response


def main(args=None):
    rclpy.init(args=args)
    node = TaskManager()
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
