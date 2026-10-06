# teleop_bridge Package Map

`teleop_bridge` is the umbrella ROS package for the teleoperation data flow. The ROS package name stays stable so launch/lifecycle scripts and installed dependencies do not churn, but the Python modules are split by functional stage.

## Data Flow

```text
Quest/Unity controller packets
  -> receiver/quest_controller_receiver
  -> teleop_bridge.mapping.hand_pose_mapper
  -> teleop_bridge.servo_bridge.servo_command_bridge
  -> MoveIt Servo / Gazebo robot arm
```

Gripper commands:

```text
TargetTwistStates gripper fields
  -> teleop_bridge.gripper_control.coupled_gripper_controller
  -> Hand-E position controllers
```

Task/object sync:

```text
Gazebo dynamic poses + generated SDF task layout
  -> teleop_bridge.task_sync.task_pose_sync_publisher
  -> /unity_sync/<object>_pose
  -> Unity ROS-TCP sync manager
```

Haptics:

```text
Gazebo contacts / proximity / Servo collision state
  -> teleop_bridge.haptic_feedback.contact_haptic_publisher
  -> Unity Quest haptic controller
```

## Module Folders

| Folder | Function |
| --- | --- |
| `mapping/` | Converts received controller/headset state into target EE twist/gripper/reset state. |
| `servo_bridge/` | Converts target state into MoveIt Servo twist commands; includes keyboard override. |
| `gripper_control/` | Converts target gripper state into Hand-E finger commands; includes coupled gripper controller. |
| `reset_control/` | Handles robot home/reset and Gazebo task-object reset. |
| `task_sync/` | Publishes Gazebo task object poses to Unity and manages runtime task manifests. |
| `haptic_feedback/` | Publishes contact/collision haptic events from Gazebo/Servo state. |
| `test_tools/` | Fake/debug hand publishers, Servo response sampler, joint test helper. |
| `common/` | Shared utilities such as generated SDF scene-layout parsing. |
| `config/` | Single-arm and dual-arm tuning YAML files. |

## Preferred Console Commands

Use these names for new docs/scripts:

```bash
ros2 run teleop_bridge hand_pose_mapper
ros2 run teleop_bridge servo_command_bridge
ros2 run teleop_bridge keyboard_servo_override
ros2 run teleop_bridge coupled_gripper_controller
ros2 run teleop_bridge reset_manager
ros2 run teleop_bridge task_pose_sync_publisher
ros2 run teleop_bridge runtime_task_manager
ros2 run teleop_bridge contact_haptic_publisher
ros2 run teleop_bridge debug_hand_generator
ros2 run teleop_bridge servo_response_sampler
```

## Compatibility Aliases

Existing lifecycle commands still work because `setup.py` keeps aliases such as:

```bash
ros2 run teleop_bridge received_pose_to_target_twist
ros2 run teleop_bridge target_twist_to_servo_cmd
ros2 run teleop_bridge coupled_hande_gripper_controller
ros2 run teleop_bridge cube_pose_sync_publisher
ros2 run teleop_bridge task_manager
ros2 run teleop_bridge dual_debug_hand_generator
```

The compatibility aliases can be removed later after docs, scripts, and saved logs fully migrate to the preferred names.
