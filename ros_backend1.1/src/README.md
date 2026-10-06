# ROS Workspace Package Map

`src/` is a ROS 2 colcon workspace. Each direct child folder is one buildable ROS package, so these package folders should not be renamed casually unless every launch file, lifecycle script, xacro include, package dependency, and install rule is migrated together.

## Runtime Packages

| Package | Role |
| --- | --- |
| `receiver` | TCP receiver for Quest/Unity controller packets. Publishes `ReceivedPoseStates` for left/right arms. |
| `teleop_bridge_msgs` | Custom ROS 2 messages shared between receiver, mapper, Servo bridge, gripper, reset, and haptic nodes. |
| `teleop_bridge` | Umbrella runtime package. Internal Python folders are organized by stage: `mapping`, `servo_bridge`, `gripper_control`, `reset_control`, `task_sync`, `haptic_feedback`, `test_tools`, and `common`. |
| `ROS-TCP-Endpoint` | Unity Robotics ROS-TCP endpoint package. Required for Unity ROS-TCP Connector communication. |

## Robot And Planning Packages

| Package | Role |
| --- | --- |
| `ur_hande_description` | Combined UR5e + Robotiq Hand-E xacro used to spawn the simulated robots. |
| `robotiq_hande_description` | Upstream-style Robotiq Hand-E meshes/xacros. Required because `ur_hande_description` includes it and mesh paths use `package://robotiq_hande_description/...`. |
| `ur_moveit_config` | MoveIt/SRDF/joint-limit configuration and self-collision inspection support. |
| `servo_test_config` | MoveIt Servo launch/config wrappers for single-arm and dual-arm Gazebo use. |

## What Not To Put Here

Do not place runtime recordings, rosbag output, temporary logs, screenshots, or generated caches under `src/`.
Use:

```text
ros_backend1.1/runtime/
ros_backend1.1/eval_results/
/tmp/*.log inside the container
```

## Why `robotiq_hande_description` Stays Separate

The gripper package looks redundant next to `ur_hande_description`, but it is a dependency, not a duplicate. The combined robot xacro imports:

```xml
$(find robotiq_hande_description)/urdf/robotiq_hande_gripper.xacro
```

and the Hand-E meshes are resolved from:

```text
package://robotiq_hande_description/meshes/...
```

Deleting or renaming that package will break robot generation in Gazebo and MoveIt.
