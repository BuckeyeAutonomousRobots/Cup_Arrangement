# Licenses and third-party notices

No new repository-wide license has been chosen. Public availability does not
replace the terms attached to individual files and dependencies. Several
project-authored ROS package manifests still contain `TODO` license fields;
they have not been silently assigned a license during export.

| Component | Existing notice |
|---|---|
| `ros_backend1.1/src/robotiq_hande_description` | Apache-2.0; retain its `LICENSE`, upstream README, citation and Robotiq/model credits |
| `ros_backend1.1/src/ROS-TCP-Endpoint` | Apache-2.0; retain its `LICENSE` and Unity Technologies notices |
| `ur_hande_description`, `ur_moveit_config` | Existing package manifests declare Apache-2.0; source notices are preserved |
| ROS 2, MoveIt, Gazebo, Universal Robots packages | External dependencies installed through the ROS distribution; their own licenses apply |
| PyTorch, torchvision, LeRobot and other learning dependencies | External dependencies; see their distributions for licenses |

The Hand-E vendor README attributes original model files to Robotiq and URDF
work to the named upstream authors. Those credits remain with the component.
Vendor CAD source files and unverified generated scene textures were omitted.

This source derives from `Noah727/Docker_Teleop`; see `docs/PROVENANCE.md`.
