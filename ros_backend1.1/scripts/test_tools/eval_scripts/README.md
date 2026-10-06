# Backend Evaluation Scripts

No-headset evaluation helpers for `ros_backend1.1`. These scripts are meant to generate repeatable CSV/JSON evidence for thesis evaluation without wearing the Quest headset.

Run from repo root or from `ros_backend1.1` unless noted. Output defaults to:

```bash
ros_backend1.1/eval_results/<timestamp>_<test_name>/
```

Set a different output root with `--output-root` or:

```bash
export EVAL_RESULTS_DIR=/path/to/results
```

## Prerequisites

Start the backend first for live ROS/Gazebo tests:

```bash
cd ros_backend1.1
./scripts/backend11_lifecycle.sh bringup_dual
./scripts/backend11_lifecycle.sh status
```

Most scripts assume:

```bash
CONTAINER=motion_planner_11
ROS_ENV='source /opt/ros/humble/setup.bash && source /workspace/install/setup.bash'
```

## Scripts

| Script | Purpose | Headset needed | Notes |
| --- | --- | --- | --- |
| `01_backend_rtf_cpu_monitor.py` | Monitor Gazebo `/clock` real-time factor plus Docker CPU/memory | No | Good baseline performance monitor. |
| `02_headless_vs_headed_comparison.py` | Run headless vs headed Gazebo and compare RTF/CPU | No | noVNC-connected condition is manual: run headed while browser is connected. |
| `03_ros_topic_rate_audit.py` | Measure message rate for important ROS topics | No | Use `--topics` for a custom topic list. |
| `04_synthetic_hand_receiver_test.py` | Debug fake hand-pose input and mapper behavior without the Quest app | No | Debug-only. Not a thesis-facing controller evaluation. Stops `quest_controller_receiver`; restart with `start_receiver`. |
| `05_servo_step_sine_response_test.py` | Fake hand motion plus Servo response sampler | No | Uses lifecycle `debug_servo_motion`. |
| `06_mapping_axis_sign_sanity_test.py` | Run synthetic X/Y/Z motion cases and save response summaries | No | Automated sanity check, not a substitute for human MR feel. |
| `07_gripper_timing_symmetry_test.py` | Directly command gripper open/close and record finger joints | No | Tests controller/joint symmetry independent of Quest input. |
| `08_reset_reliability_test.py` | Trigger reset and compare sync poses against generated SDF | No | Use `--disturb` to move objects before reset. |
| `09_task_profile_sdf_unity_consistency_test.py` | Compare task YAML, generated SDF, and Unity `active_task.json` | No | Offline consistency check; requires PyYAML. |
| `10_unity_editor_sync_smoke.py` | File-based smoke check for saved Unity scene/task objects | No | MCP live editor follow-up is listed in script output. |
| `11_haptic_publisher_topic_logic_test.py` | Record haptic output topics and summarize pulses/amplitudes | No | Best used while running gripper/contact test. |
| `12_cross_platform_backend_bringup_check.py` | Collect platform, Docker, ADB, lifecycle status report | No | Use on macOS/Ubuntu/Windows-WSL for portability evidence. |
| `13_dynamic_novnc_headed_performance_test.py` | Compare plain backend, noVNC, and headed Gazebo under dynamic fake-hand load | No | Produces RTF/CPU summary and plots. |
| `14_headset_backend_latency_trace.py` | Trace live Quest input through received pose, target twist, Servo input, and joint velocity command topics | Yes | Measures backend pipeline latency after receiver publication, not optical MR latency. |
| `15_mr_sync_visual_latency_trace.py` | Trace Gazebo object/EE poses against Unity-displayed MR visual poses | Yes | Requires the Quest build with `MREvaluationTracePublisher`; outputs sync error and visual latency CSVs. |
| `16_recording_fps_sync_latency_trace.py` | Wait for Unity camera recording, then log Quest FPS samples plus MR sync/visual latency until 15 seconds after recording stops | Yes | Requires the Quest build with `RecordingPerformanceTraceLogger`; use for recording-induced lag tests. |

## Common Runs

Performance monitor:

```bash
cd ros_backend1.1
python3 scripts/test_tools/eval_scripts/01_backend_rtf_cpu_monitor.py --duration 60 --interval 1
```

Headless/headed comparison:

```bash
cd ros_backend1.1
python3 scripts/test_tools/eval_scripts/02_headless_vs_headed_comparison.py --duration 60 --warmup 8
```

Topic-rate audit:

```bash
cd ros_backend1.1
python3 scripts/test_tools/eval_scripts/03_ros_topic_rate_audit.py --duration 20
```

Backend load and Servo response:

```bash
cd ros_backend1.1
python3 scripts/test_tools/eval_scripts/05_servo_step_sine_response_test.py --arm both --pattern circle_xy --duration 12
./scripts/backend11_lifecycle.sh start_receiver
```

Gripper timing:

```bash
cd ros_backend1.1
python3 scripts/test_tools/eval_scripts/07_gripper_timing_symmetry_test.py --cycles 4 --hold 1.5
```

Reset reliability after manually or programmatically disturbing objects:

```bash
cd ros_backend1.1
python3 scripts/test_tools/eval_scripts/08_reset_reliability_test.py --disturb --settle 4
```

Task profile consistency:

```bash
cd ros_backend1.1
python3 scripts/test_tools/eval_scripts/09_task_profile_sdf_unity_consistency_test.py
python3 scripts/test_tools/eval_scripts/10_unity_editor_sync_smoke.py
```

Haptics monitor:

```bash
cd ros_backend1.1
python3 scripts/test_tools/eval_scripts/11_haptic_publisher_topic_logic_test.py --duration 20
```

Cross-platform report:

```bash
cd ros_backend1.1
python3 scripts/test_tools/eval_scripts/12_cross_platform_backend_bringup_check.py
```

Headset backend latency trace:

```bash
cd ros_backend1.1
python3 scripts/test_tools/eval_scripts/14_headset_backend_latency_trace.py --duration 30 --output-root ../thesis_eval_raw/06_11/latency_rates/headset_backend_latency
```

During the latency trace, wear the Quest, hold grip to engage teleop, keep still for about two seconds, then make five short sharp controller movements per arm. The script records arrival-time latency from `/left_arm/received_pose_states` or `/right_arm/received_pose_states` to target twist, Servo input, and joint velocity command topics.

MR visual sync/alignment trace:

```bash
cd ros_backend1.1
python3 scripts/test_tools/eval_scripts/15_mr_sync_visual_latency_trace.py --duration 30 --output-root ../thesis_eval_raw/06_11/sync_precision/mr_sync_trace
```

During the MR sync trace, wear the Quest with an app build made after `MREvaluationTracePublisher` was added, keep the red cube and grippers visible, then make a few short arm motions and one small red-cube pickup or nudge if possible. The script records:

- `pose_alignment_samples.csv`: per-sample Gazebo-to-Unity visual position/orientation error for `Sync_RedCube` and any available arm EE visual topics.
- `pose_alignment_summary.csv`: mean/min/max sync error by object/EE pair.
- `motion_events.csv`: motion-onset latency from hand input to simulated EE motion and from simulated EE/object motion to Unity visual motion.
- `pose_samples.csv`: raw sampled poses for debugging plots.
- `topic_rates.csv`: observed rates for the traced streams.

For EE alignment, the script loads `profiles/robots/ur5e_hande_dual/robot.yaml` inside the container and applies each arm's `spawn_pose_xyz_rpy` so backend base-frame EE poses are compared in the same workspace/world frame as the Unity visual poses.

Recording FPS plus sync-latency trace:

```bash
cd ros_backend1.1
python3 scripts/test_tools/eval_scripts/16_recording_fps_sync_latency_trace.py --max-wait-sec 120 --max-duration-sec 240 --post-recording-tail-sec 15 --output-root ../thesis_eval_raw/06_11/latency_rates/recording_fps_sync_trace
```

Procedure:

- Start the script before pressing the camera recording button in the headset.
- In the Quest app, start recording from the control panel or left-controller X.
- Move one or both arms for 15-30 seconds.
- Stop recording.
- Keep the app running until the script exits automatically after the 15-second post-recording tail.

The script records:

- `recording_fps_samples.csv`: Unity-published FPS/windowed frame-time samples during recording and the post-recording tail.
- `recording_events.csv`: recording start/stop/state events and the Quest-local log path.
- `pose_alignment_samples.csv` and `pose_alignment_summary.csv`: Gazebo authoritative pose versus Unity/MR visual pose error during the recording window.
- `motion_events.csv`: hand-to-sim-EE and sim/object-to-MR-visual motion onset latency during the recording window.
- `topic_rates.csv`: rates observed during the recording window.

Unity also writes a local CSV on the Quest under:

```text
Application.persistentDataPath/RecordingPerformanceLogs/
```

Retrieve it with `adb pull` if you want the raw Quest-side FPS log in addition to the ROS-side trace output.

## Unity MCP Smoke Check

`10_unity_editor_sync_smoke.py` checks the saved Unity scene YAML. It cannot call Codex MCP by itself because MCP is an agent-side tool, not a normal ROS/backend command-line API.

When Unity MCP is available, do this live follow-up from Codex:

1. Read active scene and console.
2. Execute menu item: `Tools/Gazebo Replica/Rebuild Dual Arm Workspace In Active Scene`.
3. Inspect hierarchy for `GazeboWorkspace/TaskGroups/TaskGroup_Main/GameObjects_sync`.
4. Read console again.
5. Save the scene only if the rebuild is correct.

## Notes

- Scripts `04`, `05`, and `06` stop the live Quest receiver while synthetic input is active. Restart live headset input with:

```bash
cd ros_backend1.1
./scripts/backend11_lifecycle.sh start_receiver
```

- `14_headset_backend_latency_trace.py` does not use Quest/ROS clock synchronization. It measures backend pipeline latency with one host-side subscriber clock. For true visible end-to-end latency, pair it with a Quest screen recording or external video and count frames between visible controller motion and visible robot motion.
- `16_recording_fps_sync_latency_trace.py` makes the old manual FPS/recording-lag table mostly optional. Keep short video clips as qualitative evidence, but use the generated CSVs as the primary numeric evidence when available.
- These scripts are evaluation probes, not replacements for user studies. For headset-only tasks such as perceived latency, task completion time, or MR alignment, still collect headset demos and manually recorded results.
