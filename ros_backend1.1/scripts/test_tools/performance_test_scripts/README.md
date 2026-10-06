# Performance Test Scripts

These wrappers run the same dynamic backend performance evaluation used for the
macOS thesis data:

```text
plain backend:        headless Gazebo, desktop/noVNC disabled
noVNC only:           headless Gazebo, desktop/noVNC enabled
noVNC + headed:       headed Gazebo, desktop/noVNC enabled
```

Each condition runs a synthetic hand-pose generator so the backend is measured
under dynamic Servo/Gazebo load instead of idle simulation.

## What The Test Measures

- One-second-windowed Gazebo real-time factor from ROS `/clock`.
- Docker container CPU load from `docker stats`.
- Min/mean/max/std summaries.
- Time-series CSVs and SVG plots.

## Why This Is Justifiable

This comparison is useful for this project because the backend is meant to be
portable and because Gazebo real-time factor directly affects perceived
teleoperation responsiveness. A backend that cannot keep stable simulation time
will make the robot feel slow even if the Quest controller stream is healthy.

Across different machines, this should be presented as a portability/performance
case study, not a strict operating-system benchmark. CPU model, GPU, cooling,
Docker engine, graphics driver, and VM/WSL overhead can dominate the result. It
becomes an OS benchmark only if hardware is matched or the same computer is
tested under different operating systems.

## Linux

On Ubuntu/Linux, from the repo root:

```bash
cd ros_backend1.1
./scripts/test_tools/performance_test_scripts/run_dynamic_backend_performance_linux.sh
```

Optional shorter smoke run:

```bash
DURATION=20 WARMUP=5 ./scripts/test_tools/performance_test_scripts/run_dynamic_backend_performance_linux.sh
```

## Windows

Use Windows with WSL2 Ubuntu and Docker Desktop WSL integration enabled. From a
PowerShell terminal in the repo:

```powershell
.\ros_backend1.1\scripts\test_tools\performance_test_scripts\run_dynamic_backend_performance_windows.ps1
```

Optional shorter smoke run:

```powershell
.\ros_backend1.1\scripts\test_tools\performance_test_scripts\run_dynamic_backend_performance_windows.ps1 -Duration 20 -Warmup 5
```

The Windows wrapper runs the same backend evaluator inside WSL. Native Windows
PowerShell alone is not the canonical backend environment because the lifecycle
script is Bash-based and the ROS backend runs as Linux containers.

## Result Files

By default the evaluator writes into:

```text
thesis_eval_raw/MM_DD/runtime_performance/
thesis_eval_raw/MM_DD/logs/backend_eval_runs/
```

Copy each platform's `dynamic_performance_summary.csv` into one folder and run:

```bash
python3 ros_backend1.1/scripts/test_tools/performance_test_scripts/merge_cross_platform_performance.py \
  --input macos_dynamic_performance_summary.csv \
  --input ubuntu_dynamic_performance_summary.csv \
  --input windows_dynamic_performance_summary.csv \
  --output cross_platform_performance_summary.csv
```
