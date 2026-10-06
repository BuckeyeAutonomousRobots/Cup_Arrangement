# Setup and verification

## Linux simulator

The working environment uses Docker Engine/Compose, ROS 2 Humble, Gazebo
Fortress, and a Linux NVIDIA GPU for accelerated camera rendering. Host ROS is
not required. ROS dependencies are declared in the Dockerfile and each
`src/*/package.xml`; build output is generated inside the container.

The supported target is Linux with Docker/Compose. Native macOS Gazebo and a
complete Docker Desktop simulation launch have not been validated. On a Mac,
use a suitable Linux host/VM with this same checkout, or the remote mode below.

The repository launcher defaults to local Linux hosting:

```bash
cp config/hosting.env.example .hosting.env
python3 tools/capstone.py config
python3 tools/capstone.py up --dry-run
```

Review `.hosting.env` and `ros_backend1.1/.env.example` before launch. Local
configuration is ignored by Git. Defaults bind
published services to loopback, use a separate Compose project/container and DDS
domain, and leave the legacy backend pause option disabled. If an existing
simulation uses these names or ports, change this clone's settings first.

The supplied active profile points to `baseline_v1`. Preserve it for the current
cup task; selecting an older task can replace the scene and restart processes.

```bash
python3 tools/capstone.py build-image
python3 tools/capstone.py build-workspace
python3 tools/capstone.py up
python3 tools/capstone.py status
```

The lifecycle helper regenerates the world from checked-in scene profiles. If
using a clean build is necessary, the helper also exposes `clean_build_ws`.
These commands start a simulator; do not run them against a shared machine
without checking existing workloads and control ownership.

For NVIDIA rendering, install a compatible driver and NVIDIA Container Toolkit
using the platform's normal process, then set `CAPSTONE_GPU=1` in `.hosting.env`.
The GPU Compose override requests the GPU; the viewer remains independently
controlled by `CAPSTONE_VIEWER` and defaults off.
No driver installer or machine configuration is included in this export.

The exported container uses the generic user `robot` and workspace `/workspace`.
All bind mounts are repository-relative. No personal home directory or existing
SSH alias is required. The original research machine's layout was not changed.

## Tests

The host-only source validator checks Python syntax, shell syntax, XML files,
and common unwanted artifact names. It does not launch any project script.

Once a simulator container has been deliberately built, the backend's tests can
be run inside it:

```bash
docker exec bar_cup_arrangement_backend bash -lc \
  'source /workspace/install/setup.bash; cd /workspace; python3 -m pytest -q -p no:cacheprovider tests'
```

Substitute the configured container name when different. Some tests depend on
the ROS/container environment. A successful syntax check is not a simulator or
closed-loop policy validation.

Launcher configuration tests run without ROS or Docker:

```bash
python3 -m unittest discover -s tools -p 'test_*.py' -v
```

## Offline learning environment

The observed research environment uses Python 3.12.3 and PyTorch 2.7.1 with CUDA
12.6. Primary package versions are recorded in
`requirements/training-observed.txt`. This is an environment manifest, not a
complete cross-platform dependency lock or a guarantee that a fresh resolver
will reproduce the environment.

Create a separate environment at `models/act_v1/.venv` before using the historical
launchers. Install a compatible PyTorch/torchvision pair and the listed research
dependencies using the appropriate platform indexes. The source export does
not install packages, download weights or allocate GPUs automatically.

With the dependencies available, contract tests are independent of ROS:

```bash
cd models/policy100
../act_v1/.venv/bin/python -m unittest -v test_position_act_contract
```

Training and recorded-trial launchers require local manifests, caches and prior
run artifacts that are not public in this repository. Do not create dummy data,
remove their preflight checks, or relaunch a run with an existing `.claim` file.
