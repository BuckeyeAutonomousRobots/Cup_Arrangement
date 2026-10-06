# Run on a remote Linux machine

Use a Linux workstation or server to build and run the simulator while operating
the launcher from your own computer. The remote mode runs the same Docker
lifecycle as the [local setup](../../README.md); it does not provision a server
or copy source files.

## 1. Prepare the Linux host

Use an authorized x86-64 Ubuntu 22.04/24.04 host with capacity for the build and
simulation. Install Git, Python 3, PyYAML, Docker Engine and Compose as described
in the [Linux prerequisites](../../README.md#linux-prerequisites). For NVIDIA
rendering, configure the driver and Container Toolkit first. On a shared host,
confirm the GPU and compute window are available before starting.

In a **terminal on the remote Linux host**:

```bash
mkdir -p ~/src
cd ~/src
git clone https://github.com/BuckeyeAutonomousRobots/Cup_Arrangement.git
cd Cup_Arrangement
cp ros_backend1.1/.env.example ros_backend1.1/.env
sed -i 's/^ENABLE_GRIPPER_CAMERA=.*/ENABLE_GRIPPER_CAMERA=0/' ros_backend1.1/.env
docker info
docker compose version
python3 -c 'import yaml; print("PyYAML available")'
pwd
git rev-parse HEAD
```

Record the absolute checkout path and commit. The SSH account must be able to
use Docker without an interactive privilege prompt. The initial configuration
uses CPU/headless operation with the camera disabled. Keep the active
`baseline_v1` scene profile for the cup task.

## 2. Configure your local launcher

Use a Linux/macOS terminal, or **Ubuntu under WSL 2** on Windows, with Git,
Python 3 and an OpenSSH client. Clone the repository locally too, and keep both
checkouts on the same commit. Local Docker is not required for remote mode.
Verify SSH access normally before using the launcher:

```bash
ssh user@YOUR_HOST 'docker info && docker compose version'
```

Replace `user@YOUR_HOST` with your account and hostname, including in the examples
below. Use your normal SSH keys/agent and host-key verification; do not commit
credentials. From the **local repository root**:

```bash
cp config/hosting.env.example .hosting.env
```

Edit `.hosting.env` with these values, substituting the absolute path printed by
`pwd` on the host. Values are literal: do not use `~`, `$HOME` or shell quotes.

```dotenv
CAPSTONE_MODE=remote
CAPSTONE_REMOTE_HOST=user@YOUR_HOST
CAPSTONE_REMOTE_WORKSPACE=/home/YOUR_USER/src/Cup_Arrangement
CAPSTONE_GPU=0
CAPSTONE_VIEWER=0
CAPSTONE_VIEWER_PORT=6080
CAPSTONE_PROJECT=bar_cup_arrangement
CAPSTONE_CONTAINER=bar_cup_arrangement_backend
CAPSTONE_ROS_DOMAIN_ID=41
```

For multiple instances, choose unique project/container names, ROS domain and
viewer port. Set any conflicting ROS-TCP/Quest host ports in the **remote**
`ros_backend1.1/.env`. Leave the service bindings on loopback. Hosting values in
the local launcher are passed over SSH; other local shell variables are not.

## 3. Build, launch and check

From the **local repository root**:

```bash
python3 tools/capstone.py config
python3 tools/capstone.py up --dry-run
python3 tools/capstone.py doctor
python3 tools/capstone.py build-image
python3 tools/capstone.py build-workspace
python3 tools/capstone.py up
python3 tools/capstone.py status
```

The dry run should show `ssh` and the intended remote path. `doctor` checks only
the remote Compose version. Expect the workspace build and simulator-start
messages described in the [local checks](../../README.md#check-test-and-stop).
Run the documented `docker exec` clock/controller/tests commands in an SSH
terminal on the host, using the configured container name.

Stop this instance from the local checkout with:

```bash
python3 tools/capstone.py down
```

This removes its container, including container-only build output and logs.
Retrieve needed results first. Bind-mounted source remains on the host. The
launcher does not synchronize local edits or fetch experiment results; transfer
files explicitly and verify revisions before rebuilding.

## 4. Optional remote desktop and camera

Set `CAPSTONE_VIEWER=1` in your **local** `.hosting.env` before launch. Keep a
second **local terminal** open with this tunnel:

```bash
ssh -N -L 6080:127.0.0.1:6080 user@YOUR_HOST
```

Open `http://127.0.0.1:6080/vnc.html` locally. If port 6080 is occupied locally,
use `-L 6082:127.0.0.1:6080` and browse port 6082. If you changed the remote
viewer port, replace the final `6080` accordingly. Do not expose noVNC publicly.

For the CPU configuration, start the separate GUI in a **remote shell** after
`up`, from the remote repository root:

```bash
CONTAINER=bar_cup_arrangement_backend bash ros_backend1.1/scripts/gpu_viewer.sh start
```

For NVIDIA camera rendering, verify GPU access on the host, set
`CAPSTONE_GPU=1` and `CAPSTONE_VIEWER=1` in the local `.hosting.env`, and set
`ENABLE_GRIPPER_CAMERA=1` in the remote backend `.env`. `up` starts the EGL
server and the separate desktop viewer. This combination avoids the existing
lifecycle's automatic GPU viewer trying to use a disabled desktop. If you need
GPU operation without a desktop, use a shell on the host and the local-mode
`ENABLE_VIEWER=0` command in the
[rendering instructions](../../README.md#optional-desktop-and-gpu-rendering).

The SSH tunnel carries the browser viewer only. It does not configure cross-host
ROS 2 DDS discovery. ROS nodes run together inside the remote container; native
DDS clients on another machine need their own network configuration.

## Availability and limitations

The remote host, Docker service and any needed GPU must stay available for the
workload. A disconnected browser or SSH viewer tunnel does not stop a running
Docker container, but an interrupted build/launcher command needs inspection
before retrying. Remote mode does not schedule jobs, restart failed experiments
or guarantee unattended continuation.

A cloud VM can serve as this Linux host if it meets the same requirements;
provisioning, storage persistence and billing are outside the launcher. No fresh
remote/cloud deployment was tested for this documentation update. See
[validation details](../VALIDATION.md) and the
[local troubleshooting table](../../README.md#troubleshooting).
