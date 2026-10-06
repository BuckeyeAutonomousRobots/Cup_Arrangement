# Local, remote and cloud hosting

The simulation code, robot description, scene profiles and controllers are the
same in both modes. Hosting configuration chooses where the existing lifecycle
script runs; it does not create another simulator implementation.

## Local default

Copy `config/hosting.env.example` to `.hosting.env`. Default values are local,
CPU rendering and headless operation. A Linux Docker Engine/Compose installation
must be available on this machine. Optional GPU rendering requires an NVIDIA
driver and Container Toolkit compatible with Docker.

`CAPSTONE_PROJECT`, `CAPSTONE_CONTAINER`, `CAPSTONE_ROS_DOMAIN_ID` and the
Ignition partition isolate the selected instance. The default project/container
names are BAR-specific. Choose separate values when running multiple instances.

## Existing remote Linux host

First place the **same revision** of this repository on an authorized Linux
host with Docker/Compose and any required GPU runtime. Use the host's normal
SSH access; do not commit keys, passwords or tokens.

Set these local `.hosting.env` values to your own host and checkout:

```dotenv
CAPSTONE_MODE=remote
CAPSTONE_REMOTE_HOST=user@YOUR_HOST
CAPSTONE_REMOTE_WORKSPACE=/path/to/Cup_Arrangement
CAPSTONE_GPU=0
CAPSTONE_VIEWER=0
```

Then use the same commands:

```bash
python3 tools/capstone.py up --dry-run
python3 tools/capstone.py doctor
python3 tools/capstone.py build-image
python3 tools/capstone.py build-workspace
python3 tools/capstone.py up
python3 tools/capstone.py status
```

The launcher does not upload files, install SSH credentials, configure networks
or reserve a GPU. It runs the same lifecycle script in the specified checkout.
The host must already have capacity and permission for the workload.

## Viewer and ROS networking

Enable `CAPSTONE_VIEWER=1` for the optional desktop/noVNC viewer. Its port is
`CAPSTONE_VIEWER_PORT` (default 6080). Docker publishes it on loopback. For a
remote host, forward it explicitly using your own host value:

```bash
ssh -N -L 6080:127.0.0.1:6080 user@YOUR_HOST
```

Then open `http://127.0.0.1:6080/vnc.html` locally. Use matching port substitutions
if configured differently. Do not expose the viewer directly to the public
internet.

ROS 2 nodes run together in the container with the configured domain ID.
Optional ROS-TCP/Quest services remain loopback-bound through the backend
example configuration. Native cross-host DDS requires matching domain IDs plus
an explicitly configured routable DDS/network setup; SSH alone does not tunnel
DDS multicast. This repository does not claim to configure that automatically.

## Cloud means an existing compatible host

A cloud VM may use remote mode if it supplies Linux, compatible CPU/GPU drivers,
Docker/Compose, authorized SSH, sufficient disk/RAM, and the same repository and
licensed assets. Headless EGL rendering needs a functioning GPU runtime; CPU
rendering performance may be lower. Credentials, billing, provisioning, security
groups and persistence are the operator's responsibility. No paid resource is
created by the launcher, and no cloud provider is assumed.

No clean cloud deployment or live simulation run of the exported configuration
was performed during publication preparation. See `VALIDATION.md` for exactly
what was tested.
