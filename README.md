# Cup Arrangement

ROS 2 / Gazebo simulation and imitation-learning research for a UR5e arm with a
Robotiq Hand-E gripper. The goal is a learned policy that reaches a handled cup,
grasps and lifts it, places it on a plate, and releases it using simulated contact.

This is a separate source export for **BuckeyeAutonomousRobots/Cup_Arrangement**.
It does not replace the organization's existing `Capstone` repository.

## Current status

- The physical-contact expert and a recorded position-interface replay have
  completed cup pickup and placement in simulation.
- Learned autonomous pickup and placement are **not yet validated**. Offline
  training improvements are not equivalent to a successful robot rollout.
- Position-output ACT is being tested first on one complete demonstration before
  broader data/model comparisons. See [experiment status](docs/EXPERIMENTS.md).

The expert uses object ground truth; learned policies use wrist RGB and measured
joint positions. Expert replay is not a learned-policy result. No physical robot
operation is included in this repository's supported workflow.

## Layout

| Directory | Contents |
|---|---|
| `ros_backend1.1/` | Docker/ROS simulator, scene profiles, controllers, recording and safety checks |
| `models/baseline_v1/` | Expert collection and initial behavior-cloning utilities |
| `models/act_v1/` | Causal dataset adapter, ACT training and analysis utilities |
| `models/policy100/` | Position-target contracts, single-demo pilot, comparison/analysis source |
| `requirements/` | Observed training environment versions |
| `docs/` | Setup, experiment interpretation, provenance and exclusions |
| `tools/` | Lightweight source validation |

Keep this directory structure: several scripts locate sibling packages relative
to the repository root. Research launchers retain experiment-specific run IDs
and require their documented local data; they are not automatic start-up hooks.

## Getting started

Follow [SETUP.md](docs/SETUP.md) on a Linux machine with Docker Compose. The
simulator uses ROS 2 Humble and Gazebo Fortress inside its container. NVIDIA
rendering is optional; policy training requires a compatible CUDA environment.

```bash
git clone https://github.com/BuckeyeAutonomousRobots/Cup_Arrangement.git
cd Cup_Arrangement
cp config/hosting.env.example .hosting.env
python3 tools/capstone.py doctor
python3 tools/capstone.py build-image
python3 tools/capstone.py build-workspace
python3 tools/capstone.py up
```

Default mode is **local**, headless, with no SSH dependency or personal host
mounts. `python3 tools/capstone.py up --dry-run` shows configuration without
starting anything. An existing remote or cloud Linux host can run the same
code/assets via [optional remote hosting](docs/HOSTING.md).

For a source-only check without starting Docker, ROS, or training:

```bash
python3 tools/validate_source.py
```

Dataset recordings, model weights, optimizer states, videos, generated worlds,
private configuration, and experiment run directories are deliberately absent.
See [data and reproduction requirements](docs/DATA_AND_REPRODUCTION.md).

## Provenance and licensing

Derived from [Noah727/Docker_Teleop](https://github.com/Noah727/Docker_Teleop).
See [PROVENANCE.md](docs/PROVENANCE.md) for the source revision and export changes.
Existing third-party licenses and notices are retained. No new repository-wide
license has been selected; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
