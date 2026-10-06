# Validation

Checks performed on October 6, 2026 for the initial repository revision:

- Python AST syntax checks: 180 files.
- Bash syntax checks: 16 shell scripts.
- XML/Xacro/SDF/Collada parsing: 21 files.
- Five launcher configuration tests: local defaults, shared remote lifecycle,
  configuration precedence without shell evaluation, invalid host/network
  settings, and explicit viewer/domain controls.
- CPU and NVIDIA Compose configurations rendered with Docker Compose v5.0.2.
  Assertions verified generic `/workspace` mounts, BAR-specific container name,
  loopback-only service bindings, headless defaults, and absence of physical
  UR reverse-driver ports.
- Local launcher dry-run resolves to the local lifecycle script without SSH.
- Export scan excludes credentials, known personal host paths, run artifacts,
  private configuration, generated outputs and unused host-administration tools.
- Third-party license files, notices and credits were retained. No new global
  license was selected; unverified scene textures were omitted.

The active research environment separately passed five position-contract tests
and checkpoint reload/provenance checks for its bounded training experiments.
Those results are not a substitute for testing the exported simulator build.

## Not verified

The preparation machine's Docker client was available, but its Docker daemon was
not running. Therefore a clean Docker build and actual Gazebo launch of this
export were **not performed**. The active research simulation was not restarted
or reconfigured. Windows/WSL 2, native macOS Gazebo, a fresh GPU runtime, remote DDS networking,
and cloud deployment were not validated. The optional remote launcher was
tested for configuration/command construction, not against a newly provisioned
host.

The exported solid-color materials differ from the original textured recordings.
No pixel-identical trained-policy reproduction is claimed.

## Local/remote documentation and demo update

The local Linux and Windows/WSL 2 instructions were checked against the launcher,
lifecycle, Dockerfile and Compose files, alongside the linked Docker, Microsoft
and NVIDIA documentation. Bash examples were syntax-checked and relative links
were checked, including heading targets. The source checks, five launcher tests
and CPU/GPU Compose configuration checks passed again.

The selected expert-demo MP4 decodes successfully and contains one H.264 video
stream with no audio. Sampled frames across the recording, including its final
release, were inspected against the saved successful expert replay result.
Only this selected presentation video and its thumbnail are included; raw
recordings and experiment artifacts remain excluded.

The local Docker daemon was unavailable during this update. No image build,
container tests, live simulator launch, Windows/WSL installation or remote
runtime test was performed. No training or new demonstration was run.
