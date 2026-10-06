# Source provenance

This separate source export was prepared on October 6, 2026 from the active Linux
simulation/research checkout derived from
[Noah727/Docker_Teleop](https://github.com/Noah727/Docker_Teleop).

- Source Git branch: `linux-gpu-gripper-camera`.
- Source base commit: `f0b6968833470c7aacf8f10bc3bccc78c16844b0`.
- The export also includes subsequent uncommitted simulator and learning work.
  It is not presented as an exact checkout of that commit.
- The source working tree, recordings, running simulation and original Git
  history were left in place. No existing repository was overwritten.

The Linux source was used because some corresponding Mac backend files were
older. The public export preserves the `ros_backend1.1` and `models` hierarchy
to keep relative package references intact.

Export adjustments:

1. Omitted run artifacts, local configuration, personal notes, build trees,
   environment/bootstrap installers and retired host coordination/driver tools.
2. Changed research repository-root lookups to derive from script location and
   pretrained-weight cache lookups to use the current user's home directory.
3. Removed textures of unverified redistribution provenance; use solid colors.
4. Changed routable example robot IP defaults to loopback in simulation-only
   examples. The active research checkout was not modified by these changes.
5. Added this documentation, artifact exclusions, environment manifest and
   source validator. No new project-wide license was selected.
6. Export-only portability changes use a generic `robot` container user and
   `/workspace` mounts. The local-first launcher optionally invokes the same
   lifecycle over a user-configured SSH host/workspace. It does not provision
   cloud resources. The live research checkout retains its original paths.

`docs/FILE_MANIFEST.json` records the published source files' sizes and SHA256
hashes, excluding the manifest itself. Source-level checks and their limitations
are recorded in `docs/VALIDATION.md`.
