# Data contract and reproduction limits

## Policy observations and labels

The cached dataset uses causal 10 Hz observations: the latest wrist RGB image
and eight measured joint positions at or before the grid tick. RGB is resized
to 320 x 240. Learned inputs do not include object ground truth, future joint
states, or expert commands.

The position pilot predicts six measured arm positions at the next contiguous
tick (t + 0.1 s) and the common commanded jaw position at t. These labels are
measured future states, not recovered desired expert setpoints. Missing ticks
and terminal rows are invalid. Twenty-step target chunks are masked at gaps.

The broader dataset contains 100 successful expert episodes. The established
partition is 64 fit / 16 internal calibration / 20 outer episodes. Normalization
is fit-only. The one-demo memorization pilot uses only fit seed 12, including
its normalization. Outer episodes have been inspected in earlier development
and are not an untouched final test set.

At deployment, a position adapter derives bounded velocities from
`(target_position - measured_position) / 0.1`. Existing limits include a
0.245 rad/s velocity cap and a 0.15 rad tracking-error rejection. Runtime
freshness, watchdogs, joint/FK checks and exclusive command ownership are
additional requirements. Offline clipping is not a substitute for these guards.

## Deliberately excluded

- Raw demonstration recordings, robot logs and image caches. The single
  [public expert demo](../assets/demo/README.md) is a presentation video, not a
  training dataset.
- Checkpoints, optimizer states, downloaded pretrained weights and run folders.
- Other videos and run screenshots, slide decks and personal/session notes.
- `.env`, credentials, SSH configuration and host-administration utilities.
- Colcon build/install/log trees, generated worlds, vendored environments,
  dependency installers and CAD source models not required by the runtime.
- Prior generated wood/rubber texture images whose redistribution provenance
  was not established. The export uses solid-color material fallbacks.

The vendor Hand-E mesh package retains its Apache-2.0 license, upstream README
and model credits. Its CAD source files are omitted. See the notices file for
component-specific attribution.

## What this means for reproduction

The repository preserves code and dependency declarations, not a turnkey copy
of the research machine. Supply authorized local recordings and regenerate
manifests/caches before training. Historical experiment scripts refer to named
run directories and fixed seed partitions; inspect their contracts first.

The solid-color export scene differs visually from the textured training scene.
Restoring training-domain textures requires separately verified rights and
matching assets; changing appearance invalidates claims of pixel-identical
policy reproduction. Geometry, control limits and physical task parameters are
retained. No clean Docker build or live run of this export is claimed solely
from the source checks performed before publication.
