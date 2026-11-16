# 400-demonstration comparison and collection reliability

Published November 15, 2026 from retained local experiment reports. These are
historical recorded results, not a new simulator or training run of this clone.

## Dataset and training completed October 28

400 accepted simulated expert demonstrations, seeds 0–399, were collected in
408 attempts, retaining eight historical failures. The completion report records
a raw-data audit across 639,922 files. The cycle-excluded training view contains
175,048 valid samples and a whole-episode 320/40/40 train/validation/test split.

ACT and Diffusion each trained for 10,000 updates with batch size 16 in all three
action representations. Checkpoint reload and hash checks passed for all six.
Normalization was fitted on training episodes only and matched between policy
families within each representation. The budget represents about 1.14 sampled
training-window passes per model; it does not establish convergence.

|Model|Selected update|Validation score|
|---|---:|---:|
|ACT velocity|7,000|0.169451|
|ACT joint-position reference|10,000|0.052474|
|ACT EE-pose reference|9,000|0.020421|
|Diffusion velocity|10,000|0.103410|
|Diffusion joint-position reference|10,000|0.026122|
|Diffusion EE-pose reference|10,000|0.018379|

The score is first-action physical MAE divided by the training action standard
deviation, averaged over 256 fixed validation windows. It selects checkpoints;
it is not a task success rate. Representations have different target semantics,
and native ACT/Diffusion training losses are not directly comparable. The test
episodes were untouched at this training-completion milestone. This table does
not summarize later rollout experiments or establish reliable autonomous grasping.

The original frozen manifest SHA256 is
`7260f2097051527ca1d6d65b0f45655f4c6ade0a841e42c1298b89f9c8700ccc`.
Data, weights and host-specific manifests remain external to this public repo.

## Recording and startup fixes

A rejected recording contained a 660 ms simulation-time camera gap, exceeding
the unchanged 200 ms maximum. The recorder had synchronous PNG/JSON work inside
callbacks and best-effort image delivery. The deployed repair used bounded FIFO
background writes and reliable image/joint transport. Source timestamps and all
delivered frames were retained; writer failures reject the episode. Both full
qualification demos passed, with maximum frame gaps of 33 ms. The original logs
do not conclusively localize missing frames to the sensor versus the bridge/DDS.

A later startup probe timed out on its first controller-status request even
though all six controller activation messages had appeared. The bounded two-query
probe retained the original readiness conditions. Four controlled query tests
and 25 existing workflow tests passed. Two subsequent full episodes passed
unchanged task/data checks, with 33 ms maximum frame gaps and RTF about 0.234–0.237.
Neither needed the retry during that qualification: transient recovery was tested
with controlled responses, not claimed as observed live recovery.

The expert, physics, randomization, label provenance and acceptance limits were
not changed by these reliability fixes. Persistent failures still stop dispatch;
failed evidence is preserved rather than silently accepted.

## Source included in this update

- [Six-model training source](../models/six_model_v3r3_400_20261028_v2/README.md).
- [Recording and startup components](../tools/collection/README.md).
- [Source hashes and publication adjustments](LOCAL_UPDATE_PROVENANCE.json).

The portable simulation setup and original public demo are retained. Private
machine orchestration, SSH configuration, active claims, application materials,
datasets, training caches and checkpoints are excluded.
