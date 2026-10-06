# Experiment status — October 6, 2026

The task goal is learned autonomous cup pickup, transfer, placement and release.
That goal is **not yet achieved** by the learned policies evaluated here.

## Completed evidence

- A physical-contact expert generated 100 successful simulation demonstrations.
- One recorded expert position-interface replay succeeded: 172.58 mm lift,
  0.250 mm final horizontal placement error, and release. This is an interface
  and tracking result, not learned autonomy or multi-seed generalization.
- Position target/caching checks passed: all 100 cache/target hashes verified,
  and one full fit episode rebuilt exactly from raw recordings. Gap masking,
  split integrity, current-only observations and fit-only normalization were
  checked. Stationary command-rotation coverage remains sparse: 12 fit examples.

## One-demonstration position ACT test

One complete fit demonstration (seed 12) contains 597 cached observations, two
gap boundaries and 594 valid next-position targets. The initial 1,000-update
test and a controlled continuation to 5,000 total updates used the same data,
normalization, architecture, objective, optimizer settings and physical guards.
The continuation restored optimizer state and verified the source checkpoint and
fit probe. Its new shuffled stream was not bitwise uninterrupted continuation.

| Offline same-demo metric | 1,000 updates | 5,000 total updates |
|---|---:|---:|
| Moving-element raw velocity-equivalent MAE | 0.5080 rad/s | 0.3488 rad/s |
| Current-position/no-motion baseline MAE | 0.0823 rad/s | 0.0823 rad/s |
| Correct rotation recall | 36.7% | 18.3% |
| False motion on all-arm holds | 100% | 100% |
| False wrist rotation on steady holds | 100% | 80.5% |
| Jaw MAE | 0.201 mm | 0.0984 mm |
| Strong-close recall | 97.9% | 97.9% |
| Open recall | 77.6% | 98.0% |
| Tracking errors above 0.15 rad | 91/594 | 84/594 |

The continuation completed 4,000 additional updates within its ten-minute
training limit. Saved-checkpoint reload passed. The final policy is ineligible
for simulation control: improving some errors did not satisfy the movement and
holding gates. No simulator trial was run with either checkpoint.

These are teacher-forced diagnostics on recorded expert observations. A single
stationary sample was predicted correctly, which is not robust initiation
evidence. The fit probe did not establish a convergence plateau. The bounded
result does not prove ACT lacks capacity, and it does not isolate insufficient
demonstration diversity as the cause.

Next research decisions should separate optimization/representation limitations
from dataset coverage using controlled tests. No larger factorial experiment or
new demonstration collection is implied by this source release.
