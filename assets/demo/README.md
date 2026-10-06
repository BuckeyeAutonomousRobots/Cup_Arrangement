# Expert cup-placement demonstration

`expert-cup-placement.mp4` shows the successful October 5, 2026 seed-8 expert
position-reference replay, recorded from the simulated wrist camera. It follows
a recorded expert joint trajectory with live joint feedback. It is not a
learned-policy rollout or evidence of autonomous policy success.

The saved trial result reports 172.58 mm maximum cup lift, 0.25019 mm final
cup-to-plate XY error, and reopened fingers after placement. The video covers
48.5 seconds of simulation-time playback; the trial took about 160 seconds of
wall time. The gripper partially occludes the cup in this camera view.

The public MP4 retains the original H.264 video stream at 640 × 480, 30 fps, with
no audio. Container metadata was stripped and the index moved to the beginning
for playback. The JPEG is an unmodified frame from 40 seconds. The original
recording is preserved separately. Sampled frames from approach through release
show only the simulated scene, without a personal desktop, paths or overlays.

- Source run: `expert-position-replay-20261005-04`.
- Source video SHA-256: `1875ef65acf9b65586ea3164fa3346e1a0750f32c9843ada02f62c1ae25efc52`.
- Public video SHA-256: `0a772daffa68cb3aeba375533ea550ab8c6efde9b4c9cfce6b03ccbeac781684`.

This recording comes from the Cup Arrangement research workflow derived from
[Noah727/Docker_Teleop](https://github.com/Noah727/Docker_Teleop). Robot-model and
software credits remain in [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md).
The historical recording includes textured surfaces whose source assets are not
bundled in this repository; the current scene uses solid-color materials. No new
license or ownership claim over third-party models or materials is implied.
