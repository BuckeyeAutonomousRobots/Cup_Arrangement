# Collection reliability components

These tested components were added to the isolated two-worker collection in
October 2026. They are source components, not a complete collection launcher.

- `async_sink.py`: bounded FIFO recording writer. Preserves order, rejects
  queue overflow/write failures, and verifies draining before file closure.
  The recorder integration uses separate image and telemetry writers.
- `startup_probe_rpc_retry.py`: read-only ROS controller readiness probe.
  Queries each controller manager at most twice, with a five-second response
  deadline per query. Missing service or persistent timeout still fails.
  Camera, clock, calibration and initial-pose requirements remain enforced.

The probe requires the running dual-arm ROS setup and generated scene initial
positions in `/tmp`. It does not start a simulator or activate controllers.
The async writer does not change camera transport QoS on its own; the research
deployment also used compatible reliable image publication/subscription and a
reliable joint-state subscription.

```bash
python3 tools/collection/test_async_sink.py
python3 tools/collection/test_probe_retry.py
```

Tests cover ordered draining, disk errors, queue overflow, immediate query
success, timeout followed by success, persistent timeout and a missing service.
The query test executes the actual probe block with controlled responses;
it does not require ROS or issue robot commands.
