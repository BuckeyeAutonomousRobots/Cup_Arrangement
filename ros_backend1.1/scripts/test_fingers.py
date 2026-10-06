"""Bounded finger-only tracking test; leaves fingers open and arm stationary."""
import json
import signal
import time
from cup_task import GroundTruthController, active_task, task_lock, exclusive_commands, interrupted, check_cancelled

signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)
with task_lock(), exclusive_commands():
    c = GroundTruthController(active_task())
    names = ['right_robotiq_hande_'+s+'_finger_joint' for s in ('left','right')]
    try:
        initial = [c.q[n] for n in c.names]
        for target in [0., -.01, 0.]:
            end = time.monotonic()+15
            while time.monotonic()<end:
                check_cancelled()
                c.command([0]*6)
                c.grip.publish(c.Msg(data=[target,target]))
                c.ros.spin_once(c.node,timeout_sec=.02)
            q=[c.q[n] for n in names]
            drift=max(abs(c.q[n]-a) for n,a in zip(c.names,initial))
            print(json.dumps(dict(target=target,fingers=q,center_offset_m=(q[1]-q[0])/2,arm_drift_rad=drift)),flush=True)
            if max(abs(x-target) for x in q)>.001:
                raise RuntimeError('Finger tracking error exceeds 1 mm')
    finally:
        c.grip.publish(c.Msg(data=[0.,0.]))
        c.close()
