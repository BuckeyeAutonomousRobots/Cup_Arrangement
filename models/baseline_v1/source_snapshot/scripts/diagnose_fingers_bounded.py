"""Approved free-space stationary-arm diagnostic, <=90 wall seconds, no lift."""
import json,signal,time
from cup_task import GroundTruthController,active_task,task_lock,exclusive_commands,interrupted,check_cancelled
signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
with task_lock(),exclusive_commands():
    c=GroundTruthController(active_task())
    begun=time.monotonic()
    initial=[c.q[n] for n in c.names]
    try:
        if c.fk(c.np.array(initial))[2,3]<.45:raise RuntimeError('Not at clear home height')
        for label,target in [('open',[0.,0.]),('left_only',[-.005,0.]),('reopen',[0.,0.]),('right_only',[0.,-.005]),('reopen',[0.,0.]),('both',[-.005,-.005]),('final_open',[0.,0.])]:
            stage=time.monotonic();stable=None;last_report=0
            while time.monotonic()-stage<10:
                check_cancelled()
                if time.monotonic()-begun>80:raise RuntimeError('Diagnostic deadline')
                c.ros.spin_once(c.node,timeout_sec=.02)
                if time.monotonic()-c.last_joint>2:raise RuntimeError('Stale arm state')
                drift=max(abs(c.q[n]-a) for n,a in zip(c.names,initial))
                if drift>.01:raise RuntimeError('Arm drift exceeded 0.01 rad')
                c.command([0.]*6);c.grip.publish(c.Msg(data=target))
                fingers=c.finger_positions()
                good=max(abs(a-b) for a,b in zip(fingers,target))<.0005
                stable=(stable or time.monotonic()) if good else None
                if time.monotonic()-last_report>1:
                    print(json.dumps(dict(stage=label,target=target,fingers=fingers,arm_drift_rad=drift)),flush=True);last_report=time.monotonic()
                if stable is not None and time.monotonic()-stable>.3:break
            else:raise RuntimeError('Finger tracking failure in '+label)
        print(json.dumps({'result':'PASS','wall_seconds':time.monotonic()-begun}),flush=True)
    finally:
        end=time.monotonic()+5
        while time.monotonic()<end:
            c.command([0.]*6);c.grip.publish(c.Msg(data=[0.,0.]))
            c.ros.spin_once(c.node,timeout_sec=.02)
        print(json.dumps({'cleanup_fingers':c.finger_positions()}),flush=True)
        c.close()
