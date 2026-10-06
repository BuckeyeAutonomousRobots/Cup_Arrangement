"""EXPERT trajectory playback adapter, NOT a learned policy or success evidence."""
import argparse
import json
from pathlib import Path
import socket
import time
import numpy as np

def velocity(target, current):
    target=np.asarray(target); current=np.asarray(current)
    if target.shape!=(6,) or current.shape!=(6,) or not np.isfinite([target,current]).all():
        raise ValueError('Invalid joint vector')
    error=target-current
    if np.max(np.abs(error))>.15: raise RuntimeError('Position tracking error exceeds 0.15 rad')
    return np.clip(error/.1,-.245,.245),error

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--socket',type=Path,required=True);p.add_argument('--log',type=Path,required=True)
    p.add_argument('--self-test',action='store_true');a=p.parse_args()
    q=np.load(a.cache/'states.npy'); t=np.load(a.cache/'ticks.npy')/10
    actions=np.load(a.cache/'actions.npy');t=t-t[0]
    assert np.diff(t).min()>0 and np.diff(t).max()<=.3
    assert np.isfinite(q).all() and np.isfinite(actions).all()
    if a.self_test:
        np.testing.assert_allclose(velocity(np.zeros(6),np.zeros(6))[0],0)
        np.testing.assert_allclose(velocity(np.ones(6)*.04,np.zeros(6))[0],.245)
        try: velocity(np.ones(6),np.zeros(6))
        except RuntimeError: pass
        else: raise AssertionError('tracking guard failed')
        print(json.dumps({'unit_tests':'passed','reference_duration_s':t[-1],
            'max_gap_s':float(np.diff(t).max()),'label':'expert replay, not policy'}));return
    start=None; elapsed=0.;last=None
    with a.log.open('x') as log, socket.socket(socket.AF_UNIX) as server:
        server.bind(str(a.socket));a.socket.chmod(0o600)
        try:
            server.listen(1);server.settimeout(120);conn,_=server.accept()
            with conn,conn.makefile('rwb') as stream:
                conn.settimeout(550)
                for line in stream:
                    wall=time.monotonic();r=json.loads(line);sim=float(r['observation_sim_s'])
                    if start is None:
                        assert np.max(np.abs(np.asarray(r['state'])[:6]-q[0,:6]))<.02,'Initial pose mismatch'
                        start=sim
                    if last is not None:
                        assert sim>last,'Nonmonotonic observation time'
                        # Pause reference progress during observation recovery.
                        if not r.get('reset_policy'):elapsed+=sim-last
                    last=sim
                    target=np.array([np.interp(elapsed+.1,t,q[:,j]) for j in range(6)])
                    v,err=velocity(target,r['state'][:6])
                    idx=max(0,np.searchsorted(t,elapsed,side='right')-1)
                    jaw=float(actions[min(idx,len(actions)-1),6])
                    log.write(json.dumps({'id':r['id'],'elapsed_s':elapsed,'target':target.tolist(),
                        'error':err.tolist(),'velocity':v.tolist(),'jaw':jaw})+'\n');log.flush()
                    stream.write((json.dumps({'id':r['id'],'action':[*v.tolist(),jaw],
                        'inference_wall_s':time.monotonic()-wall})+'\n').encode());stream.flush()
        finally:a.socket.unlink(missing_ok=True)

if __name__=='__main__':main()
