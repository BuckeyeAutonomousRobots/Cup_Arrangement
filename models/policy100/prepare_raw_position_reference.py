"""Recover a joint-only replay reference; never fills training image gaps."""
import bisect
import json
from pathlib import Path
import numpy as np
from train_compare import ROOT,check_workloads
from train_act import JOINTS, GRIP, sha
import fcntl

def main():
    with open('/tmp/baseline_collection_batch.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check_workloads()
        manifest=ROOT/'models/policy100/runs/ACT_Diffusion_100_20261004/dataset_manifest.json'
        e=next(e for e in json.loads(manifest.read_text())['episodes'] if e['seed']==8)
        ep=Path(e['episode']); cache=Path(e['cache'])
        assert sha(ep/'robot_data.jsonl')==e['robot_data_sha256']
        series={n:[] for n in JOINTS};grip=[]
        for line in (ep/'robot_data.jsonl').open():
            r=json.loads(line);m=r['message']
            if r['topic']=='/joint_states':
                s=m['header']['stamp'];t=s['sec']+s['nanosec']*1e-9
                for n,v in zip(m['name'],m['position']):
                    if n in series:series[n].append((t,v))
            elif r['topic']==GRIP and r['receipt_sim_s'] is not None:
                assert abs(m['data'][0]-m['data'][1])<1e-6
                grip.append((r['receipt_sim_s'],float(np.mean(m['data']))))
        for v in [*series.values(),grip]:v.sort()
        times={n:[x[0] for x in v] for n,v in series.items()};gt=[x[0] for x in grip]
        old=np.load(cache/'ticks.npy'); ticks=np.arange(old[0],old[-1]+1)
        states=[];actions=[];ages=[]
        for tick in ticks:
            t=tick/10;row=[]
            for n in JOINTS:
                i=bisect.bisect_right(times[n],t)-1
                assert i>=0 and 0<=t-times[n][i]<=.025,'Raw joint gap'
                row.append(series[n][i][1]);ages.append(t-times[n][i])
            i=bisect.bisect_right(gt,t)-1;assert i>=0
            states.append(row);actions.append([0.]*6+[grip[i][1]])
        out=ROOT/'models/policy100/runs/position_target_pilot_20261005/raw_seed8_reference'
        out.mkdir(exist_ok=False)
        states=np.asarray(states,np.float32)
        np.testing.assert_allclose(states[np.searchsorted(ticks,old)],np.load(cache/'states.npy'),atol=1e-6)
        for k,v in [('states',states),('ticks',ticks),('actions',actions)]:np.save(out/(k+'.npy'),v)
        report={'status':'completed','source_robot_data_sha256':e['robot_data_sha256'],
            'reference_samples':len(ticks),'recovered_camera_dropped_ticks':len(ticks)-len(old),
            'max_joint_age_s':max(ages),'existing_state_equivalence':'passed',
            'note':'Joint-only expert playback reference; no training images filled or synthesized.'}
        (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
