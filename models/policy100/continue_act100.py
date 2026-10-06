"""One offline ACT continuation; existing model runs remain read-only."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import traceback
from datetime import datetime
import train_compare as t

SOURCE=t.ROOT/'models/policy100/runs/ACT_Diffusion_100_20261004'
OUT=t.ROOT/'models/policy100/runs/ACT_100_10k_20261005'
DEADLINE='2026-10-05T09:40:00-04:00'
TARGET_STEPS=10000

def main():
    deadline=datetime.fromisoformat(DEADLINE).timestamp()
    signal.signal(signal.SIGTERM,t.stopped);signal.signal(signal.SIGINT,t.stopped)
    t.torch.set_num_threads(2)
    import cv2
    cv2.setNumThreads(1)
    with open('/tmp/baseline_collection_batch.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);t.check_workloads();t.check_deadline(deadline)
        assert not OUT.exists(),'Existing run; inspect, never relaunch'
        src=SOURCE/'ACT_100'
        status=json.loads((src/'status.json').read_text())
        assert status['status']=='completed' and status['step']==5000 and status['checkpoint_reload_pass']
        assert t.sha(src/'last_policy/model.safetensors')==t.sha(src/'last_raw_policy/model.safetensors')==status['last_policy_sha256']
        assert t.sha(SOURCE/'dataset_manifest.json')==status['dataset_sha256']
        OUT.mkdir()
        state={'status':'validating','pid':os.getpid(),'started':t.now(),'deadline':DEADLINE,'target_steps':TARGET_STEPS,
               'source':str(src),'source_checkpoint_sha256':status['last_policy_sha256'],
               'source_optimizer_sha256':t.sha(src/'optimizer.pt'),'trainer_sha256':t.sha(Path(t.__file__)),
               'offline_only':True}
        def update(**kw):
            state.update(kw,updated=t.now());t.atomic_json(OUT/'status.json',state);print(json.dumps(state),flush=True)
        update()
        try:
            manifest=json.loads((SOURCE/'dataset_manifest.json').read_text())
            for e in manifest['episodes']:
                t.check_deadline(deadline)
                for key,digest in e['cache_sha256'].items():
                    assert t.sha(Path(e['cache'])/(key+'.npy'))==digest,(e['seed'],key)
            shutil.copy2(SOURCE/'dataset_manifest.json',OUT/'dataset_manifest.json')
            (OUT/'cache').symlink_to(SOURCE/'cache',target_is_directory=True)
            stats=dict(t.np.load(SOURCE/'act_normalization.npz'))
            train=[e for e in manifest['episodes'] if e['split']=='train']
            val=[e for e in manifest['episodes'] if e['split']=='validation']
            assert len(train)==80 and len(val)==20
            t.check_workloads();assert t.torch.cuda.is_available()
            t.torch.cuda.set_per_process_memory_fraction(.55)
            update(status='training')
            t.train_one('ACT_100',OUT,train,val,stats,None,TARGET_STEPS,deadline,update,resume=src)
            assert t.sha(src/'last_policy/model.safetensors')==state['source_checkpoint_sha256']
            assert t.sha(src/'optimizer.pt')==state['source_optimizer_sha256']
            update(status='completed',finished=t.now(),source_preserved=True)
        except InterruptedError as e:update(status='stopped_at_deadline',error=str(e),finished=t.now())
        except Exception:
            update(status='failed',error=traceback.format_exc(),finished=t.now());raise

if __name__=='__main__':main()
