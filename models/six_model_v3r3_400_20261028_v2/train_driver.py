"""Sequential six-model bounded training; no simulation or collection launch."""
from common import *
import subprocess as sp,fcntl,time,shutil,traceback,signal
from transition_guard import completed, gpu_idle, preserve_old, CHECKPOINT_BUDGET_GIB
PYTHON=str(O.parent/'act_v1/.venv/bin/python');STOP=False
def cancel(*_):
 global STOP;STOP=True

def main():
 signal.signal(signal.SIGTERM,cancel);signal.signal(signal.SIGINT,cancel)
 with open('/tmp/baseline_collection_batch.lock','a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  with (O/'TRAINING_CLAIM').open('x') as f:f.write(str(os.getpid()))
  completed(full=True);gpu_idle();preserve_old()
  assert shutil.disk_usage(O).free/2**30 > 60+CHECKPOINT_BUDGET_GIB+5
  check=json.loads((O/'CACHE_VALIDATION.json').read_text());assert check['status']=='passed' and sha(O/'FROZEN_MANIFEST.json')==check['manifest_sha256']
  pins={n:sha(O/n) for n in ['common.py','train_one.py','FROZEN_MANIFEST.json','CACHE_VALIDATION.json','train_driver.py','transition_guard.py','COLLECTION_COMPLETE.json','SPLIT.json','PROTOCOL.txt']};save(O/'TRAINING_PINS.json',pins)
  state={'status':'running','pid':os.getpid(),'started':now(),'models':[],'collection_completed':True,'planned_models':[f'{k}_{r}' for k in KINDS for r in REPS],'frozen_manifest_sha256':sha(O/'FROZEN_MANIFEST.json')};save(O/'TRAINING_PROGRESS.json',state)
  try:
   for kind in KINDS:
    for rep in REPS:
     for n,h in pins.items():assert sha(O/n)==h
     completed(full=True);gpu_idle()
     assert shutil.disk_usage(O).free/2**30>65
     if STOP or (O/'STOP').exists():raise InterruptedError('User stop')
     name=f'{kind}_{rep}';record={'model':name,'status':'starting','started':now()};state['models'].append(record);state['current']=name;save(O/'TRAINING_PROGRESS.json',state)
     with (O/(name+'.training.log')).open('x') as log:
      child=sp.Popen([PYTHON,'-u',str(O/'train_one.py'),'--kind',kind,'--rep',rep,'--steps','10000','--batch','16'],stdout=log,stderr=sp.STDOUT,stdin=sp.DEVNULL)
      record['pid']=child.pid;save(O/'TRAINING_PROGRESS.json',state);deadline=time.monotonic()+5500
      while child.poll() is None:
       if STOP or (O/'STOP').exists() or time.monotonic()>deadline or shutil.disk_usage(O).free/2**30<65:
        child.terminate();child.wait(timeout=45);raise InterruptedError('Training bound/stop/reserve guard')
       time.sleep(5)
      record['returncode']=child.returncode;record['finished']=now();record['status']='completed' if child.returncode==0 else 'failed';save(O/'TRAINING_PROGRESS.json',state)
      assert child.returncode==0,'Training failed: '+name
      result=json.loads((O/'runs'/name/'status.json').read_text());assert result['status']=='completed' and result['checkpoint_reload_pass']
      record['best']=json.loads((O/'runs'/name/'best_metrics.json').read_text());save(O/'TRAINING_PROGRESS.json',state)
   preserve_old()
   state.update(status='completed',finished=now());save(O/'TRAINING_PROGRESS.json',state)
  except BaseException:state.update(status='blocked',error=traceback.format_exc(),finished=now());save(O/'TRAINING_PROGRESS.json',state);raise
if __name__=='__main__':main()
