"""Authorized continuation: same fit demo, at most 4000 MORE updates / 600 training seconds.

No remote commands, ROS imports, collection, simulator control, or outer evaluation.
An unmet gate at this budget is inconclusive about architectural capacity.
"""
import fcntl
import json
import os
from pathlib import Path
import random
import signal
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import torch
from torch.utils.data import DataLoader
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy

from position_act_contract import sha, partition, validate_targets, fit_normalization
from position_dataset import PositionChunks
from single_position_metrics import evaluate, gates
from train_compare import check_workloads

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'models/policy100/runs'
SOURCE = BASE / 'SinglePositionACT_seed12_20261006_01'
OUT = BASE / 'SinglePositionACT_seed12_continue_20261006_01'
AUDIT = BASE / 'PositionStages12_audit_20261006_01'
IMAGE = 'observation.images.wrist'
STOP = False


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def stopped(*_):
    global STOP
    STOP = True


def main():
    signal.signal(signal.SIGTERM, stopped)
    signal.signal(signal.SIGINT, stopped)
    start = time.monotonic()
    last_check = -float('inf')

    def guard(force=False):
        nonlocal last_check
        if STOP or time.monotonic() - start > 850:
            raise InterruptedError('Pilot wall bound or termination request')
        if force or time.monotonic() - last_check >= 10:
            check_workloads()
            last_check = time.monotonic()

    with open('/tmp/baseline_collection_batch.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guard(True)
        audit = json.loads((AUDIT / 'summary.json').read_text())
        assert audit['status'] == 'passed' and audit['stage2_allowed'] and audit['seed'] == 12
        assert not OUT.exists(), 'No duplicate run or automatic retry'
        OUT.mkdir()
        report = dict(status='preparing', pid=os.getpid(), started=datetime.now(timezone.utc).isoformat(),
                      seed=12, steps_limit=5000, additional_steps_limit=4000, training_seconds_limit=600, offline_only=True,
                      selection='Final checkpoint at 5000 total updates or continuation training-time bound. No best-checkpoint search, no calibration or outer evaluation.',
                      training='Restore seed12 ACT at step1000 and its AdamW optimizer; ImageNet ResNet18, 20 absolute-position targets, uniform shuffled batch 8, AdamW lr1e-4/backbone1e-5, decay1e-4, gradclip1, seed42; stock masked L1 + KL.',
                      normalization='Only valid rows from selected FIT seed12 for the one-demo overfit. The fixed 64/16/20 partition remains unchanged; no calibration or outer data used.',
                      metrics_predeclared='Raw position error divided by0.1: >=90% reduction vs identity on moving elements (target speed>.02rad/s); all observed stationary command-rotation samples correct, rotation recall>=95%; all-arm holding false<=1%, wrist steady false<=0.5%; jaw MAE<=0.2mm, strong-close/open recall>=95%; zero tracking-error violations>.15rad; adapter saturation at most5percentage points above the fixed measured-reference fraction. Actual commands remain clipped at+/-.245rad/s.',
                      runtime='Stock ACT .01 temporal ensemble, reset at each gap. Exact single-observation calls for final metrics. All actual runtime/ownership/latency checks still required before any live trial.',
                      scope='One complete successful recorded episode with two masked cache gaps. Memorization only, not generalization or autonomous task success.',
                      history=[], live_trial_run=False)

        def update(**values):
            report.update(values, elapsed_seconds=time.monotonic() - start)
            atomic(OUT / 'status.json', report)
            print(json.dumps({k:report.get(k) for k in ('status','step','elapsed_seconds','training_seconds','error')}), flush=True)

        policy = optimizer = None
        step = 0
        try:
            update()
            torch.set_num_threads(2); torch.set_num_interop_threads(1)
            assert torch.cuda.is_available()
            torch.cuda.set_per_process_memory_fraction(.55)
            random.seed(42); np.random.seed(42); torch.manual_seed(42)
            source = BASE / 'ACT_Diffusion_100_20261004/dataset_manifest.json'
            prior = BASE / 'FeatureBC_calibrated_20261005/summary.json'
            manifest = json.loads(source.read_text())
            assert sha(source) == audit['dataset_manifest_sha256']
            splits = partition(manifest, json.loads(prior.read_text()))
            entry = next(e for e in splits['fit'] if e['seed'] == 12)
            cache = Path(entry['cache'])
            for name, digest in entry['cache_sha256'].items():
                assert sha(cache / (name + '.npy')) == digest
            target_dir = BASE / 'position_target_pilot_20261005'
            side_path = target_dir / 'seed-12.npz'
            assert sha(side_path) == next(e['target_sha256'] for e in audit['episodes'] if e['seed'] == 12)
            q, command, ticks = [np.load(cache / (k + '.npy')) for k in ('states','actions','ticks')]
            side = dict(np.load(side_path))
            validate_targets(q, command, ticks, side)
            stats = fit_normalization([entry], target_dir)
            np.savez(OUT / 'normalization.npz', **stats)
            identity = np.column_stack([q[:, :6], q[:, 6:8].mean(1)])
            report['identity'] = evaluate(identity, q, command, side, ticks)
            reference=side['targets'].copy()
            reference[~side['valid']]=identity[~side['valid']]
            report['measured_reference']=evaluate(reference,q,command,side,ticks)
            report['saturation_scoring_reason']='Before training, stage1 measured reference saturated112/594 ticks because measured motion can slightly exceed the commanded .245rad/s limit (peak .24990). Count this inherent adapter saturation separately from runtime clamps; retain all actual motion guards.'
            weights = (Path.home() / '.cache/torch/hub/checkpoints/resnet18-f37072fd.pth')
            assert sha(weights) == 'f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'
            atomic(OUT / 'provenance.json', dict(audit_sha256=sha(AUDIT / 'summary.json'), source_manifest_sha256=sha(source),
                  source_episode=entry, target_sha256=sha(side_path), fit_seeds=[12],
                  preserved_splits={k:[e['seed'] for e in v] for k,v in splits.items()},
                  normalization_sha256=sha(OUT / 'normalization.npz'), backbone_sha256=sha(weights),
                  source_hashes={name:sha(Path(__file__).parent / name) for name in
                                 ['continue_single_position_once.py','single_position_metrics.py','position_act_contract.py','position_dataset.py','train_compare.py']},
                  torch_version=torch.__version__, numpy_version=np.__version__))
            ds = PositionChunks([entry], target_dir, stats)
            loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0, drop_last=True)
            iterator = iter(loader)
            config = ACTConfig(input_features={IMAGE:PolicyFeature(type=FeatureType.VISUAL, shape=(3,240,320)),
                                               'observation.state':PolicyFeature(type=FeatureType.STATE, shape=(8,))},
                               output_features={'action':PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
                               device='cuda', push_to_hub=False, chunk_size=20, n_action_steps=1,
                               temporal_ensemble_coeff=.01, optimizer_lr=1e-4, optimizer_lr_backbone=1e-5)
            policy = ACTPolicy(config).cuda()
            optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=1e-4, weight_decay=1e-4)
            source_status=json.loads((SOURCE/'summary.json').read_text())
            assert source_status['status']=='completed' and source_status['step']==1000
            assert sha(SOURCE/'final_policy/model.safetensors')==source_status['checkpoint_sha256']
            original=ACTPolicy.from_pretrained(SOURCE/'final_policy').cuda()
            policy.load_state_dict(original.state_dict());del original
            saved=torch.load(SOURCE/'optimizer.pt',map_location='cuda',weights_only=False)
            assert saved['step']==1000
            optimizer.load_state_dict(saved['optimizer'])
            assert all(int(item['step'])==1000 for item in optimizer.state.values())
            source_stats=dict(np.load(SOURCE/'normalization.npz'))
            for key in stats:np.testing.assert_array_equal(stats[key],source_stats[key])
            atomic(OUT/'resume_provenance.json',dict(source_checkpoint_sha256=source_status['checkpoint_sha256'],
                  source_optimizer_sha256=sha(SOURCE/'optimizer.pt'),source_summary_sha256=sha(SOURCE/'summary.json'),
                  source_step=1000,optimizer_steps_verified=True,normalization_identical=True,
                  rng='Original sampler/RNG state unavailable. Fresh seed42 shuffled stream, not bitwise uninterrupted continuation.'))
            step=1000
            images = np.load(cache / 'images.npy', mmap_mode='r')
            mean = np.array([.485,.456,.406],np.float32)[:,None,None]
            std = np.array([.229,.224,.225],np.float32)[:,None,None]

            def obs(i):
                image = np.asarray(images[i],np.float32).transpose(2,0,1) /255
                state = ((q[i]-stats['state_mean'])/stats['state_std']).astype(np.float32)
                return {IMAGE:torch.from_numpy((image-mean)/std)[None].cuda(),
                        'observation.state':torch.from_numpy(state)[None].cuda()}

            probe_ids = np.linspace(0,len(ds)-1,min(64,len(ds)),dtype=int)

            def probe():
                policy.eval(); errors=[]
                with torch.no_grad():
                    for idx in probe_ids:
                        guard()
                        _,i=ds.indices[int(idx)]
                        pred = policy.predict_action_chunk(obs(i))[0,0].cpu().numpy()*stats['action_std']+stats['action_mean']
                        errors.append(np.abs(pred-side['targets'][i]))
                return (np.mean(errors,axis=0)).tolist()

            baseline_probe=probe()
            np.testing.assert_allclose(baseline_probe,source_status['history'][-1]['probe_position_mae'],rtol=1e-4,atol=1e-7)
            update(status='training',step=1000,source_probe_reproduction_pass=True,fit_samples=len(ds),model_parameters=sum(p.numel() for p in policy.parameters()))
            training_start=time.monotonic(); losses=[]
            for next_step in range(1001,5001):
                guard()
                if time.monotonic()-training_start>=600:
                    break
                policy.train()
                try: batch=next(iterator)
                except StopIteration: iterator=iter(loader);batch=next(iterator)
                batch={k:v.cuda() for k,v in batch.items()}
                loss,detail=policy(batch)
                assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),1)
                assert torch.isfinite(norm)
                optimizer.step();step=next_step;losses.append(float(loss.detach()))
                time.sleep(.02)
                if step%50==0 or step==1:
                    update(step=step,training_seconds=time.monotonic()-training_start,last50_loss=float(np.mean(losses[-50:])))
                if step%250==0:
                    checkpoint_metric=dict(step=step,mean_training_loss=float(np.mean(losses[-250:])),probe_position_mae=probe())
                    report['history'].append(checkpoint_metric)
                    update(training_seconds=time.monotonic()-training_start)
            policy.eval();policy.save_pretrained(OUT/'final_policy')
            torch.save(dict(step=step,optimizer=optimizer.state_dict()),OUT/'optimizer.pt')
            with torch.no_grad(): expected=policy.predict_action_chunk(obs(0)).clone()
            loaded=ACTPolicy.from_pretrained(OUT/'final_policy').cuda().eval()
            with torch.no_grad():torch.testing.assert_close(loaded.predict_action_chunk(obs(0)),expected,rtol=1e-4,atol=1e-5)
            del policy,optimizer,iterator,loader
            policy=loaded;optimizer=None
            update(status='evaluating',step=step,training_seconds=time.monotonic()-training_start,
                   checkpoint_reload_pass=True,checkpoint_sha256=sha(OUT/'final_policy/model.safetensors'))
            # Single-sample inference matches intended deployment. No batched
            # numerical equivalence assumption and no timing reset at each action.
            outputs=[];latencies=[];policy.reset()
            with torch.no_grad():
                for i in range(len(ticks)):
                    guard()
                    if i and ticks[i]!=ticks[i-1]+1:policy.reset()
                    batch=obs(i);torch.cuda.synchronize();ts=time.monotonic()
                    output=policy.select_action(batch)[0].cpu().numpy()
                    latencies.append(time.monotonic()-ts)
                    outputs.append(output*stats['action_std']+stats['action_mean'])
            position=np.stack(outputs)
            metrics=evaluate(position,q,command,side,ticks)
            gate_result=gates(metrics,report['measured_reference'])
            np.savez_compressed(OUT/'same_demo_predictions.npz',position=position,targets=side['targets'],
                                states=q,commands=command,ticks=ticks,valid=side['valid'])
            profile=dict(median_s=float(np.median(latencies)),p95_s=float(np.quantile(latencies,.95)),max_s=float(max(latencies)),
                         note='Offline single-sample GPU inference only; not IPC/ROS latency validation.')
            update(status='completed',result='offline_pass_requires_runtime_preflight' if all(gate_result.values()) else 'inconclusive_budget_limited',
                   same_demo=metrics,gates=gate_result,offline_inference=profile,
                   eligible_for_runtime_preflight=all(gate_result.values()),
                   interpretation='An unmet gate after this bounded continuation is not proof of insufficient ACT capacity. No further run or simulator trial is automatic.',
                   fit_probe_plateau=(len(report['history'])>=3 and
                       (max(float(np.mean(h['probe_position_mae'][:6])) for h in report['history'][-3:])-
                        min(float(np.mean(h['probe_position_mae'][:6])) for h in report['history'][-3:])) /
                       max(float(np.mean([np.mean(h['probe_position_mae'][:6]) for h in report['history'][-3:]])),1e-12)<.02),
                   convergence_rule='Descriptive plateau only if last3 fixed fit-probe mean arm-position errors span<2% of their mean; no checkpoint selection.',
                   finished=datetime.now(timezone.utc).isoformat())
            atomic(OUT/'summary.json',report)
        except BaseException as exc:
            if policy is not None and step:
                policy.save_pretrained(OUT/'interrupted_policy')
                if optimizer is not None:torch.save(dict(step=step,optimizer=optimizer.state_dict()),OUT/'interrupted_optimizer.pt')
            update(status='stopped' if isinstance(exc,(InterruptedError,KeyboardInterrupt)) else 'failed',
                   step=step,error=repr(exc),traceback=traceback.format_exc())
            raise


if __name__=='__main__':main()
