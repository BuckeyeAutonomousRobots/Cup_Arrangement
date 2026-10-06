"""Aggregate completed replay, compare equally clipped first-action baselines."""
import json
from pathlib import Path
import numpy as np
from replay_temporal_comparison import metrics

base=Path(__file__).parent/'runs';root=base/'ACT_temporal_replay_20261004'
r=json.loads((root/'summary.json').read_text());assert r['status']=='completed'
def aggregate(rows):
    n=sum(x['samples'] for x in rows);rot=sum(x['rotation_samples'] for x in rows);hold=sum(x['holding_samples'] for x in rows);steady=sum(x['steady_samples'] for x in rows)
    return dict(samples=n,arm_clamp_samples=sum(x['arm_clamp_samples'] for x in rows),
        wrist_clamp_samples=sum(x['wrist_clamp_samples'] for x in rows),
        arm_clamp_fraction=sum(x['arm_clamp_samples'] for x in rows)/n,
        rotation_mae_rad_s=sum(x['rotation_mae_rad_s']*x['rotation_samples'] for x in rows)/rot,
        correct_rotation_fraction=sum(x['correct_rotation_fraction']*x['rotation_samples'] for x in rows)/rot,
        holding_false_rotation_fraction=sum(x['holding_false_rotation_fraction']*x['holding_samples'] for x in rows)/hold,
        steady_false_rotation_fraction=sum((x['steady_false_rotation_fraction'] or 0)*x['steady_samples'] for x in rows)/steady,
        applied_mae_per_action=(sum(np.array(x['applied_mae_per_action'])*x['samples'] for x in rows)/n).tolist(),
        raw_wrist_peak_abs_rad_s=max(x['raw_wrist_peak_abs_rad_s'] for x in rows))
out={}
for arm,info in r['arms'].items():
    first=[]
    for e in info['episodes']:
        d=np.load(base/'ACT_rotation_sequences_20261004'/f'seed-{e["seed"]}.npz')
        first.append(metrics(d[arm],d['expert'],d['ticks']))
    out[arm]=dict(first_action=aggregate(first),continuous_ensemble=aggregate([e['continuous'] for e in info['episodes']]),
                  reset_at_gaps=aggregate([e['reset_at_gaps'] for e in info['episodes']]),
                  approach_command_integrals=[dict(seed=e['seed'],predicted=e['continuous']['predicted_approach_command_deg'],expert=e['continuous']['expert_approach_command_deg']) for e in info['episodes']])
(root/'comparison.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
