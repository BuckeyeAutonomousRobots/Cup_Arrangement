"""Summarize and plot saved paired ablation predictions, without inference."""
import json
from pathlib import Path
import numpy as np

root = Path(__file__).parent/'runs/P3_ACT_50-finger-ablation-20261004'
s = json.loads((root/'summary.json').read_text())
rows=[]
svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="1500" viewBox="0 0 1200 1500">',
     '<rect width="1200" height="1500" fill="white"/><g font-family="Arial" fill="#111827">',
     '<text x="50" y="30" font-size="22">P3 finger-input ablation — all ten validation episodes</text>',
     '<text x="50" y="55" font-size="15">Sequential ACT | black: expert; blue: original; orange: held open; green: delayed 0.5s</text>',
     '<text x="50" y="77" font-size="14">Targets per finger (mm); time relative to expert closing onset. Offline synthetic interventions.</text>']
for k,e in enumerate(s['episodes']):
    d=np.load(root/f'seed-{e["seed"]}.npz'); t=d['relative_sim_s']; mask=d['analysis_mask']
    r={'seed':e['seed'],'expert_onset_elapsed_s':e['expert_onset_elapsed_s'],'conditions':{}}
    for mode in ['sequential','reset_each']:
        for cond in ['original','held_open','delayed']:
            key=mode+'_'+cond; p=d[key][:,6]*1000
            strong=np.flatnonzero(mask&(p<-4))
            r['conditions'][key]={'target_at_2s_mm':float(p[-1]),
                'min_target_mm':float(p[mask].min()),
                'first_strong_close_s':None if not len(strong) else float(t[strong[0]]),
                'onset_025mm_s':e['conditions'][key]['onset_relative_s']['0.25']}
    rows.append(r)
    x=65+(k%2)*590; y=120+(k//2)*270
    svg.append(f'<text x="{x}" y="{y-12}" font-size="17">Validation seed {e["seed"]}</text>')
    for v in [-1,-2,-3,-4,-5]:
        yy=y+(0-v)/5.5*190
        svg.append(f'<path d="M{x},{yy} h480" stroke="#ddd"/><text x="{x-28}" y="{yy+4}" font-size="12">{v}</text>')
    for v in [-2,-1,0,1,2]:
        xx=x+(v+2)/4*480
        svg.append(f'<text x="{xx-5}" y="{y+212}" font-size="12">{v}</text>')
    svg.append(f'<path d="M{x+240},{y} v190" stroke="#888" stroke-dasharray="4 4"/>')
    for key,color in [('expert_actions','#111827'),('sequential_original','#2563eb'),('sequential_held_open','#d97706'),('sequential_delayed','#059669')]:
        pts=' '.join(f'{x+(tt+2)/4*480:.1f},{y-pp*1000/5.5*190:.1f}' for tt,pp in zip(t[mask],d[key][mask,6]))
        svg.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
    svg.append(f'<text x="{x+140}" y="{y+232}" font-size="12">Relative simulation seconds</text>')
svg.append('</g></svg>'); (root/'finger_ablation.svg').write_text('\n'.join(svg))
aggregate={}
for key in rows[0]['conditions']:
    vals=[r['conditions'][key] for r in rows]
    aggregate[key]={'strong_close_episodes':sum(v['first_strong_close_s'] is not None for v in vals),
                    'target_at_2s_mm_mean':float(np.mean([v['target_at_2s_mm'] for v in vals])),
                    'target_at_2s_mm_range':[min(v['target_at_2s_mm'] for v in vals),max(v['target_at_2s_mm'] for v in vals)]}
out={'episodes':rows,'aggregate':aggregate,'status':s['status'],'hashes':s['hashes']}
(root/'comparison.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(aggregate,indent=2))
for r in rows:
    c=r['conditions']; print(r['seed'],*[round(c['sequential_'+v]['target_at_2s_mm'],3) for v in ['original','held_open','delayed']],
                             round(c['reset_each_delayed']['onset_025mm_s']-c['reset_each_original']['onset_025mm_s'],2))
