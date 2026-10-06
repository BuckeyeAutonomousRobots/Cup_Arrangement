"""Plot saved offline replay results; no inference or robot interfaces."""
from pathlib import Path
import json
import numpy as np

root = Path(__file__).parent/'runs/P3_ACT_50-offline-closure-20260928'
svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="920" viewBox="0 0 1200 920">',
       '<rect width="1200" height="920" fill="white"/>',
       '<g font-family="Arial" fill="#111827"><text x="40" y="30" font-size="22">P3 on expert observations — offline replay, not a robot rollout</text>',
       '<text x="40" y="58" font-size="15">Black: expert | Blue: sequential P3 | Orange: reset each frame</text>']
for row, seed in enumerate([8, 18, 2]):
    d = np.load(root/f'seed-{seed}.npz')
    t = d['elapsed_sim_s']
    for col, channel in enumerate([6, 5]):
        x, y = 70+col*590, 110+row*270
        scale = 1000 if channel == 6 else 1
        for key, label, color in [('expert_actions', 'Expert command', '#111827'),
                                  ('sequential_ensemble', 'P3 sequential', '#2563eb'),
                                  ('reset_each_observation', 'P3 reset each frame', '#d97706')]:
            low, high = (-5.5, .5) if channel == 6 else (-.26, .26)
            points = ' '.join(f'{x+v/t[-1]*480:.2f},{y+190-(w-low)/(high-low)*190:.2f}'
                              for v, w in zip(t, d[key][:, channel]*scale))
            svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.5"/>')
        title = f'Seed {seed} ({"train" if seed == 2 else "validation"}): '+('jaw target (mm)' if channel == 6 else 'wrist 3 velocity (rad/s)')
        svg.append(f'<text x="{x}" y="{y-15}" font-size="16">{title}</text>')
        svg.append(f'<rect x="{x}" y="{y}" width="480" height="190" fill="none" stroke="#9ca3af"/>')
        for val in ([0,-2,-4] if channel == 6 else [-.2,0,.2]):
            yy = y+190-(val-low)/(high-low)*190
            svg.append(f'<text x="{x-40}" y="{yy+4}" font-size="12">{val}</text><path d="M{x},{yy} h480" stroke="#ddd" stroke-dasharray="3 5"/>')
        for val in range(0, int(t[-1])+1, 10):
            xx = x+val/t[-1]*480
            svg.append(f'<text x="{xx}" y="{y+210}" font-size="12">{val}</text>')
        svg.append(f'<text x="{x+100}" y="{y+232}" font-size="13">Elapsed expert simulation time (s)</text>')
    first = np.flatnonzero(d['expert_actions'][:, 6] < -.004)[0]
    mask = np.arange(len(t)) < first
    print(json.dumps({'seed':seed, 'approach_wrist3_mean_abs_expert':float(np.abs(d['expert_actions'][mask,5]).mean()),
                      'approach_wrist3_mean_abs_P3':float(np.abs(d['sequential_ensemble'][mask,5]).mean())}))
svg.append('</g></svg>')
(root/'expert_closure_comparison.svg').write_text('\n'.join(svg))
