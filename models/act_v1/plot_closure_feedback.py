"""Plot hypothetical jaw feedback from saved JSON; standard library only."""
import json
from pathlib import Path

root=Path(__file__).parent/'runs/P3_ACT_50-feedback-20261004'
svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="650" viewBox="0 0 1120 650">',
     '<rect width="1120" height="650" fill="white"/><g font-family="Arial" fill="#111827">',
     '<text x="50" y="35" font-size="23">Can partial closure build into full closure? Seed 8, offline model feedback</text>',
     '<text x="50" y="65" font-size="15">Solid: predicted jaw target | Dashed: hypothetical finger position | mm per finger</text>',
     '<text x="50" y="90" font-size="15">Grey: held open | Blue: ideal tracking | Orange: speed-limited tracking (2.8 mm/s)</text>']
for col,visual in enumerate(['frozen_at_expert_close_onset','recorded_expert_sequence']):
    x=65+col*555;y=155;w=470;h=330
    svg.append(f'<text x="{x}" y="{y-20}" font-size="17">'+('Frozen image and arm pose' if col==0 else 'Recorded expert images and arm poses')+'</text>')
    for val in range(0,-7,-1):
        yy=y-val/6*h
        svg.append(f'<path d="M{x},{yy} h{w}" stroke="#ddd"/><text x="{x-28}" y="{yy+5}" font-size="13">{val}</text>')
    for sec in range(6):
        xx=x+sec/5*w
        svg.append(f'<text x="{xx}" y="{y+h+20}" font-size="13">{sec}</text>')
    for cond,color in [('held_open_control','#6b7280'),('ideal_tracking','#2563eb'),('rate_limited_tracking','#d97706')]:
        rows=json.loads((root/(visual+'__'+cond+'.json')).read_text())
        for which in ['target','state']:
            pts=' '.join(f'{x+r["t"]/5*w:.2f},{y-(r["applied_jaw_target"] if which=="target" else sum(r["input_fingers"])/2)*1000/6*h:.2f}' for r in rows)
            dash=' stroke-dasharray="6 4"' if which=='state' else ''
            svg.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"{dash}/>')
    svg.append(f'<text x="{x+120}" y="{y+h+48}" font-size="14">Hypothetical elapsed time (s)</text>')
svg.append('<text x="50" y="590" font-size="15">No physics, cup contact, or image rerendering. Predicted arm commands are not executed.</text>')
svg.append('<text x="50" y="616" font-size="15">This tests model-feedback consistency, not grasp success. Threshold used: target/fingers below −4 mm.</text></g></svg>')
(root/'feedback_comparison.svg').write_text('\n'.join(svg))
