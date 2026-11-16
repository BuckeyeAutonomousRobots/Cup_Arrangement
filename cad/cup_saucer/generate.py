"""Export circular CAD using baseline dimensions, with simulation handle geometry."""
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import xml.etree.ElementTree as ET
import cadquery as cq
import yaml

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
PROFILE = ROOT / 'ros_backend1.1/profiles/tasks/baseline_v1/task.yaml'
GENERATOR = ROOT / 'ros_backend1.1/simulation/tools/generate_dual_arm_world_from_profiles.py'
spec = importlib.util.spec_from_file_location('simulation_geometry', GENERATOR)
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)
task = yaml.safe_load(PROFILE.read_text())
# Locate objects without depending on the container key name.
def objects(value):
    if isinstance(value, dict):
        if value.get('type') in ('espresso_cup', 'saucer'):
            yield value
        for child in value.values():
            yield from objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from objects(child)
objs = {o['type']: o for o in objects(task)}

def shape_from_link(link):
    parts = []
    for visual in link.findall('visual'):
        g = visual.find('geometry')
        if g.find('box') is not None:
            dims = [float(x)*1000 for x in g.findtext('box/size').split()]
            p = cq.Workplane('XY').box(*dims).val()
        else:
            r = float(g.findtext('cylinder/radius'))*1000
            h = float(g.findtext('cylinder/length'))*1000
            p = cq.Workplane('XY').circle(r).extrude(h).translate((0,0,-h/2)).val()
        pose = [float(x) for x in visual.findtext('pose','0 0 0 0 0 0').split()]
        assert pose[3:5] == [0,0], 'Unexpected roll/pitch; update conversion'
        p = p.rotate((0,0,0),(0,0,1),math.degrees(pose[5])).translate(tuple(x*1000 for x in pose[:3]))
        parts.append(p)
    shape = parts[0].fuse(*parts[1:]).clean() if len(parts)>1 else parts[0]
    return shape.translate((0,0,-shape.BoundingBox().zmin))

link = ET.Element('link')
gen.add_espresso_cup(link, objs['espresso_cup'], objs['espresso_cup']['color_rgba'])
# Preserve the generator's bottom and handle, replace faceted walls with
# true analytic cylindrical surfaces as requested. Simulation files are unchanged.
for visual in list(link.findall('visual')):
    if visual.attrib['name'].startswith('wall_'):
        link.remove(visual)
radius = objs['espresso_cup']['radius'] * 1000
height = objs['espresso_cup']['height'] * 1000
wall = 3.0
body = cq.Workplane('XY').circle(radius).extrude(height).val()
cavity = cq.Workplane('XY').workplane(offset=wall).circle(radius-wall).extrude(height).val()
cup = body.cut(cavity).fuse(shape_from_link(link)).clean()
s = objs['saucer']
saucer = cq.Workplane('XY').circle(s['radius']*1000).extrude(s['height']*1000).val()
report = {'units':'mm', 'source':'checked-in baseline_v1; remote host unavailable during export',
          'geometry_revision':'Circular cup body replaces simulation wall segments; nominal dimensions and handle preserved',
          'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [PROFILE,GENERATOR]}, 'parts':{}}
loaded = {}
for name,shape in [('cup',cup),('saucer',saucer)]:
    assert shape.isValid() and len(shape.Solids()) == 1
    path = OUT / (name+'.step')
    cq.exporters.export(shape,str(path))
    other = cq.importers.importStep(str(path)).val()
    assert other.isValid() and len(other.Solids()) == 1
    bb,ob = shape.BoundingBox(),other.BoundingBox()
    assert abs(shape.Volume()-other.Volume()) < 1e-5
    assert max(abs(getattr(bb,k)-getattr(ob,k)) for k in ['xlen','ylen','zlen']) < 1e-6
    if name == 'cup':
        assert sum(face.geomType() == 'CYLINDER' for face in other.Faces()) >= 2
    loaded[name] = other
    report['parts'][name]={'valid_solid':True,'solid_count':1,'volume_mm3':other.Volume(),
        'bounds_xyz_mm':[ob.xlen,ob.ylen,ob.zlen], 'step_sha256':hashlib.sha256(path.read_bytes()).hexdigest(), 'roundtrip_passed':True}
assembly = cq.Assembly(name='cup_on_saucer')
assembly.add(saucer,name='saucer',color=cq.Color(*s['color_rgba'][:3]))
assembly.add(cup.translate((0,0,s['height']*1000)),name='cup',color=cq.Color(*objs['espresso_cup']['color_rgba'][:3]))
assembly.export(str(OUT/'cup_on_saucer.step'))
assert len(cq.importers.importStep(str(OUT/'cup_on_saucer.step')).solids().vals()) == 2
(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')

# Reuse the depth-tested renderer for the reimported STEP files.
from render import render
render()
print(json.dumps(report['parts'],indent=2))
