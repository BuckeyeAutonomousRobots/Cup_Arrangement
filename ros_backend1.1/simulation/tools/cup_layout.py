"""Pure seeded sampler. Bounds are model-center coordinates in Gazebo world meters."""
import math
import random


def sample_layout(task, seed):
    cfg = task['randomization']
    bounds = [cfg[key] for key in ('x_bounds', 'y_bounds', 'cup_yaw_bounds')]
    if any(len(b) != 2 or not all(math.isfinite(v) for v in b) or b[0] > b[1] for b in bounds):
        raise ValueError('Bounds must be finite ordered pairs')
    sep = cfg['minimum_separation']
    if not math.isfinite(sep) or sep < 0:
        raise ValueError('minimum_separation must be finite and nonnegative')
    rng = random.Random(seed)
    for _ in range(int(cfg.get('max_attempts', 10000))):
        cx, cy = rng.uniform(*bounds[0]), rng.uniform(*bounds[1])
        sx, sy = rng.uniform(*bounds[0]), rng.uniform(*bounds[1])
        if math.hypot(cx-sx, cy-sy) >= sep:
            objects = {o['id']: o for o in task['objects']}
            top = objects['Sync_TaskPlatform']['size_xyz'][1]
            return {'Sync_EspressoCup': [cx, cy, top + objects['Sync_EspressoCup']['height']/2 + 0.001, rng.uniform(*bounds[2])],
                    'Sync_Saucer': [sx, sy, top + objects['Sync_Saucer']['height']/2 + 0.001, 0.0]}
    raise ValueError('No valid separated layout found; check bounds and minimum_separation')
