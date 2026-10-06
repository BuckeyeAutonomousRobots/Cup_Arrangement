"""Explicit episode-level split; shared by trainer and offline tests."""
import json
from pathlib import Path


def load_manifest(path):
    rows = json.loads(Path(path).read_text())['episodes']
    seen_seeds, seen_paths = set(), set()
    result = []
    for row in rows:
        seed, split = row['seed'], row['split']
        episode = Path(row['path'])
        if type(seed) is not int or seed < 0 or split not in ('train', 'validation') or not episode.is_absolute():
            raise ValueError('Invalid seed, split or episode path')
        episode = episode.resolve()
        if seed in seen_seeds or episode in seen_paths:
            raise ValueError('Duplicate seed/path across dataset splits')
        seen_seeds.add(seed); seen_paths.add(episode)
        result.append(dict(seed=seed, split=split, path=episode))
    if not all(any(r['split']==s for r in result) for s in ('train','validation')):
        raise ValueError('Both train and validation episodes are required')
    return sorted(result, key=lambda r:r['seed'])
