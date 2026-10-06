#!/usr/bin/env python3
"""Encode recorded wrist frames at simulation-time speed (not wall time)."""
import argparse
import bisect
import json
import math
from pathlib import Path
import cv2

p = argparse.ArgumentParser()
p.add_argument('episode', type=Path)
a = p.parse_args()
frames = [json.loads(line) for line in (a.episode/'frames.jsonl').open()]
times = [f['stamp_sim_s'] for f in frames]
if not frames or any(b <= a for a, b in zip(times, times[1:])):
    raise RuntimeError('Missing or nonmonotonic frames')
first = cv2.imread(str(a.episode/frames[0]['file']))
if first is None:
    raise RuntimeError('Missing first frame')
writer = cv2.VideoWriter(str(a.episode/'wrist_preview.mp4'),
                         cv2.VideoWriter_fourcc(*'mp4v'), 30,
                         (first.shape[1], first.shape[0]))
if not writer.isOpened():
    raise RuntimeError('Video writer unavailable')
last = -1
try:
    for k in range(math.ceil((times[-1]-times[0])*30)+1):
        i = min(len(frames)-1, max(0, bisect.bisect_right(times, times[0]+k/30)-1))
        if i != last:
            im = cv2.imread(str(a.episode/frames[i]['file']))
            last = i
            if im is None:
                raise RuntimeError('Missing frame')
        writer.write(im)
finally:
    writer.release()
print(a.episode/'wrist_preview.mp4')
