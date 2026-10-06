"""Gap-safe ACT position-target loader. Not connected to production trainer.

Uses the pilot target sidecars; future joints occur ONLY in action labels.
Images/state stay at the current source index. Never crosses a camera gap.
"""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

def chunk_labels(targets, valid, ticks, index, length):
    if not valid[index]: raise ValueError('Current target has no contiguous future state')
    n=1
    while n<length and index+n<len(ticks) and valid[index+n] and ticks[index+n]==ticks[index]+n:
        n+=1
    labels=np.repeat(targets[index+n-1:index+n],length,axis=0)
    labels[:n]=targets[index:index+n]
    return labels, np.arange(length)>=n

class PositionChunks(Dataset):
    def __init__(self, entries, target_directory, stats, chunk_size=20):
        self.episodes=[];self.indices=[];self.stats=stats;self.chunk_size=chunk_size
        for entry in entries:
            p=Path(entry['cache']); side=np.load(Path(target_directory)/f"seed-{entry['seed']}.npz")
            e={k:np.load(p/(k+'.npy'),mmap_mode='r') for k in ('images','states','ticks')}
            assert np.array_equal(e['ticks'],side['ticks'])
            e.update(targets=side['targets'],valid=side['valid'])
            self.indices.extend((len(self.episodes),int(i)) for i in np.flatnonzero(e['valid']))
            self.episodes.append(e)
    def __len__(self):return len(self.indices)
    def __getitem__(self,index):
        number,i=self.indices[index];e=self.episodes[number];s=self.stats
        labels,pad=chunk_labels(e['targets'],e['valid'],e['ticks'],i,self.chunk_size)
        image=np.asarray(e['images'][i],np.float32).transpose(2,0,1)/255
        image=(image-np.array([.485,.456,.406],np.float32)[:,None,None])/np.array([.229,.224,.225],np.float32)[:,None,None]
        return {'observation.images.wrist':torch.from_numpy(image),
            'observation.state':torch.from_numpy(((e['states'][i]-s['state_mean'])/s['state_std']).astype(np.float32)),
            'action':torch.from_numpy(((labels-s['action_mean'])/s['action_std']).astype(np.float32)),
            'action_is_pad':torch.from_numpy(pad)}

def self_test():
    ticks=np.array([0,1,3,4,5]); valid=np.array([True,False,True,True,False])
    y=np.arange(35,dtype=np.float32).reshape(5,7)
    a,p=chunk_labels(y,valid,ticks,0,4)
    assert p.tolist()==[False,True,True,True] and np.array_equal(a[0],y[0])
    a,p=chunk_labels(y,valid,ticks,2,4)
    assert p.tolist()==[False,False,True,True] and np.array_equal(a[1],y[3])
    try:chunk_labels(y,valid,ticks,1,4)
    except ValueError:pass
    else:raise AssertionError('Invalid current target accepted')
    print('PASS: gap masking, future-label chunk order, terminal padding, invalid-index rejection')
if __name__=='__main__':self_test()
