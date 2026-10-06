"""Offline regression tests for temporal alignment; no GPU/ROS commands."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from train_compare import DiffusionChunks, IMAGE

class AlignmentTest(unittest.TestCase):
    def test_gap_and_initial_padding(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); ticks=np.array([10,11,12,20,21])
            arrays={'ticks':ticks,'images':np.zeros((5,240,320,3),np.uint8),
                    'states':np.repeat(np.arange(5,dtype=np.float32)[:,None],8,axis=1),
                    'actions':np.repeat(np.arange(5,dtype=np.float32)[:,None],7,axis=1)}
            for k,v in arrays.items(): np.save(p/(k+'.npy'),v)
            stats={'state_center':np.zeros(8,np.float32),'state_scale':np.ones(8,np.float32),
                   'action_center':np.zeros(7,np.float32),'action_scale':np.ones(7,np.float32)}
            ds=DiffusionChunks([{'cache':tmp}],stats)
            first=ds[0]; self.assertEqual(first[IMAGE].shape,(2,3,240,320))
            self.assertEqual(first['action'][:4,0].tolist(),[0,0,1,2])
            self.assertEqual(first['action_is_pad'][:5].tolist(),[True,False,False,False,True])
            middle=ds[1];self.assertEqual(middle['observation.state'][:,0].tolist(),[0,1])
            self.assertEqual(middle['action'][1,0],1)
            gap=ds[3];self.assertEqual(gap['observation.state'][:,0].tolist(),[3,3])
            self.assertEqual(gap['action'][:4,0].tolist(),[3,3,4,4])
            self.assertTrue(gap['action_is_pad'][0]);self.assertTrue(gap['action_is_pad'][3])
            self.assertTrue(torch.isfinite(gap[IMAGE]).all())

if __name__=='__main__': unittest.main()
