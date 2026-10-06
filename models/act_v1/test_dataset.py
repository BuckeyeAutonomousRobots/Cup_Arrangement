import tempfile,unittest
from pathlib import Path
import numpy as np
from train_act import Chunks,IMAGE

class ChunkTests(unittest.TestCase):
    def test_chunks_do_not_cross_missing_time_or_episode_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            np.save(p/'images.npy',np.zeros((3,240,320,3),dtype=np.uint8))
            np.save(p/'states.npy',np.zeros((3,8),dtype=np.float32))
            np.save(p/'actions.npy',np.repeat(np.arange(3,dtype=np.float32)[:,None],7,axis=1))
            np.save(p/'ticks.npy',np.array([0,1,3]))
            stats={'state_mean':np.zeros(8,dtype=np.float32),'state_std':np.ones(8,dtype=np.float32),
                'action_mean':np.zeros(7,dtype=np.float32),'action_std':np.ones(7,dtype=np.float32)}
            d=Chunks([{'cache':str(p)}],stats,chunk=4)
            a=d[0];self.assertEqual(a['action_is_pad'].tolist(),[False,False,True,True])
            self.assertEqual(a['action'][:,0].tolist(),[0,1,1,1])
            self.assertEqual(d[1]['action_is_pad'].tolist(),[False,True,True,True])
            self.assertEqual(d[2]['action_is_pad'].tolist(),[False,True,True,True])
            self.assertEqual(tuple(a[IMAGE].shape),(3,240,320))
            self.assertTrue(np.isfinite(a[IMAGE].numpy()).all())

if __name__=='__main__':unittest.main()
