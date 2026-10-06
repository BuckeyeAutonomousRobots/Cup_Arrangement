import unittest
from observation_recovery import ObservationRecovery

class RecoveryTests(unittest.TestCase):
    def test_normal_and_boundary(self):
        r=ObservationRecovery()
        self.assertEqual(r.update(0,.1,0,[.1]*8,0,False),'normal')

    def test_requires_new_camera_and_joints(self):
        r=ObservationRecovery()
        self.assertEqual(r.update(0,1,.898,[1]*8,1,False),'wait')
        self.assertEqual(r.update(.1,1.01,1.01,[1]*8,2,False),'wait')
        self.assertEqual(r.update(.2,1.02,1.02,[1.02]*8,3,False),'resume')
        self.assertAlmostEqual(r.events[0]['duration_wall_s'],.2)

    def test_pending_must_drain(self):
        r=ObservationRecovery()
        r.update(0,1,.8,[1]*8,1,True)
        self.assertEqual(r.update(.1,1.01,1.01,[1.01]*8,2,True),'wait')
        self.assertEqual(r.update(.2,1.02,1.02,[1.02]*8,3,False),'resume')

    def test_timeout_even_with_fresh_data(self):
        r=ObservationRecovery(); r.update(0,1,.8,[1]*8,1,False)
        with self.assertRaisesRegex(RuntimeError,'0.5'):
            r.update(.5,1.01,1.01,[1.01]*8,2,False)

    def test_joint_staleness_and_future_image(self):
        for image,joints in [(1,[.8]*8),(1.2,[1]*8)]:
            self.assertEqual(ObservationRecovery().update(0,1,image,joints,1,False),'wait')

    def test_frequent_pause(self):
        r=ObservationRecovery(); r.update(0,1,.8,[1]*8,1,False)
        r.update(.1,1.01,1.01,[1.01]*8,2,False)
        with self.assertRaisesRegex(RuntimeError,'frequent'):
            r.update(9,2,1.8,[2]*8,3,False)

    def test_total_budget(self):
        r=ObservationRecovery()
        for i in range(3):
            r.update(i*11,1,.8,[1]*8,1,False)
            r.update(i*11+.1,1.01,1.01,[1.01]*8,2,False)
        with self.assertRaisesRegex(RuntimeError,'frequent'):
            r.update(40,2,1.8,[2]*8,3,False)

if __name__=='__main__': unittest.main()
