import unittest
import numpy as np

from tools.audit_scoring_reachability import sample_joint_space, primitive_signed_distance, segment_bound_intersects


class ScoringReachabilityTests(unittest.TestCase):
    def test_reproducible_bounded_independent_joint_samples(self):
        ranges=np.tile([0.,2.],(7,1));ready=np.full(7,1.)
        q,labels,proposals=sample_joint_space(ranges,ready,count=100,seed=17,margin=.05)
        again,_,_=sample_joint_space(ranges,ready,count=100,seed=17,margin=.05)
        np.testing.assert_array_equal(q,again)
        np.testing.assert_array_equal(q[0],ready)
        self.assertEqual(q.shape,(100,7));self.assertTrue(np.all((q>=.05)&(q<=1.95)))
        self.assertEqual(np.count_nonzero(labels=='global_uniform'),50)
        self.assertGreater(proposals,0)

    def test_invalid_range_margin_rejected_not_clipped(self):
        with self.assertRaises(ValueError):
            sample_joint_space(np.tile([0.,1.],(7,1)),np.zeros(7),margin=.05)

    def test_exact_primitive_distance_interior_and_exterior(self):
        self.assertAlmostEqual(primitive_signed_distance([0,0,0],'box',[1,2,3]),-1.)
        self.assertAlmostEqual(primitive_signed_distance([2,0,0],'box',[1,2,3]),1.)
        self.assertAlmostEqual(primitive_signed_distance([0,0,2],'capsule',[.1,1.,0]),.9)
        self.assertAlmostEqual(primitive_signed_distance([0,0,2],'cylinder',[.1,1.,0]),1.)
        self.assertAlmostEqual(primitive_signed_distance([0,0,0],'sphere',[.1,0,0]),-.1)

    def test_conservative_line_bound_is_only_a_geometric_screen(self):
        self.assertTrue(segment_bound_intersects([0,0,0],[2,0,0],[1,.1,0],.2))
        self.assertFalse(segment_bound_intersects([0,0,0],[2,0,0],[1,.3,0],.2))
        self.assertFalse(segment_bound_intersects([0,0,0],[2,0,0],[3,0,0],.2))


if __name__=='__main__':unittest.main()
