import unittest
import numpy as np
from sim.a1_defence_switch_probe import join_at_rest
from sim.motion_spline import C2JointSpline


class JoinTests(unittest.TestCase):
    def test_explicit_hold_preserves_both_paths_and_derivatives(self):
        a=C2JointSpline([0.,1.,2.],np.array([[0.]*7,[.2]*7,[.3]*7]))
        b=C2JointSpline([0.,.7,1.],np.array([[.3]*7,[.1]*7,[0.]*7]))
        joined=join_at_rest(a,b,.1)
        self.assertAlmostEqual(joined.duration_s,3.1)
        np.testing.assert_allclose(joined.sample(2.05),.3)
        np.testing.assert_allclose(joined.derivatives(2.05)[0],0.)

    def test_no_implicit_pose_adapter_or_zero_hold(self):
        a=C2JointSpline([0.,1.],np.zeros((2,7)))
        b=C2JointSpline([0.,1.],np.ones((2,7)))
        with self.assertRaises(ValueError):join_at_rest(a,b,.1)
        with self.assertRaises(ValueError):join_at_rest(a,a,0.)


if __name__=='__main__':unittest.main()
