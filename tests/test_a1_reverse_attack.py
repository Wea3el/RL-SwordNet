import numpy as np
import unittest

from sim.a1_reverse_attack import reverse_curve
from sim.motion_spline import C2JointSpline


class ReverseReferenceTests(unittest.TestCase):
    def test_time_reversal_preserves_path_and_reverses_velocity(self):
        times = np.array([1.2, 1.8, 3.1, 3.9])
        q = np.array([[0.1, 0.3, 0.4, 0.8, 0.9, 0.2, 0.1],
                      [0.3, 0.5, 0.2, 0.6, 0.7, 0.4, 0.2],
                      [0.6, 0.7, 0.3, 0.3, 0.5, 0.7, 0.4],
                      [0.9, 0.8, 0.6, 0.2, 0.6, 0.8, 0.5]])
        original = C2JointSpline(times, q)
        reversed_curve = reverse_curve(original)
        sample = np.linspace(times[0], times[-1], 201)
        reverse_time = times[0]+times[-1]-sample
        np.testing.assert_allclose(reversed_curve.sample(sample), original.sample(reverse_time), atol=2e-14)
        rv, ra = reversed_curve.derivatives(sample)
        v, a = original.derivatives(reverse_time)
        np.testing.assert_allclose(rv, -v, atol=2e-13)
        np.testing.assert_allclose(ra, a, atol=2e-12)
        np.testing.assert_allclose(reversed_curve.knot_qpos, q[::-1])
        np.testing.assert_allclose(reverse_curve(reversed_curve).knot_time_s, times)
        np.testing.assert_allclose(reverse_curve(reversed_curve).knot_qpos, q)


    def test_reversal_does_not_wrap_large_joint_coordinates(self):
        q = np.array([[6.1]*7, [6.2]*7, [6.25]*7])
        reversed_curve = reverse_curve(C2JointSpline([0., .4, 1.], q))
        self.assertGreaterEqual(np.min(reversed_curve.sample(np.linspace(0., 1., 101))), 6.1)
        self.assertLessEqual(np.max(reversed_curve.sample(np.linspace(0., 1., 101))), 6.25)
