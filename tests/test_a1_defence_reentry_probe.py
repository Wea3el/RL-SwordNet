import unittest

import numpy as np

from sim.a1_defence_reentry_probe import compose_reentry
from sim.motion_spline import C2JointSpline


class ReentryTests(unittest.TestCase):
    def setUp(self):
        self.transfer = C2JointSpline([0., 2.], np.array([[4.]*7, [4.5]*7]))
        self.parry = C2JointSpline([0., 1., 3.], np.array([[4.5]*7, [4.7]*7, [4.5]*7]))

    def test_explicit_wait_and_attack_clock_are_preserved(self):
        curve, wait = compose_reentry(self.transfer, self.parry, 4., .1)
        self.assertAlmostEqual(wait, 1.9)
        self.assertAlmostEqual(curve.duration_s, 6.9)
        ts = np.linspace(0., 3., 51)
        np.testing.assert_allclose(curve.sample(3.9+ts), self.parry.sample(ts), atol=1e-12)
        np.testing.assert_allclose(curve.sample(3.), [4.5]*7)
        np.testing.assert_allclose(curve.derivatives(3.)[0], 0.)
        self.assertGreater(float(curve.knot_qpos.min()), np.pi)  # no hidden wrap

    def test_no_implicit_extension_of_opponent_preparation(self):
        with self.assertRaisesRegex(ValueError, 'Insufficient'):
            compose_reentry(self.transfer, self.parry, 2., .1)

    def test_mismatched_entry_cannot_teleport(self):
        wrong = C2JointSpline([0., 1.], np.array([[4.4]*7, [4.5]*7]))
        with self.assertRaisesRegex(ValueError, 'connector'):
            compose_reentry(self.transfer, wrong, 4., .1)


if __name__ == '__main__':
    unittest.main()
