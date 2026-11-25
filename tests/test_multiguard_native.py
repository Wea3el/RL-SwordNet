import unittest

import numpy as np

from sim.motion_spline import C2JointSpline
from sim.multiguard_native import boundary_ready, join_exact, load_action


class MultiGuardNativeTests(unittest.TestCase):
    def test_sequence_has_no_implicit_bridge_and_preserves_curves(self):
        a, b, c = np.ones(7), np.ones(7) + .1, np.ones(7) - .05
        first = C2JointSpline([0., .5, 1.], [a, a + .05, b])
        second = C2JointSpline([0., .7, 1.2], [b, a, c])
        result, segments = join_exact([first, second], .12)
        np.testing.assert_array_equal(result.sample(0.), a)
        np.testing.assert_array_equal(result.sample(result.duration_s), c)
        self.assertEqual(segments, [(0., 1.), (1.12, 2.3200000000000003)])
        for source, (start, _) in zip((first, second), segments):
            ts = np.linspace(0., source.duration_s, 41)
            np.testing.assert_allclose(result.sample(ts + start), source.sample(ts), rtol=0, atol=2e-12)

    def test_disconnected_or_bad_pause_is_rejected(self):
        first = C2JointSpline([0., 1.], [np.ones(7), np.ones(7) + .1])
        second = C2JointSpline([0., 1.], [np.ones(7), np.ones(7) + .2])
        with self.assertRaises(ValueError):
            join_exact([first, second])
        for pause in (0., -1., True, float('nan')):
            with self.subTest(pause=pause), self.assertRaises(ValueError):
                join_exact([first], pause)

    def test_unscreened_source_cannot_be_run(self):
        for flag in (False, 'false', 1, None):
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                load_action({'actions': {'failed': {'candidate_pass': flag}}}, 'failed')

    def test_scheduled_pause_does_not_prove_measured_settling(self):
        ts = np.linspace(0, .2, 201)
        q = np.ones((201, 7)); v = np.zeros_like(q)
        self.assertTrue(boundary_ready(ts, q, v, q[0], .05, .2)['ready'])
        v[:150] = .2
        result = boundary_ready(ts, q, v, q[0], .05, .2)
        self.assertFalse(result['ready'])
        self.assertAlmostEqual(result['measured_stable_duration_s'], .05)


if __name__ == '__main__':
    unittest.main()
