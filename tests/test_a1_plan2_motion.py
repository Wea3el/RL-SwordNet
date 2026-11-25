import unittest

import numpy as np

from sim.a1_plan2_motion import closed_curve, phase_at


class Plan2MotionTests(unittest.TestCase):
    def test_exact_closed_ready_and_rest_endpoints(self):
        ready = np.ones(7)
        curve = closed_curve([0.,2.,3.,4.,6.], [ready,ready+.1,ready+.2,ready+.3,ready], ready)
        np.testing.assert_array_equal(curve.sample(0.), ready)
        np.testing.assert_array_equal(curve.sample(6.), ready)
        for t in (0.,6.):
            velocity, acceleration = curve.derivatives(t)
            np.testing.assert_array_equal(velocity, np.zeros(7))
            np.testing.assert_array_equal(acceleration, np.zeros(7))

    def test_through_strike_not_stop_and_reverse(self):
        ready = np.ones(7)
        curve = closed_curve([0.,2.,3.,4.,6.], [ready,ready+.1,ready+.2,ready+.3,ready], ready)
        self.assertTrue(np.all(curve.derivatives(3.)[0] > 0))
        self.assertLess(curve.continuity_report()['maximum_velocity_jump_rad_s'], 1e-12)
        self.assertLess(curve.continuity_report()['maximum_acceleration_jump_rad_s2'], 1e-12)

    def test_reject_different_exit_ready(self):
        with self.assertRaises(ValueError):
            closed_curve([0.,1.], [np.zeros(7),np.ones(7)], np.zeros(7))

    def test_early_entry_not_relabelled_as_strike(self):
        self.assertEqual(phase_at(2.999,3.,4.7,6.5), 'approach')
        self.assertEqual(phase_at(3.,3.,4.7,6.5), 'strike_or_hold')
        self.assertEqual(phase_at(5.,3.,4.7,6.5), 'withdraw')
        self.assertEqual(phase_at(7.,3.,4.7,6.5), 'return_ready')


if __name__ == '__main__':
    unittest.main()
