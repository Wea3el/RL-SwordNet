"""Numerical motion-graph interpolation contracts; no physical certification."""
import unittest

import numpy as np

from sim.motion_spline import C2JointSpline


def joints(values):
    return np.repeat(np.asarray(values, dtype=float)[:, None], 7, axis=1)


class MotionSplineTests(unittest.TestCase):
    def test_exact_waypoints_and_nonzero_pass_through_speed(self):
        s = C2JointSpline([0., 1., 3., 4.], joints([0., 1., 2., 3.]))
        np.testing.assert_array_equal(s.sample(s.knot_time_s), s.knot_qpos)
        v, a = s.derivatives([1., 3.])
        np.testing.assert_array_equal(v, np.full((2, 7), .5))
        np.testing.assert_array_equal(a, np.zeros((2, 7)))
        self.assertEqual(s.duration_s, 4.)

    def test_reversals_and_flat_intervals_stop_only_the_affected_joint(self):
        q = joints([0., 1., 2., 3., 4.])
        q[:, 0] = [0., 1., 0., 0., -1.]
        s = C2JointSpline([0., 1., 2., 3., 4.], q)
        self.assertEqual(s.derivatives(1.)[0][0], 0.)
        self.assertEqual(s.derivatives(2.)[0][0], 0.)
        self.assertEqual(s.derivatives(3.)[0][0], 0.)
        self.assertEqual(s.derivatives(1.)[0][1], 1.)
        np.testing.assert_array_equal(s.sample(np.linspace(2., 3., 101))[:, 0], np.zeros(101))

    def test_c2_joins_and_endpoint_rest(self):
        s = C2JointSpline([2., 2.7, 4., 5.], joints([0., .8, 1.3, -.2]))
        r = s.continuity_report()
        self.assertLess(r["maximum_position_jump_rad"], 1e-12)
        self.assertLess(r["maximum_velocity_jump_rad_s"], 1e-12)
        self.assertLess(r["maximum_acceleration_jump_rad_s2"], 1e-12)
        self.assertLess(r["endpoint_acceleration_max_rad_s2"], 1e-12)
        for time in (-2., 2., 5., 9.):
            for derivative in s.derivatives(time):
                np.testing.assert_array_equal(derivative, np.zeros(7))
        np.testing.assert_array_equal(s.sample(-2.), s.knot_qpos[0])
        np.testing.assert_array_equal(s.sample(9.), s.knot_qpos[-1])
        # Nearby samples need not be equal: bound their change by the actual
        # acceleration/jerk peak rather than confusing C2 with constant motion.
        peaks = s.analytic_extrema()
        epsilon = 1e-9
        change_bounds = [max(peaks["peak_acceleration_rad_s2"]), max(peaks["peak_jerk_rad_s3"])]
        for t in s.knot_time_s[1:-1]:
            for i in (0, 1):
                np.testing.assert_allclose(s.derivatives(t-epsilon)[i],
                                           s.derivatives(t+epsilon)[i],
                                           atol=2*epsilon*change_bounds[i]+1e-12, rtol=0.)

    def test_no_overshoot_for_nonuniform_multijoint_paths(self):
        rng = np.random.default_rng(90210)
        for _ in range(20):
            times = np.r_[0., np.cumsum(rng.uniform(.05, 2., 9))]
            q = rng.uniform(-2., 2., (10, 7))
            q[4:6, 0] = .2
            s = C2JointSpline(times, q)
            for i in range(9):
                sample = s.sample(np.linspace(times[i], times[i+1], 101))
                lo, hi = np.minimum(q[i], q[i+1]), np.maximum(q[i], q[i+1])
                self.assertTrue(np.all(sample >= lo-1e-13))
                self.assertTrue(np.all(sample <= hi+1e-13))
            r = s.range_report()
            self.assertLessEqual(r["maximum_analytic_overshoot_rad"], 1e-12)
            self.assertLessEqual(r["maximum_control_envelope_violation_rad"], 1e-12)

    def test_monotone_control_polygon_and_bounded_internal_secants(self):
        q = joints([0., .1, 3., 3.01])
        s = C2JointSpline([0., .2, .3, 4.], q)
        self.assertTrue(np.all(np.diff(s.bezier_controls, axis=1) >= -1e-14))
        secants = np.diff(q, axis=0)/np.diff(s.knot_time_s)[:, None]
        np.testing.assert_allclose(s.knot_velocities[1:-1], np.minimum(secants[:-1], secants[1:]))

    def test_analytic_peaks_match_known_single_quintic(self):
        s = C2JointSpline([0., 2.], joints([0., 1.]))
        r = s.analytic_extrema()
        np.testing.assert_allclose(r["peak_speed_rad_s"], np.full(7, 1.875/2), atol=1e-12)
        np.testing.assert_allclose(r["peak_acceleration_rad_s2"], np.full(7, (10/np.sqrt(3))/4), atol=1e-12)
        np.testing.assert_allclose(r["peak_jerk_rad_s3"], np.full(7, 60/8), atol=1e-12)
        np.testing.assert_allclose(r["position_min_rad"], np.zeros(7), atol=1e-12)
        np.testing.assert_allclose(r["position_max_rad"], np.ones(7), atol=1e-12)

    def test_holds_and_reports_do_not_claim_physical_validity(self):
        s = C2JointSpline([0., 1., 3., 4.], joints([0., 1., 1., 2.]))
        r = s.report()
        self.assertEqual(r["hold_duration_s"], 2.)
        self.assertEqual(r["hold_intervals"], [{"start_s": 1., "end_s": 3.}])
        self.assertFalse(r["physical_validation_passed"])
        self.assertFalse(r["range"]["physical_joint_bounds_supplied"])
        np.testing.assert_array_equal(s.sample(np.linspace(1., 3., 17)), np.ones((17, 7)))

    def test_rejects_bad_shape_time_nonfinite_and_nonreal_values(self):
        cases = [([0., 0.], joints([0., 1.])), ([1., 0.], joints([0., 1.])),
                 ([0.], joints([0.])), ([[0., 1.]], joints([0., 1.])),
                 ([0., np.nan], joints([0., 1.])), ([0., np.inf], joints([0., 1.])),
                 ([0., 1.], np.zeros((2, 6))), ([0., 1.], joints([0., np.nan])),
                 ([False, True], joints([0., 1.])), (["0", "1"], joints([0., 1.])),
                 ([0., 1.], np.zeros((2, 7), complex))]
        for times, q in cases:
            with self.subTest(times=times, shape=np.shape(q)), self.assertRaises(ValueError):
                C2JointSpline(times, q)
        with self.assertRaises(ValueError):
            C2JointSpline([0., np.nextafter(0., 1.)], joints([0., 1.]))

    def test_sample_validation_shape_and_input_isolation(self):
        times, q = np.array([0., 1., 2.]), joints([0., .5, 1.])
        s = C2JointSpline(times, q)
        times[1], q[1, 0] = .25, 99.
        self.assertEqual(s.knot_time_s[1], 1.)
        self.assertEqual(s.sample(1.)[0], .5)
        self.assertEqual(s.sample(np.zeros((2, 3))).shape, (2, 3, 7))
        self.assertEqual(s.sample(np.array([])).shape, (0, 7))
        for value in (np.nan, np.inf, "0.1", True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                s.sample(value)
        with self.assertRaises(ValueError):
            s.knot_qpos[0, 0] = 10.


if __name__ == "__main__":
    unittest.main()
