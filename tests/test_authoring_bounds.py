from dataclasses import replace
import unittest

import numpy as np

from sim.authoring_bounds import intersect_authoring_bounds
from sim.a1_guard_defence_screen import _authoring_bounds as guard_bounds
from sim.a1_high_parry_repair import _authoring_bounds as high_bounds
from sim.joint_constraints import load_joint_constraints, model_joint_ranges_rad


class AuthoringBoundsTests(unittest.TestCase):
    def setUp(self):
        self.constraints = load_joint_constraints()
        self.physical = model_joint_ranges_rad(self.constraints)
        self.entry = self.physical.mean(axis=1)

    def test_intersection_copies_inputs_and_never_widens(self):
        requested = self.physical + np.deg2rad([-10., 10.])
        before = requested.copy()
        result = intersect_authoring_bounds(self.physical, requested)
        np.testing.assert_array_equal(result, self.physical)
        np.testing.assert_array_equal(requested, before)
        result[0, 0] += .1
        np.testing.assert_array_equal(requested, before)

    def test_empty_and_malformed_windows_are_rejected(self):
        requested = self.physical.copy()
        requested[3] = self.physical[3, 1] + np.array([.1, .2])
        with self.assertRaisesRegex(ValueError, 'no interior'):
            intersect_authoring_bounds(self.physical, requested)
        for bad in (self.physical[:6], np.full((7, 2), np.nan), self.physical[:, ::-1]):
            with self.assertRaises(ValueError):
                intersect_authoring_bounds(self.physical, bad)

    def test_current_mapped_j3_upper_bound_cannot_be_overridden(self):
        self.assertLess(np.rad2deg(self.physical[3, 1]), 108.)
        for builder, margin in ((high_bounds, 3.), (guard_bounds, 5.)):
            args = (self.physical, self.entry) + ((25.,) if builder is guard_bounds else ())
            bounds = builder(*args)
            self.assertAlmostEqual(bounds[3, 1], self.physical[3, 1]-np.deg2rad(margin))
            self.assertTrue(np.all(bounds[:, 0] >= self.physical[:, 0]))
            self.assertTrue(np.all(bounds[:, 1] <= self.physical[:, 1]))

    def test_historical_identity_mapping_preserves_the_previous_windows(self):
        old = replace(self.constraints, offset_deg=(0.,)*7)
        physical = model_joint_ranges_rad(old)
        entry = physical.mean(axis=1)
        high = high_bounds(physical, entry)
        guard = guard_bounds(physical, entry, 25.)
        np.testing.assert_allclose(np.rad2deg(high[3]), [-17., 115.], atol=1e-12)
        np.testing.assert_allclose(np.rad2deg(guard[3]), [-15., 115.], atol=1e-12)


if __name__ == '__main__':
    unittest.main()
