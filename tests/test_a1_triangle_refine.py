import json
import unittest

import numpy as np

from sim.a1_triangle_refine import refinement_waypoints, robustness_metrics, _distinct_seeds, _json_ready


class TriangleRefinementTests(unittest.TestCase):
    def test_upward_follow_does_not_force_old_unreachable_extension(self):
        center = np.array([1.43, 0., 1.22])
        for name, sign in [('cut_up_upper_from_left', -1), ('cut_up_upper_from_right', 1)]:
            points, direction = refinement_waypoints(center, name)
            np.testing.assert_allclose(points[1], [1.35, 0., 1.18])
            self.assertLess(points[2, 0], points[1, 0])
            self.assertLess(points[2, 2], 1.2719)
            self.assertGreater(points[2, 2], points[1, 2])
            self.assertGreater(sign*points[2, 1], 0)
            np.testing.assert_array_equal(direction, [0, sign, 1])

    def test_straight_down_can_prepare_high_but_closer(self):
        points, direction = refinement_waypoints([1.43, 0, 1.22], 'cut_straight_down_upper')
        np.testing.assert_allclose(points[0], [1.20, 0, 1.34])
        self.assertGreater(points[0, 2], points[1, 2])
        self.assertLess(points[0, 0], points[1, 0])
        self.assertLess(points[2, 2], points[1, 2])
        np.testing.assert_array_equal(direction, [0, 0, -1])

    def test_witness_is_not_a_changed_scoring_center(self):
        for name in ('thrust_upper', 'cut_up_upper_from_left', 'cut_straight_down_upper'):
            center = np.array([1.43, 0, 1.22])
            before = center.copy()
            points, _ = refinement_waypoints(center, name)
            np.testing.assert_array_equal(center, before)
            self.assertLess(np.linalg.norm(points[1]-center), .105)

    def test_nominal_success_with_old_shallow_margin_is_not_robust(self):
        result = {'sampled_reference_pass': True, 'true_min_joint_margin_rad': .01,
                  'slots': {s: {'minimum_region_gaps_m': {'upper': -.016}} for s in ('a1', 'a2')}}
        metrics = robustness_metrics(result)
        self.assertFalse(metrics['robustness_pass'])
        self.assertFalse(metrics['joint_margin_pass'])
        self.assertFalse(metrics['penetration_pass'])
        self.assertTrue(metrics['diagnostic_robustness_assumptions'])
        self.assertTrue(metrics['non_admission_guarantee'])

    def test_both_slots_and_reference_success_required(self):
        result = {'sampled_reference_pass': True, 'true_min_joint_margin_rad': .05,
                  'slots': {'a1': {'minimum_region_gaps_m': {'upper': -.04}},
                            'a2': {'minimum_region_gaps_m': {'upper': -.02}}}}
        self.assertFalse(robustness_metrics(result)['robustness_pass'])
        result['slots']['a2']['minimum_region_gaps_m']['upper'] = -.04
        self.assertTrue(robustness_metrics(result)['robustness_pass'])
        result['sampled_reference_pass'] = False
        self.assertFalse(robustness_metrics(result)['robustness_pass'])

    def test_duplicate_seeds_are_not_counted_as_independent_trials(self):
        self.assertEqual(len(_distinct_seeds([np.zeros(7), np.zeros(7), np.ones(7)])), 2)

    def test_guard_repair_solver_arrays_and_failure_values_are_serializable(self):
        evidence = {'solver_result': {'qpos': np.zeros(7), 'score': np.float64(np.inf)},
                    'fit_pass': np.bool_(False)}
        saved = json.loads(json.dumps(_json_ready(evidence), allow_nan=False))
        self.assertEqual(saved['solver_result']['qpos'], [0.]*7)
        self.assertIsNone(saved['solver_result']['score'])
        self.assertIs(saved['fit_pass'], False)

    def test_invalid_waypoints_are_rejected(self):
        for center in ([1, 2], [1, 2, float('nan')]):
            with self.assertRaises(ValueError):
                refinement_waypoints(center, 'thrust_upper')
        with self.assertRaises(ValueError):
            refinement_waypoints([1, 2, 3], 'unapproved')


if __name__ == '__main__':
    unittest.main()
