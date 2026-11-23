import unittest

import numpy as np

from sim.fencing_rules import BladeSection, point_cylinder_distance
from tools.audit_target_candidates import candidate_geometry, sphere_clearances, summarize_samples


class TargetAuditTests(unittest.TestCase):
    def test_arbitrary_candidates_preserve_xy_and_mirror(self):
        geometry = candidate_geometry([[1.43, 0, 1.05], [.32, 0, 1.05]])
        self.assertEqual(set(geometry), {'original_ball', 'compact_torso_probe',
            'wide_torso_probe', 'compact_plus_head_probe'})
        centers, radii = geometry['compact_plus_head_probe']
        np.testing.assert_array_equal(centers[0, :, 0], [1.43]*4)
        np.testing.assert_array_equal(centers[1, :, 0], [.32]*4)
        np.testing.assert_array_equal(centers[:, :, 2], [[.9,1.05,1.2,1.45]]*2)
        np.testing.assert_array_equal(radii, [.12,.12,.12,.1])

    def test_vectorized_distance_matches_production_geometry_and_flat_caps(self):
        start = np.array([[[0.,0.,0.], [0.,0.,2.]]])
        end = np.array([[[1.,0.,0.], [1.,0.,2.]]])
        radii = np.array([[.1,.15]])
        centers = np.array([[[1.2,0,0], [.5,.13,0], [1.1,.2,0]]])
        targets = np.array([.05,.05,.05])
        result = sphere_clearances(start,end,radii,centers,targets)[0]
        expected = [min(point_cylinder_distance(c,BladeSection(a,b,r))[0]
            for a,b,r in zip(start[0],end[0],radii[0]))-.05 for c in centers[0]]
        np.testing.assert_allclose(result, expected, atol=1e-14)
        self.assertAlmostEqual(result[0], .15)  # Not extended capsule-end .05.
        self.assertLess(result[1], 0.)

    def test_union_occupancy_is_not_scored_repeatedly(self):
        result = summarize_samples(np.arange(6.), [.1,-.1,-.2,.1,-.1,.2], [.1]*6,
                                   first_recorded_entry_for_attacker_s=2.5)
        self.assertEqual([e['first_inside_saved_time_s'] for e in result['sampled_entry_episodes']], [1.,4.])
        self.assertEqual(len(result['sampled_entry_episodes_after_first_recorded_attacker_entry']), 1)
        self.assertEqual(result['first_new_inside_vs_original_saved_time_s'], 1.)
        self.assertTrue(result['adds_hit_to_original_no_sampled_hit'])

    def test_initial_overlap_not_misrepresented_as_new_entry(self):
        result = summarize_samples([0.,1.,2.], [-.1,-.1,.1], [.1,.1,.1])
        self.assertTrue(result['initial_overlap_saved_frame'])
        self.assertEqual(result['sampled_entry_episodes'], [])
        self.assertEqual(result['first_inside_saved_time_s'], 0.)

    def test_discrete_negative_never_certifies_continuous_miss(self):
        result = summarize_samples([0.,1.], [.01,.01], [.01,.01])
        self.assertIsNone(result['first_inside_saved_time_s'])
        self.assertFalse(result['negative_result_is_continuous_miss_certificate'])

    def test_post_fault_geometry_cannot_be_presented_as_match_success(self):
        result = summarize_samples([0.,1.,2.], [.1,.1,-.1], [.1]*3,
                                   source_fault_or_foul_s=1.5)
        self.assertEqual(result['first_occupied_saved_time_at_or_after_source_fault_or_foul_s'], 2.)
        self.assertFalse(result['geometric_overlap_is_legal_match_success'])

    def test_opponent_terminal_and_later_shadow_entry_remain_distinct(self):
        result = summarize_samples([0.,1.,2.,3.], [.1,.1,-.1,.1], [.1,.1,-.1,.1],
            first_recorded_entry_for_attacker_s=2., source_official_terminal_time_s=1.)
        self.assertEqual(result['first_recorded_entry_for_attacker_s'], 2.)
        self.assertEqual(result['source_official_terminal_time_s'], 1.)
        self.assertTrue(result['source_first_recorded_attacker_entry_after_official_terminal'])
        self.assertEqual(len(result['sampled_entry_episodes_after_source_official_terminal']), 1)
        self.assertEqual(result['sampled_entry_episodes_after_first_recorded_attacker_entry'], [])
        self.assertFalse(result['geometric_overlap_is_legal_match_success'])

    def test_invalid_geometry_and_time_rejected(self):
        with self.assertRaises(ValueError):
            candidate_geometry([[1.,2.,3.]])
        with self.assertRaises(ValueError):
            summarize_samples([0.,0.], [1.,1.], [1.,1.])
        with self.assertRaises(ValueError):
            sphere_clearances(np.zeros((1,1,3)), np.zeros((1,1,3)), np.ones((1,1)),
                              np.zeros((1,1,3)), np.ones(1))


if __name__ == '__main__': unittest.main()
