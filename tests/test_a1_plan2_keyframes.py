import json
import unittest

import numpy as np

from sim.assembly import ROOT
from sim.a1_plan2_keyframes import attack_requests, static_dual_check, tip_cell
from sim.limited_duel import build_limited_duel


class Plan2KeyframeTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((ROOT/'config/a1_plan2_keyframes_v1.json').read_text())
        self.scoring = json.loads((ROOT/'config/scoring_spec_v3_proposal.json').read_text())

    def test_six_target_positions_not_six_claimed_cut_directions(self):
        requests = attack_requests(self.cfg, self.scoring, [1., 0., 0.], .79)
        self.assertEqual(len(requests), 6)
        for v in requests.values():
            self.assertEqual(tip_cell(v['requested_tip_m'], False), v['requested_score_cell'])
            self.assertIsNone(v['intended_future_motion_direction'])
            np.testing.assert_allclose(np.array(v['grip_m'])+[.79, 0., 0.], v['requested_tip_m'])

    def test_defender_left_is_mirrored_not_camera_left(self):
        self.assertEqual(tip_cell([1., -.06, .9], False), 'low_left')
        self.assertEqual(tip_cell([0., .06, .9], True), 'low_left')
        self.assertEqual(tip_cell([1., 0., .975], False), 'mid_left')
        self.assertEqual(tip_cell([1., 0., 1.125], True), 'high_left')

    def test_requests_are_not_training_permission(self):
        self.assertFalse(self.cfg['admitted_to_training'])
        self.assertFalse(self.cfg['continuous_path_tested'])
        self.assertFalse(self.cfg['attack_or_defence_validated'])
        self.assertEqual(len(self.cfg['interception_requests']), 3)

    def test_invalid_axis_rejected(self):
        with self.assertRaises(ValueError):
            attack_requests(self.cfg, self.scoring, [2., 0., 0.], .79)

    def test_static_geometry_uses_current_profile_and_angle_valid_guard(self):
        # The old direction-candidate ready pose is deliberately not repaired:
        # it violates the later J3 calibration. Use the explicit current guard.
        bank = json.loads((ROOT/'assets/a1_pose_bank_v1/catalog.json').read_text())
        ready = np.array(bank['guards']['upright_side_ready']['qpos_rad'])
        scene = build_limited_duel(initial_qpos=ready)
        result = static_dual_check(scene, ready, ready, self.scoring)
        for value in result.values():
            self.assertTrue(value['no_illegal_static_overlap'])
            self.assertFalse(value['blade_overlaps_opponent_target_union'])
            self.assertEqual(len(value['opponent_target_sphere_gaps_m']), 3)
            self.assertIsNone(value['entire_blade_intersected_cells'])


if __name__ == '__main__':
    unittest.main()
