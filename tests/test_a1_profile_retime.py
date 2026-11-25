import copy
import unittest

import numpy as np

from sim.a1_profile_retime import ACTIONS, SCALES, full_window_s, retimed_bank, scaled_curve, scan_scales


def action():
    return {'knot_time_s': [0., .5, 1.5], 'knot_qpos':
            [[0.]*7, [.2]*7, [.5]*7], 'candidate_pass': True,
            'sampled_reference_pass': True, 'start_guard': 'a', 'end_guard': 'b',
            'audit': {'old_native': True}, 'windup_end_s': .25,
            'active_interval_s': [.3, 1.2], 'admitted_to_training': False}


class ProfileRetimingTests(unittest.TestCase):
    def test_uniform_scaling_preserves_geometry_and_scales_derivatives(self):
        old, new = scaled_curve(action(), 1.), scaled_curve(action(), 2.5)
        times = np.linspace(0., old.duration_s, 73)
        np.testing.assert_allclose(new.sample(times*2.5), old.sample(times), atol=1e-14)
        ov, oa = old.derivatives(times); nv, na = new.derivatives(times*2.5)
        np.testing.assert_allclose(nv*2.5, ov, atol=1e-14)
        np.testing.assert_allclose(na*2.5**2, oa, atol=1e-13)
        np.testing.assert_array_equal(new.knot_qpos, old.knot_qpos)

    def test_all_scales_kept_and_lowest_passing_selected(self):
        periods = []
        def screen(curve, period):
            periods.append(period)
            ratio = 4./(curve.duration_s/1.5)**2
            return {'worst_utilization': ratio,
                    'ideal_demand_within_caps_at_samples': ratio <= 1.}
        result = scan_scales(action(), screen)
        self.assertEqual(result['selected_scale'], 2.)
        self.assertEqual(len(result['trials']), 7)
        np.testing.assert_allclose(periods, np.asarray(SCALES)*.02)
        self.assertFalse(result['native_tested'])

    def test_no_pass_and_exceptions_preserved_not_clipped(self):
        def fails(curve, period):
            raise ValueError('joint range')
        result = scan_scales(action(), fails)
        self.assertIsNone(result['selected_scale'])
        self.assertEqual([x['error'] for x in result['trials']], ['joint range']*7)

    def test_complete_window_cap_does_not_truncate_reference(self):
        row = action(); row['knot_time_s'] = [0., 10., 100.]
        result = scan_scales(row, lambda c, p: {
            'worst_utilization': 0., 'ideal_demand_within_caps_at_samples': True})
        self.assertIsNone(result['selected_scale'])
        self.assertEqual(result['trials'][0]['duration_s'], 100.)
        self.assertEqual(result['trials'][0]['minimum_complete_native_window_s'], 102.75)
        self.assertEqual(full_window_s(1.), 6.)

    def test_only_requested_action_records_change_and_no_certificate_inherited(self):
        original = {'actions': {n: action() for n in ACTIONS},
                    'guards': {'a': {'qpos_rad': [0.]*7}}, 'scoring': {'v': 2},
                    'blade_geometry': {'sha256': 'test'}, 'limitations': []}
        original['actions']['keep'] = action()
        snapshot = copy.deepcopy(original)
        scan = scan_scales(action(), lambda c, p: {'worst_utilization':
              1.5/(c.duration_s/1.5), 'ideal_demand_within_caps_at_samples': c.duration_s >= 2.25})
        bank = retimed_bank(original, {n: scan for n in ACTIONS})
        self.assertEqual(original, snapshot)
        self.assertEqual(bank['actions']['keep'], original['actions']['keep'])
        self.assertEqual(bank['guards'], original['guards'])
        self.assertEqual(bank['scoring'], original['scoring'])
        self.assertEqual(bank['collision_geometry_sha256'], 'test')
        for name in ACTIONS:
            row = bank['actions'][name]
            self.assertEqual(row['knot_qpos'], original['actions'][name]['knot_qpos'])
            self.assertEqual(row['windup_end_s'], .4)
            np.testing.assert_allclose(row['active_interval_s'], [.48, 1.92])
            self.assertTrue(row['candidate_pass'])
            self.assertFalse(row['native_dynamics_tested'])
            self.assertFalse(row['admitted_to_training'])
            self.assertFalse(row['audit']['whole_path_native_tested'])

    def test_all_failed_action_not_left_eligible(self):
        original = {'actions': {n: action() for n in ACTIONS}, 'guards': {},
                    'scoring': {}, 'blade_geometry': {'sha256': 'test'}}
        scan = scan_scales(action(), lambda c, p: {
            'worst_utilization': 1.01, 'ideal_demand_within_caps_at_samples': False})
        bank = retimed_bank(original, {n: scan for n in ACTIONS})
        for row in bank['actions'].values():
            self.assertFalse(row['candidate_pass'])
            self.assertEqual(row['knot_time_s'], [0., .5, 1.5])
        self.assertEqual(bank['outgoing_actions'], {})

    def test_invalid_clock_scale_and_tail_rejected(self):
        for s in (0., -1., float('nan'), True):
            with self.assertRaises(ValueError):
                scaled_curve(action(), s)
        with self.assertRaises(ValueError):
            full_window_s(1., tail_s=1.)
        with self.assertRaises(ValueError):
            scan_scales(action(), lambda c, p: {}, scales=[1., 1.])


if __name__ == '__main__':
    unittest.main()
