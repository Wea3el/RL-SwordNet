"""Small plan/identity tests; none is a physical defence certificate."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from sim.constrained_trial import ScheduledReference, hold_reference
from sim.multiguard_defence_probe import (plan_pair, recorded_plan_checks, run,
                                         validate_matched_plan)


def fixture():
    q = np.asarray([3., .7, 3., 1., 2., 1., 0.])
    poses = {'attack_entry': q.tolist(), 'attack_exit': (q+.03).tolist(),
             'defence_entry': (q+.06).tolist(), 'defence_exit': (q-.04).tolist()}
    return {'guards': {name: {'qpos_rad': value} for name, value in poses.items()},
            'actions': {
                'cut': {'candidate_pass': True, 'kind': 'attack', 'target_region': 'left',
                        'start_guard': 'attack_entry', 'end_guard': 'attack_exit',
                        'knot_time_s': [0., 2.],
                        'knot_qpos': [poses['attack_entry'][:], poses['attack_exit'][:]]},
                'parry': {'candidate_pass': True, 'kind': 'defence',
                          'start_guard': 'defence_entry', 'end_guard': 'defence_exit',
                          'knot_time_s': [0., 4.],
                          'knot_qpos': [poses['defence_entry'][:], poses['defence_exit'][:]]}},
            'scoring': {}}


class MultiGuardDefenceProbeTests(unittest.TestCase):
    def test_duration_covers_both_whole_motions_and_two_second_tail(self):
        p = plan_pair(fixture(), 'cut', 'parry', defence_offset_s=3.)
        self.assertEqual(p['attack_start_s'], .75)
        self.assertEqual(p['defence_start_s'], 3.75)
        self.assertEqual(p['duration_s'], 9.75)
        self.assertTrue(all(validate_matched_plan(p).values()))
        for slot in ('a1', 'a2'):
            np.testing.assert_array_equal(p['baseline'][slot].target(0), p['defended'][slot].target(0))
        np.testing.assert_array_equal(p['defended']['a2'].target(100),
                                      fixture()['guards']['defence_exit']['qpos_rad'])

    def test_negative_offset_preserves_early_defence_without_negative_clock(self):
        p = plan_pair(fixture(), 'cut', 'parry', defence_offset_s=-.65)
        self.assertAlmostEqual(p['defence_start_s'], .1)
        self.assertAlmostEqual(p['duration_s'], 6.1)
        for bad in (-.651, -1., True, float('nan'), float('inf')):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                plan_pair(fixture(), 'cut', 'parry', defence_offset_s=bad)

    def test_slot_swap_does_not_relabel_or_negate_joint_coordinates(self):
        a = plan_pair(fixture(), 'cut', 'parry', attacker='a1')
        b = plan_pair(fixture(), 'cut', 'parry', attacker='a2')
        self.assertEqual(b['defender'], 'a1')
        np.testing.assert_array_equal(a['defended']['a1'].curve.knot_qpos,
                                      b['defended']['a2'].curve.knot_qpos)
        np.testing.assert_array_equal(a['defended']['a2'].curve.knot_qpos,
                                      b['defended']['a1'].curve.knot_qpos)
        with self.assertRaises(ValueError): plan_pair(fixture(), 'cut', 'parry', attacker='a3')

    def test_rejects_wrong_roles_unscreened_paths_unknown_targets_and_endpoint_repair(self):
        changes = (
            lambda r: r['actions']['parry'].update(kind='attack'),
            lambda r: r['actions']['parry'].update(candidate_pass='false'),
            lambda r: r['actions']['cut'].update(target_region='unknown'),
            lambda r: r['actions']['cut']['knot_qpos'][0].__setitem__(0, 3.2),
            lambda r: r['actions']['parry']['knot_qpos'][1].__setitem__(1, -1.),
        )
        for change in changes:
            r = fixture(); change(r)
            with self.subTest(change=change), self.assertRaises(ValueError):
                plan_pair(r, 'cut', 'parry')

    def test_tampered_baseline_attack_or_guard_is_rejected(self):
        for mismatch in ('attacker_clock', 'defender_clock', 'defender_guard', 'short_window'):
            p = plan_pair(fixture(), 'cut', 'parry')
            if mismatch == 'attacker_clock':
                p['baseline']['a1'] = ScheduledReference(p['baseline']['a1'].curve, .8)
            elif mismatch == 'defender_clock':
                p['baseline']['a2'] = ScheduledReference(p['baseline']['a2'].curve, .8)
            elif mismatch == 'defender_guard':
                p['baseline']['a2'] = hold_reference(np.ones(7), .75, 4.)
            else:
                p['duration_s'] = 5.
            with self.subTest(mismatch=mismatch), self.assertRaises(ValueError):
                validate_matched_plan(p)

    def test_baseline_cannot_move_while_claiming_to_hold(self):
        p = plan_pair(fixture(), 'cut', 'parry')
        p['baseline']['a2'] = p['defended']['a2']
        with self.assertRaises(ValueError): validate_matched_plan(p)

    def test_recorded_references_are_bound_to_prescribed_program(self):
        p = plan_pair(fixture(), 'cut', 'parry')
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            summary = {'program_start_s': {s: prog.start_s for s, prog in p['baseline'].items()}}
            (folder/'summary.json').write_text(json.dumps(summary))
            arrays = {s+'_'+key: getattr(prog.curve, key) for s, prog in p['baseline'].items()
                      for key in ('knot_time_s', 'knot_qpos', 'coefficients')}
            np.savez_compressed(folder/'references.npz', **arrays)
            self.assertTrue(all(recorded_plan_checks(folder, p['baseline']).values()))
            arrays['a1_knot_qpos'] = arrays['a1_knot_qpos'].copy()
            arrays['a1_knot_qpos'][1, 0] += .01
            np.savez_compressed(folder/'references.npz', **arrays)
            checks = recorded_plan_checks(folder, p['baseline'])
            self.assertFalse(checks['a1_recorded_curve_matches_plan'])

    def test_existing_output_is_preserved_without_running_physics(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            sentinel = folder/'kept.txt'; sentinel.write_text('do not overwrite')
            with self.assertRaises(FileExistsError):
                run(folder/'missing_source', folder, attack_name='cut', defence_name='parry')
            self.assertEqual(sentinel.read_text(), 'do not overwrite')


if __name__ == '__main__':
    unittest.main()
