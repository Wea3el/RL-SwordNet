"""Bounded manual-J0 program contracts; no native physics or GUI required."""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from sim import manual_j0_trial as trial
from sim.joint_constraints import load_joint_constraints
from sim.motion_spline import C2JointSpline


class ManualJ0TrialTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.constraints = load_joint_constraints()
        self.display = np.deg2rad(np.asarray(self.constraints.bounds_deg).mean(axis=1))
        self.guard = trial.choose_pose(self.display, self.constraints)
        middle = self.guard.copy()
        middle[1] += .1
        self.attack = C2JointSpline([0., 1., 2.], [self.guard, middle, self.guard])
        source = self.root / 'source.json'
        self.scoring = {'base_distance': 1.75, 'regions': [], 'fixture_only': True}
        source.write_text(json.dumps({'scoring': self.scoring, 'actions': {trial.ATTACK: {
            'kind': 'attack',
            'knot_time_s': self.attack.knot_time_s.tolist(),
            'knot_qpos': self.attack.knot_qpos.tolist(), 'target_region': 'upper'}}}))
        reference = self.root / 'manual_reference.json'
        reference.write_text('{"fixture_only": true}')
        self.source = source
        self.reference = reference
        self.stack.enter_context(patch.object(trial, 'SOURCE', source))
        self.stack.enter_context(patch.object(trial, 'DEFAULT_REFERENCE', reference))
        self.stack.enter_context(patch.object(trial, 'load_reference', return_value=(
            {'facing_variant': {'base_yaw_delta_rad': 0.}}, self.display, self.guard)))
        self.stack.enter_context(patch('mujoco.mj_step', side_effect=AssertionError('No native physics')))

    def test_same_attack_initial_state_and_conditions_in_both_slots(self):
        for attacker in ('a1', 'a2'):
            with self.subTest(attacker=attacker):
                base, horizon, bm, bs = trial.make_programs('baseline', attacker=attacker)
                moved, other_horizon, dm, ds = trial.make_programs('defended', attacker=attacker)
                self.assertEqual(horizon, other_horizon)
                self.assertEqual(bs, ds)
                self.assertEqual({k: v for k, v in bm.items() if k != 'variant'},
                                 {k: v for k, v in dm.items() if k != 'variant'})
                for slot in ('a1', 'a2'):
                    np.testing.assert_array_equal(base[slot].target(0.), moved[slot].target(0.))
                for field in ('knot_time_s', 'knot_qpos', 'coefficients'):
                    np.testing.assert_array_equal(getattr(base[attacker].curve, field),
                                                  getattr(self.attack, field))
                    np.testing.assert_array_equal(getattr(moved[attacker].curve, field),
                                                  getattr(self.attack, field))
                self.assertEqual(base[attacker].start_s, .75)

    def test_only_defender_j0_reference_changes_and_returns(self):
        for attacker in ('a1', 'a2'):
            for delta in (-.125, .125):
                with self.subTest(attacker=attacker, delta=delta):
                    base, horizon, _, _ = trial.make_programs('baseline', attacker=attacker, delta_rad=delta)
                    moved, _, _, _ = trial.make_programs('defended', attacker=attacker, delta_rad=delta)
                    defender = 'a2' if attacker == 'a1' else 'a1'
                    curve = moved[defender].curve
                    np.testing.assert_array_equal(curve.knot_qpos[:, 1:],
                        np.tile(self.guard[1:], (len(curve.knot_time_s), 1)))
                    self.assertAlmostEqual(curve.knot_qpos[1, 0]-self.guard[0], delta)
                    for t in np.linspace(0., horizon, 51):
                        np.testing.assert_allclose(moved[defender].target(t)[1:],
                                                   base[defender].target(t)[1:], atol=1e-12, rtol=0.)
                    np.testing.assert_array_equal(curve.knot_qpos[-1], self.guard)
                    np.testing.assert_array_equal(moved[defender].target(horizon), self.guard)

    def test_whole_return_and_two_second_tail_are_preserved(self):
        for approach in (.5, 1.):
            programs, horizon, metadata, _ = trial.make_programs('defended', approach_s=approach)
            spec = metadata['planned_defender_timing']
            self.assertAlmostEqual(spec['release_s'], programs['a1'].end_s+.2)
            self.assertAlmostEqual(programs['a2'].end_s, spec['release_s']+approach)
            self.assertGreaterEqual(horizon-max(p.end_s for p in programs.values()), 2.-1e-12)
            self.assertAlmostEqual(horizon/.001, round(horizon/.001))

    def test_default_four_knot_route_is_preserved(self):
        programs, horizon, metadata, _ = trial.make_programs('defended')
        curve = programs['a2'].curve
        np.testing.assert_allclose(curve.knot_time_s, [0., 1., 2.2, 3.2], atol=1e-12, rtol=0.)
        np.testing.assert_allclose(curve.knot_qpos[:, 0]-self.guard[0],
                                   [0., -.125, -.125, 0.], atol=1e-12, rtol=0.)
        self.assertEqual(programs['a2'].start_s, .75)
        self.assertEqual(metadata['planned_defender_timing']['approach_s'], 1.)
        self.assertNotIn('route', metadata['planned_defender_timing'])
        self.assertGreaterEqual(horizon-programs['a2'].end_s, 2.-1e-12)

    def test_explicit_route_matches_baseline_and_only_moves_j0_in_both_slots(self):
        route = {'time_s': [0., .8, 1.8, 6.5, 7.5],
                 'offset_rad': [0., -.125, 0., 0., 0.]}
        for attacker in ('a1', 'a2'):
            with self.subTest(attacker=attacker):
                base, horizon, bm, bs = trial.make_programs('baseline', attacker=attacker, route=route)
                moved, other_horizon, dm, ds = trial.make_programs('defended', attacker=attacker, route=route)
                defender = 'a2' if attacker == 'a1' else 'a1'
                self.assertEqual(horizon, other_horizon)
                self.assertEqual(bs, ds)
                self.assertEqual({k: v for k, v in bm.items() if k != 'variant'},
                                 {k: v for k, v in dm.items() if k != 'variant'})
                for slot in ('a1', 'a2'):
                    np.testing.assert_array_equal(base[slot].target(0.), moved[slot].target(0.))
                for field in ('knot_time_s', 'knot_qpos', 'coefficients'):
                    np.testing.assert_array_equal(getattr(moved[attacker].curve, field),
                                                  getattr(self.attack, field))
                curve = moved[defender].curve
                np.testing.assert_array_equal(curve.knot_time_s, route['time_s'])
                np.testing.assert_array_equal(curve.knot_qpos[:, 1:],
                    np.tile(self.guard[1:], (len(route['time_s']), 1)))
                np.testing.assert_allclose(curve.knot_qpos[:, 0]-self.guard[0],
                                           route['offset_rad'], atol=1e-12, rtol=0.)
                np.testing.assert_array_equal(moved[defender].target(.749), self.guard)
                np.testing.assert_array_equal(curve.sample(0.), self.guard)
                np.testing.assert_array_equal(curve.sample(curve.duration_s), self.guard)
                for endpoint in (0., curve.duration_s):
                    velocity, acceleration = curve.derivatives(endpoint)
                    np.testing.assert_array_equal(velocity, np.zeros(7))
                    np.testing.assert_array_equal(acceleration, np.zeros(7))
                for knot in curve.knot_time_s[1:-1]:
                    for order in (0, 1, 2):
                        left = curve.sample(knot-1e-7) if order == 0 else curve.derivatives(knot-1e-7)[order-1]
                        right = curve.sample(knot+1e-7) if order == 0 else curve.derivatives(knot+1e-7)[order-1]
                        np.testing.assert_allclose(left, right, atol=1e-5, rtol=0.)
                self.assertGreaterEqual(horizon-max(p.end_s for p in moved.values()), 2.-1e-12)

    def test_explicit_route_rejects_bad_shape_numbers_clocks_and_teleports(self):
        bad_routes = [
            [], {}, {'time_s': [0., 1.], 'offset_rad': [0., 0.], 'extra': 1},
            {'time_s': 0., 'offset_rad': [0., -.1, 0.]},
            {'time_s': [[0., 1., 2.]], 'offset_rad': [0., -.1, 0.]},
            {'time_s': [0., 1., 2.], 'offset_rad': [0., 0.]},
            {'time_s': [0., True, 2.], 'offset_rad': [0., -.1, 0.]},
            {'time_s': [0., 1., 2.], 'offset_rad': [0., True, 0.]},
            {'time_s': [0., float('nan'), 2.], 'offset_rad': [0., -.1, 0.]},
            {'time_s': [0., 1., 2.], 'offset_rad': [0., float('inf'), 0.]},
            {'time_s': [0., 1., 1.], 'offset_rad': [0., -.1, 0.]},
            {'time_s': [0., 2., 1.], 'offset_rad': [0., -.1, 0.]},
            {'time_s': [.1, 1., 2.], 'offset_rad': [0., -.1, 0.]},
            {'time_s': [0., 1., 2.], 'offset_rad': [-.1, -.1, 0.]},
            {'time_s': [0., 1., 2.], 'offset_rad': [0., -.1, -.1]},
            {'time_s': [0., 1., 2.], 'offset_rad': [0., 0., 0.]},
            {'time_s': [0., 10., 20.], 'offset_rad': [0., 4., 0.]},
            {'time_s': [0., .001, .002], 'offset_rad': [0., -.125, 0.]},
            {'time_s': [0., 30., 60.], 'offset_rad': [0., -.125, 0.]},
        ]
        for variant in ('baseline', 'defended'):
            for route in bad_routes:
                with self.subTest(variant=variant, route=route), self.assertRaises(ValueError):
                    trial.make_programs(variant, route=route)

    def test_attack_selection_requires_an_existing_attack(self):
        source = json.loads(self.source.read_text())
        alternate = dict(source['actions'][trial.ATTACK], target_region='left')
        source['actions']['alternate_cut'] = alternate
        source['actions']['not_attack'] = dict(alternate, kind='defence')
        self.source.write_text(json.dumps(source))
        _, _, metadata, _ = trial.make_programs('defended', attack_name='alternate_cut')
        self.assertEqual(metadata['attacker_action'], 'alternate_cut')
        self.assertEqual(metadata['target_region'], 'left')
        for name in ('missing', 'not_attack'):
            with self.subTest(attack_name=name), self.assertRaises(ValueError):
                trial.make_programs('baseline', attack_name=name)

    def test_invalid_inputs_and_invalid_moving_counterpart_are_rejected(self):
        for variant in ('baseline', 'defended'):
            for delta in (0., float('nan'), float('inf'), -float('inf'), True, False, 10.):
                with self.subTest(variant=variant, delta=delta), self.assertRaises(ValueError):
                    trial.make_programs(variant, delta_rad=delta)
            for duration in (0., -1., float('nan'), float('inf'), True, False, .001, 3.):
                with self.subTest(variant=variant, duration=duration), self.assertRaises(ValueError):
                    trial.make_programs(variant, approach_s=duration)
        for kwargs in ({'variant': 'other'}, {'variant': 'baseline', 'attacker': 'other'}):
            with self.assertRaises(ValueError):
                trial.make_programs(**kwargs)

    def test_metadata_identifies_manual_offline_probe_not_policy_or_admission(self):
        _, _, metadata, _ = trial.make_programs('defended', attacker='a2')
        self.assertEqual(metadata['attacker_slot'], 'a2')
        self.assertEqual(metadata['defender_slot'], 'a1')
        self.assertTrue(metadata['manual_reference_is_rounded_screenshot'])
        self.assertTrue(metadata['uses_known_clock_for_offline_probe_only'])
        self.assertTrue(metadata['only_defender_command_J0_changes'])
        self.assertFalse(metadata['policy_reaction_demonstrated'])
        self.assertFalse(metadata['admitted_to_training'])
        self.assertEqual(metadata['source_attack_sha256'], hashlib.sha256(self.source.read_bytes()).hexdigest())
        self.assertEqual(metadata['manual_reference_sha256'], hashlib.sha256(self.reference.read_bytes()).hexdigest())

    def test_review_rejects_late_hit_after_complete_attack(self):
        root = self.root / 'review'
        for variant in ('baseline', 'defended'):
            _, horizon, metadata, scoring = trial.make_programs(variant)
            folder = root / variant
            native = folder / 'native'
            native.mkdir(parents=True)
            (folder / 'plan.json').write_text(json.dumps(dict(metadata,
                requested_duration_s=horizon, scoring=scoring)))
            (native / 'summary.json').write_text(json.dumps({
                'physics_steps': 1, 'program_start_s': {'a1': .75}}))
        np.savez(root / 'baseline/native/references.npz', a1_knot_time_s=[0., 1., 2.])
        with patch.object(trial, 'compare', return_value={
                'status': 'qualified', 'successful_physical_defence': True,
                'baseline_first_triangle_hit_s': 2.8}):
            result = trial.review(root)
        self.assertEqual(result['baseline_attack_window_s'], [.75, 2.75])
        self.assertEqual(result['status'], 'no_threat_in_attack_window')
        self.assertFalse(result['baseline_first_hit_in_attack_window'])
        self.assertFalse(result['successful_physical_defence'])
        with self.assertRaises(FileExistsError):
            trial.review(root)


if __name__ == '__main__':
    unittest.main()
