"""Contact-shift overlays preserve the base experiment, without native physics."""
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from sim import manual_contact_shift_trial as trial
from sim.constrained_trial import ScheduledReference
from sim.joint_constraints import hardware_to_model_rad, load_joint_constraints
from sim.motion_spline import C2JointSpline


class ManualContactShiftTrialTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        constraints = load_joint_constraints()
        self.guard = hardware_to_model_rad(np.asarray(constraints.bounds_deg).mean(axis=1), constraints)
        self.times = np.array([0., 2.5, 3., 3.5, 5.5, 13.25])
        self.base_offsets = np.array([[0.]*7, [-.625, -.07, 0., .25, 0., .25, 0.],
            [-.67, -.07, 0., .25, 0., .25, 0.], [-.725, -.07, 0., .25, 0., .25, 0.],
            [0.]*7, [0.]*7])
        self.base_curve = C2JointSpline(self.times, self.guard+self.base_offsets)
        attack_mid = self.guard.copy()
        attack_mid[1] += .1
        self.attack = C2JointSpline([0., 1., 2.], [self.guard, attack_mid, self.guard])
        self.horizon = 16.
        self.scoring = {'fixture_only': True}
        self.coordinated = self.stack.enter_context(patch.object(trial, 'coordinated_programs',
            side_effect=lambda recipe, variant, **kw: self.base_result('coordinated', recipe, variant, **kw)))
        self.wrist = self.stack.enter_context(patch.object(trial, 'wrist_programs',
            side_effect=lambda recipe, variant, **kw: self.base_result('wrist', recipe, variant, **kw)))
        self.stack.enter_context(patch('mujoco.mj_step', side_effect=AssertionError('No native physics')))
        self.source, self.reference = self.root/'source.json', self.root/'reference.json'
        self.base_recipe = {'candidate_id': 'fixture_base', 'fixture': True}

    def base_result(self, kind, recipe, variant, *, attacker, source, reference):
        self.assertEqual(variant, 'defended')
        defender = 'a2' if attacker == 'a1' else 'a1'
        expanded = (copy.deepcopy(recipe) if kind == 'coordinated' else
                    {'wrist': copy.deepcopy(recipe), 'base': {'candidate_id': 'nested_base'}})
        metadata = {'experiment': 'fixture_base', 'recipe': expanded,
            'initial_guard_cad_rad': self.base_curve.sample(0.).tolist(),
            'attacker_slot': attacker, 'defender_slot': defender,
            'attacker_action': 'fixture_attack', 'target_region': 'upper',
            'source_attack_sha256': 'a'*64, 'manual_reference_sha256': 'b'*64,
            'implementation_sha256': 'c'*64, 'admitted_to_training': False}
        if kind == 'wrist':
            metadata['base_J0_J3_reference_exactly_unchanged'] = True
        return {attacker: ScheduledReference(self.attack, .75),
                defender: ScheduledReference(self.base_curve, .75)}, self.horizon, metadata, self.scoring

    @staticmethod
    def recipe(kind='coordinated'):
        return {'schema_version': 1, 'candidate_id': 'shift_fixture',
            'status': 'offline_candidate_not_for_training', 'base_kind': kind,
            'base_recipe': 'config/fixture_base.json',
            'offsets_rad': [[0.]*7, [.02, .01, 0., -.03, 0., .02, 0.],
                            [.03, .015, 0., -.04, 0., .015, 0.],
                            [.01, .015, 0., -.04, 0., .015, 0.], [0.]*7, [0.]*7]}

    def make(self, recipe=None, variant='defended', attacker='a1', base_recipe=None):
        return trial.make_programs(self.recipe() if recipe is None else recipe, variant,
            attacker=attacker, base_recipe=self.base_recipe if base_recipe is None else base_recipe,
            source=self.source, reference=self.reference)

    def test_both_base_kinds_slots_and_variants_preserve_attack_reset_horizon(self):
        for kind in ('coordinated', 'wrist'):
            for attacker in ('a1', 'a2'):
                with self.subTest(kind=kind, attacker=attacker):
                    baseline, bh, bm, bs = self.make(self.recipe(kind), 'baseline', attacker)
                    moved, mh, mm, ms = self.make(self.recipe(kind), 'defended', attacker)
                    defender = 'a2' if attacker == 'a1' else 'a1'
                    self.assertEqual(bh, self.horizon)
                    self.assertEqual(bh, mh)
                    self.assertEqual(bs, ms)
                    self.assertEqual({k: v for k, v in bm.items() if k != 'variant'},
                                     {k: v for k, v in mm.items() if k != 'variant'})
                    for key in ('knot_time_s', 'knot_qpos', 'coefficients'):
                        np.testing.assert_array_equal(getattr(moved[attacker].curve, key), getattr(self.attack, key))
                        np.testing.assert_array_equal(getattr(baseline[attacker].curve, key), getattr(self.attack, key))
                    self.assertEqual(moved[attacker].start_s, .75)
                    self.assertEqual(moved[defender].start_s, .75)
                    for slot in ('a1', 'a2'):
                        np.testing.assert_array_equal(baseline[slot].target(0.), moved[slot].target(0.))
                    for t in np.linspace(0., bh, 13):
                        np.testing.assert_allclose(baseline[defender].target(t), self.guard, atol=1e-12, rtol=0.)
                    self.assertGreaterEqual(mh-max(p.end_s for p in moved.values()), 2.-1e-12)
        self.assertGreater(self.coordinated.call_count, 0)
        self.assertGreater(self.wrist.call_count, 0)

    def test_same_grid_unchanged_polynomials_and_c2_return(self):
        recipe = self.recipe()
        programs, horizon, metadata, _ = self.make(recipe)
        curve = programs['a2'].curve
        np.testing.assert_array_equal(curve.knot_time_s, self.times)
        np.testing.assert_allclose(curve.knot_qpos-self.base_curve.knot_qpos,
                                   recipe['offsets_rad'], atol=1e-15, rtol=0.)
        np.testing.assert_array_equal(curve.coefficients[:, :, [2, 4, 6]],
                                     self.base_curve.coefficients[:, :, [2, 4, 6]])
        for endpoint in (0., curve.duration_s):
            np.testing.assert_array_equal(curve.sample(endpoint), self.guard)
            for derivative in curve.derivatives(endpoint):
                np.testing.assert_array_equal(derivative, np.zeros(7))
        for knot in curve.knot_time_s[1:-1]:
            np.testing.assert_allclose(curve.sample(knot-1e-7), curve.sample(knot+1e-7), atol=1e-6, rtol=0.)
        np.testing.assert_array_equal(programs['a2'].target(horizon), self.guard)
        self.assertEqual(metadata['allowed_compensation_joints'], ['J0', 'J1', 'J3', 'J5'])
        self.assertEqual(metadata['unchanged_joint_polynomials'], ['J2', 'J4', 'J6'])

    def test_whole_interval_offset_can_fail_even_when_all_nodes_are_within_limit(self):
        recipe = self.recipe()
        recipe['offsets_rad'] = [[0., 0., 0., 0., 0., v, 0.] for v in [0., -.25, -.25, .25, .25, 0.]]
        for variant in ('baseline', 'defended'):
            with self.subTest(variant=variant), self.assertRaisesRegex(ValueError, 'Whole-curve'):
                self.make(recipe, variant)

    def test_invalid_recipe_numbers_shapes_axes_and_endpoints_rejected(self):
        bad = [None, {}, []]
        for key, value in (('schema_version', True), ('base_kind', 'other'),
                           ('base_recipe', '../outside.json'), ('base_recipe', '/tmp/other.json'),
                           ('candidate_id', ''), ('status', 'trained'),
                           ('offsets_rad', [[0.]*6]*6), ('offsets_rad', [[0.]*7]*5)):
            recipe = self.recipe()
            recipe[key] = value
            bad.append(recipe)
        for row, col, value in ((0, 0, .01), (5, 3, .01), (1, 1, .250001),
                                 (1, 2, .01), (1, 4, .01), (1, 6, .01),
                                 (1, 1, False), (1, 3, True), (1, 0, float('nan')),
                                 (1, 5, float('inf')), (1, 1, '0.01')):
            recipe = self.recipe()
            recipe['offsets_rad'][row][col] = value
            bad.append(recipe)
        recipe = self.recipe()
        recipe['offsets_rad'] = [[0.]*7]*6
        bad.append(recipe)
        for variant in ('baseline', 'defended'):
            for recipe in bad:
                with self.subTest(variant=variant, recipe=recipe), self.assertRaises(ValueError):
                    trial.make_programs(recipe, variant, base_recipe=self.base_recipe,
                        source=self.source, reference=self.reference)

    def test_true_joint_range_and_speed_limits_are_not_replaced_by_offset_cap(self):
        # Base J1 reaches .01 rad; a small negative overlay crosses the real lower limit.
        guard = self.guard.copy()
        guard[1] = .08
        self.base_curve = C2JointSpline(self.times, guard+self.base_offsets)
        recipe = self.recipe()
        recipe['offsets_rad'][1][1] = -.02
        for variant in ('baseline', 'defended'):
            with self.assertRaisesRegex(ValueError, 'limits|range'):
                self.make(recipe, variant)
        self.base_curve = C2JointSpline([0., 1., 1.01, 3.], [self.guard]*4)
        self.horizon = 5.75
        recipe = self.recipe()
        recipe['offsets_rad'] = [[0.]*7, [0., .2, 0., 0., 0., 0., 0.],
                                [0., -.2, 0., 0., 0., 0., 0.], [0.]*7]
        for variant in ('baseline', 'defended'):
            with self.assertRaisesRegex(ValueError, 'speed'):
                self.make(recipe, variant)

    def test_incomplete_base_return_or_missing_two_second_tail_is_rejected(self):
        self.horizon = 15.999
        with self.assertRaisesRegex(ValueError, 'two-second'):
            self.make()
        self.horizon = 16.
        q = self.base_curve.knot_qpos.copy()
        q[-1, 0] += .01
        self.base_curve = C2JointSpline(self.times, q)
        with self.assertRaisesRegex(ValueError, 'original guard'):
            self.make()

    def test_full_base_expansion_and_hashes_record_only_an_unproven_own_tip_target(self):
        _, _, metadata, _ = self.make(self.recipe('wrist'), attacker='a2')
        full = metadata['recipe']
        self.assertIn('wrist', full['base_expanded'])
        self.assertIn('base', full['base_expanded'])
        self.assertNotIn('base_J0_J3_reference_exactly_unchanged', metadata)
        self.assertTrue(metadata['base_metadata']['base_J0_J3_reference_exactly_unchanged'])
        self.assertEqual(metadata['recipe_sha256'], trial._digest(full))
        self.assertEqual(metadata['expanded_base_recipe_sha256'], trial._digest(full['base_expanded']))
        self.assertEqual(metadata['contact_point_target']['owner'], 'a1')
        self.assertEqual(metadata['contact_point_target']['direction'], 'toward_own_blade_tip')
        for key in ('own_blade_contact_shift_demonstrated', 'policy_reaction_demonstrated',
                    'defence_certified', 'admitted_to_training'):
            self.assertFalse(metadata[key])
        self.assertFalse(metadata['contact_point_target']['achieved'])
        _, _, other, _ = self.make(self.recipe('wrist'), base_recipe={'candidate_id': 'changed_base'})
        self.assertNotEqual(metadata['recipe_sha256'], other['recipe_sha256'])

    def test_screen_and_cli_honor_attacker_slot_without_physics(self):
        planned = self.make(attacker='a2')
        model = object()
        with patch.object(trial, 'make_programs', return_value=planned) as make, \
                patch('sim.limited_model.build_limited_floor', return_value=SimpleNamespace(model=model)), \
                patch('sim.a1_dynamics_screen.screen_curve', return_value={'fixture': True}) as screen:
            result = trial.screen(self.root/'screen', self.recipe(), attacker='a2')
        make.assert_called_once_with(self.recipe(), 'defended', attacker='a2')
        screen.assert_called_once_with(model, planned[0]['a1'].curve, .002)
        self.assertFalse(result['native_dynamics_tested'])
        recipe_path = self.root/'recipe.json'
        recipe_path.write_text(json.dumps(self.recipe()))
        output = self.root/'cli'
        with patch('sys.argv', ['manual_contact_shift_trial', '--output', str(output),
                               '--variant', 'screen', '--recipe', str(recipe_path), '--attacker-slot', 'a2']), \
                patch.object(trial, 'screen', return_value={}) as mocked, patch('builtins.print'):
            trial.main()
        mocked.assert_called_once_with(output, self.recipe(), attacker='a2')


if __name__ == '__main__':
    unittest.main()
