"""Whole-arm adaptive program validation using small independent fixtures."""
from contextlib import ExitStack
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from sim import manual_adaptive_trial as trial
from sim.joint_constraints import hardware_to_model_rad, load_joint_constraints
from sim.motion_spline import C2JointSpline


class ManualAdaptiveTrialTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.constraints = load_joint_constraints()
        self.display = np.deg2rad(np.asarray(self.constraints.bounds_deg).mean(axis=1))
        self.guard = hardware_to_model_rad(np.asarray(self.constraints.bounds_deg).mean(axis=1), self.constraints)
        mid = self.guard.copy()
        mid[1] += .1
        self.attack = C2JointSpline([0., 1., 2.], [self.guard, mid, self.guard])
        self.source = self.root/'source.json'
        self.scoring = {'base_distance': 1.75, 'regions': [], 'fixture_only': True}
        self.bank = {'scoring': self.scoring, 'actions': {trial.ATTACK: {
            'kind': 'attack', 'target_region': 'upper',
            'knot_time_s': self.attack.knot_time_s.tolist(),
            'knot_qpos': self.attack.knot_qpos.tolist()}}}
        self.source.write_text(json.dumps(self.bank))
        self.reference = self.root/'reference.json'
        self.reference.write_text('{"fixture_only":true}')
        self.stack.enter_context(patch.object(trial, 'load_reference', return_value=(
            {'facing_variant': {'base_yaw_delta_rad': 0.}}, self.display, self.guard)))
        self.stack.enter_context(patch('mujoco.mj_step', side_effect=AssertionError('No native physics')))

    def recipe(self):
        mid = self.guard + np.array([.4, -.35, .3, .35, -.45, .35, .4])
        return {'schema_version': 1, 'candidate_id': 'whole_arm_fixture',
                'status': 'offline_candidate_not_for_training', 'start_s': 1.15,
                'time_s': [0., 2.5, 5.],
                'qpos_cad_rad': [self.guard.tolist(), mid.tolist(), self.guard.tolist()]}

    def make(self, recipe=None, variant='defended', attacker='a1'):
        return trial.make_programs(self.recipe() if recipe is None else recipe, variant,
            attacker=attacker, source=self.source, reference=self.reference)

    def test_matched_attack_initial_guard_baseline_and_duration_both_slots(self):
        for attacker in ('a1', 'a2'):
            with self.subTest(attacker=attacker):
                baseline, bh, bm, bs = self.make(variant='baseline', attacker=attacker)
                moving, mh, mm, ms = self.make(attacker=attacker)
                defender = 'a2' if attacker == 'a1' else 'a1'
                self.assertEqual(bh, mh)
                self.assertEqual(bh, 16.)
                self.assertEqual(bs, ms)
                self.assertEqual({k: v for k, v in bm.items() if k != 'variant'},
                                 {k: v for k, v in mm.items() if k != 'variant'})
                for slot in ('a1', 'a2'):
                    np.testing.assert_array_equal(baseline[slot].target(0.), moving[slot].target(0.))
                for key in ('knot_time_s', 'knot_qpos', 'coefficients'):
                    np.testing.assert_array_equal(getattr(moving[attacker].curve, key), getattr(self.attack, key))
                self.assertEqual(moving[attacker].start_s, .75)
                self.assertEqual(moving[defender].start_s, 1.15)
                for t in np.linspace(0., bh, 17):
                    np.testing.assert_allclose(baseline[defender].target(t), self.guard, atol=1e-12, rtol=0.)

    def test_all_seven_joints_and_new_grid_allowed_without_small_offset_policy(self):
        recipe = self.recipe()
        programs, _, metadata, _ = self.make(recipe)
        curve = programs['a2'].curve
        np.testing.assert_array_equal(curve.knot_time_s, recipe['time_s'])
        np.testing.assert_array_equal(curve.knot_qpos, recipe['qpos_cad_rad'])
        self.assertTrue(np.all(np.abs(curve.knot_qpos[1]-self.guard) > .25))
        for endpoint in (0., curve.duration_s):
            np.testing.assert_array_equal(curve.sample(endpoint), self.guard)
            for derivative in curve.derivatives(endpoint):
                np.testing.assert_array_equal(derivative, np.zeros(7))
        self.assertTrue(metadata['all_defender_joints_may_be_reauthored'])
        self.assertFalse(metadata['fixed_previous_time_grid_or_polynomials_required'])
        self.assertFalse(metadata['heuristic_small_offset_limits_applied'])

    def test_explicit_ready_pose_changes_both_resets_and_requires_new_matched_baseline(self):
        recipe = self.recipe()
        ready = self.guard + np.array([.15, .07, 0., 0., 0., -.02, 0.])
        shift = ready-self.guard
        recipe['ready_qpos_cad_rad'] = ready.tolist()
        recipe['qpos_cad_rad'] = (np.asarray(recipe['qpos_cad_rad'])+shift).tolist()
        for attacker in ('a1', 'a2'):
            with self.subTest(attacker=attacker):
                baseline, bh, bm, _ = self.make(recipe, 'baseline', attacker)
                moving, mh, mm, _ = self.make(recipe, 'defended', attacker)
                defender = 'a2' if attacker == 'a1' else 'a1'
                self.assertEqual(bh, mh)
                np.testing.assert_array_equal(baseline[defender].target(0.), ready)
                np.testing.assert_array_equal(moving[defender].target(0.), ready)
                np.testing.assert_array_equal(moving[defender].target(mh), ready)
                np.testing.assert_array_equal(moving[attacker].target(0.), self.attack.sample(0.))
                np.testing.assert_array_equal(mm['initial_guard_cad_rad'], ready)
                np.testing.assert_array_equal(mm['user_reference_guard_cad_rad'], self.guard)
                self.assertEqual(mm['ready_pose_source'], 'explicit')
                self.assertTrue(mm['requires_new_matched_baseline'])
                self.assertEqual(bm['initial_guard_cad_rad'], mm['initial_guard_cad_rad'])
        # The shared reviewer must refuse the old screenshot-reset baseline.
        _, old_horizon, old_meta, scoring = self.make(variant='baseline')
        _, new_horizon, new_meta, _ = self.make(recipe)
        root = self.root/'different_ready_review'
        for variant, metadata, horizon in (('baseline', old_meta, old_horizon), ('defended', new_meta, new_horizon)):
            folder = root/variant
            folder.mkdir(parents=True)
            (folder/'plan.json').write_text(json.dumps(dict(metadata,
                requested_duration_s=horizon, scoring=scoring)))
        with self.assertRaisesRegex(ValueError, 'baseline threat, reset'):
            trial.review(root)

    def test_omitted_ready_remains_compatible_and_explicit_same_ready_needs_no_new_reset(self):
        default, _, metadata, _ = self.make()
        self.assertEqual(metadata['ready_pose_source'], 'original')
        self.assertFalse(metadata['requires_new_matched_baseline'])
        recipe = self.recipe()
        recipe['ready_qpos_cad_rad'] = self.guard.tolist()
        explicit, _, metadata, _ = self.make(recipe)
        self.assertEqual(metadata['ready_pose_source'], 'explicit')
        self.assertFalse(metadata['requires_new_matched_baseline'])
        for slot in ('a1', 'a2'):
            np.testing.assert_array_equal(default[slot].curve.coefficients, explicit[slot].curve.coefficients)

    def test_explicit_ready_rejects_bad_shape_numbers_limits_and_mismatched_path(self):
        for value in ([0.]*6, [True]*7, [float('nan')]*7, ['0.1']*7,
                      (self.guard + np.array([10., 0., 0., 0., 0., 0., 0.])).tolist()):
            recipe = self.recipe()
            recipe['ready_qpos_cad_rad'] = value
            for variant in ('baseline', 'defended'):
                with self.subTest(ready=value, variant=variant), self.assertRaises(ValueError):
                    self.make(recipe, variant)
        recipe = self.recipe()
        ready = self.guard.copy()
        ready[0] += .1
        recipe['ready_qpos_cad_rad'] = ready.tolist()
        with self.assertRaisesRegex(ValueError, 'selected ready pose'):
            self.make(recipe)

    def test_explicit_start_and_longer_path_extend_full_tail_not_the_attack_clock(self):
        recipe = self.recipe()
        recipe['start_s'] = 0.
        programs, horizon, _, _ = self.make(recipe)
        self.assertEqual(programs['a2'].start_s, 0.)
        self.assertEqual(programs['a1'].start_s, .75)
        self.assertEqual(horizon, 16.)
        recipe['start_s'] = 1.25
        recipe['time_s'] = [0., 14.1, 15.]
        programs, horizon, _, _ = self.make(recipe)
        self.assertGreaterEqual(horizon, 18.25)
        self.assertGreaterEqual(horizon-max(p.end_s for p in programs.values()), 2.-1e-12)
        self.assertAlmostEqual(horizon/.001, round(horizon/.001))

    def test_bad_shapes_numbers_times_and_nonmatching_endpoints_are_rejected(self):
        bad = [None, {}, []]
        for key, value in (('schema_version', True), ('candidate_id', ''), ('status', 'trained'),
                           ('start_s', True), ('start_s', -1.), ('start_s', float('nan')),
                           ('start_s', [1.]), ('time_s', [0., True, 5.]),
                           ('time_s', [0., 2.5, 2.5]), ('time_s', [0., 2.5, 2.]),
                           ('time_s', [.1, 2.5, 5.]), ('time_s', [0., float('inf'), 5.]),
                           ('qpos_cad_rad', [[0.]*6]*3)):
            recipe = self.recipe()
            recipe[key] = value
            bad.append(recipe)
        for row, col, value in ((1, 2, False), (1, 4, float('inf')), (1, 6, '0.1'),
                                 (0, 0, self.guard[0]+1e-12), (2, 3, self.guard[3]+1e-12)):
            recipe = self.recipe()
            recipe['qpos_cad_rad'][row][col] = value
            bad.append(recipe)
        recipe = self.recipe()
        recipe['attack'] = 'different_attack'
        bad.append(recipe)
        for variant in ('baseline', 'defended'):
            for recipe in bad:
                with self.subTest(variant=variant, recipe=recipe), self.assertRaises(ValueError):
                    trial.make_programs(recipe, variant, source=self.source, reference=self.reference)

    def test_limit_speed_and_max_duration_checks_also_apply_to_waiting_baseline(self):
        out_of_range = self.recipe()
        out_of_range['time_s'] = [0., 20., 40.]
        out_of_range['qpos_cad_rad'][1][0] += 10.
        too_fast = self.recipe()
        too_fast['time_s'] = [0., .001, .002]
        too_long = self.recipe()
        too_long['time_s'] = [0., 30., 60.]
        for variant in ('baseline', 'defended'):
            for recipe in (out_of_range, too_fast, too_long):
                with self.subTest(variant=variant, recipe=recipe), self.assertRaises(ValueError):
                    self.make(recipe, variant)
        for kwargs in ({'variant': 'wrong'}, {'attacker': 'wrong'}):
            with self.assertRaises(ValueError):
                self.make(**kwargs)

    def test_fixed_center_attack_must_exist_and_satisfy_common_reference_speed(self):
        for mode in ('missing', 'defence', 'too_fast'):
            bank = copy.deepcopy(self.bank)
            if mode == 'missing':
                del bank['actions'][trial.ATTACK]
            elif mode == 'defence':
                bank['actions'][trial.ATTACK]['kind'] = 'defence'
            else:
                bank['actions'][trial.ATTACK]['knot_time_s'] = [0., .001, .002]
            self.source.write_text(json.dumps(bank))
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.make()

    def test_provenance_is_copied_and_does_not_claim_style_contact_or_success(self):
        recipe = self.recipe()
        programs, _, metadata, _ = self.make(recipe)
        old = programs['a2'].curve.knot_qpos.copy()
        recipe['qpos_cad_rad'][1][0] += .1
        np.testing.assert_array_equal(programs['a2'].curve.knot_qpos, old)
        self.assertNotEqual(metadata['recipe'], recipe)
        self.assertEqual(metadata['source_attack_sha256'], hashlib.sha256(self.source.read_bytes()).hexdigest())
        self.assertEqual(metadata['recipe_sha256'], hashlib.sha256(trial._canonical(metadata['recipe'])).hexdigest())
        self.assertEqual(metadata['coordinate_system'], 'original_CAD_radians_no_additional_zero_shift')
        self.assertTrue(metadata['manual_style_is_qualitative_reference_not_a_physical_certificate'])
        for key in ('shared_controller_geometry_limits_and_scoring_changed',
                    'contact_relocation_or_downward_force_inferred_from_reference',
                    'policy_reaction_demonstrated', 'defence_certified', 'admitted_to_training'):
            self.assertFalse(metadata[key])

    def test_screen_and_cli_use_the_actual_defender_and_preserve_previous_output(self):
        planned = self.make(attacker='a2')
        model = object()
        output = self.root/'screen'
        with patch.object(trial, 'make_programs', return_value=planned) as make, \
                patch('sim.limited_model.build_limited_floor', return_value=SimpleNamespace(model=model)), \
                patch('sim.a1_dynamics_screen.screen_curve', return_value={'fixture': True}) as screen:
            result = trial.screen(output, self.recipe(), attacker='a2')
        make.assert_called_once_with(self.recipe(), 'defended', attacker='a2')
        screen.assert_called_once_with(model, planned[0]['a1'].curve, .002)
        self.assertFalse(result['native_dynamics_tested'])
        before = (output/'screen.json').read_bytes()
        with self.assertRaises(FileExistsError):
            trial.screen(output, self.recipe())
        self.assertEqual((output/'screen.json').read_bytes(), before)
        recipe_path = self.root/'recipe.json'
        recipe_path.write_text(json.dumps(self.recipe()))
        with patch('sys.argv', ['manual_adaptive_trial', '--output', str(self.root/'cli'),
                               '--variant', 'screen', '--recipe', str(recipe_path), '--attacker-slot', 'a2']), \
                patch.object(trial, 'screen', return_value={}) as mocked, patch('builtins.print'):
            trial.main()
        mocked.assert_called_once_with(self.root/'cli', self.recipe(), attacker='a2')

    def test_existing_trial_directory_is_not_overwritten_or_simulated(self):
        root = self.root/'saved'
        (root/'baseline').mkdir(parents=True)
        (root/'baseline/sentinel').write_text('keep')
        with patch.object(trial, 'run_constrained_trial') as native, self.assertRaises(FileExistsError):
            trial.run(root, self.recipe(), 'baseline')
        native.assert_not_called()
        self.assertEqual((root/'baseline/sentinel').read_text(), 'keep')


if __name__ == '__main__':
    unittest.main()
