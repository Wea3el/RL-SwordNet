"""Exact-prefix and bounded wrist-overlay contracts; no native physics."""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from sim import manual_coordinated_trial as coordinated
from sim import manual_wrist_trial as trial
from sim.joint_constraints import load_joint_constraints
from sim.motion_spline import C2JointSpline


class ManualWristTrialTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.constraints = load_joint_constraints()
        self.display = np.deg2rad(np.asarray(self.constraints.bounds_deg).mean(axis=1))
        self.guard = coordinated.choose_pose(self.display, self.constraints)
        mid = self.guard.copy()
        mid[1] += .1
        self.attack = C2JointSpline([0., 1., 2.], [self.guard, mid, self.guard])
        self.source = self.root / 'source.json'
        self.scoring = {'base_distance': 1.75, 'regions': [], 'fixture_only': True}
        self.source.write_text(json.dumps({'scoring': self.scoring, 'actions': {
            'fixture_cut': {'kind': 'attack', 'target_region': 'upper',
                'knot_time_s': self.attack.knot_time_s.tolist(),
                'knot_qpos': self.attack.knot_qpos.tolist()}}}))
        self.reference = self.root / 'reference.json'
        self.reference.write_text('{"fixture_only":true}')
        self.reference_mock = self.stack.enter_context(patch.object(coordinated, 'load_reference',
            return_value=({'facing_variant': {'base_yaw_delta_rad': 0.}}, self.display, self.guard)))
        self.stack.enter_context(patch('mujoco.mj_step', side_effect=AssertionError('No native physics')))

    @staticmethod
    def base_recipe():
        return {'schema_version': 1, 'candidate_id': 'base_tracking_fixture',
                'status': 'offline_candidate_not_for_training', 'attack': 'fixture_cut',
                'start_s': .75, 'time_s': [0., 2.5, 3., 3.5, 5.5, 13.25],
                'offsets_rad': [[0.]*7, [-.625, -.07, 0., .25, 0., .25, 0.],
                                [-.67, -.07, 0., .25, 0., .25, 0.],
                                [-.725, -.07, 0., .25, 0., .25, 0.],
                                [0.]*7, [0.]*7]}

    @staticmethod
    def recipe():
        return {'schema_version': 1, 'candidate_id': 'small_wrist_fixture',
                'status': 'offline_candidate_not_for_training',
                'base_recipe': 'fixture_base.json',
                'wrist_offsets_rad': [[0., 0., 0.], [.10, -.10, .10],
                                      [.15, -.12, .15], [.15, -.12, .10],
                                      [0., 0., 0.], [0., 0., 0.]]}

    def make(self, recipe=None, variant='defended', attacker='a1', base=None):
        return trial.make_programs(self.recipe() if recipe is None else recipe, variant,
            attacker=attacker, base_recipe=self.base_recipe() if base is None else base,
            source=self.source, reference=self.reference)

    def base(self, attacker='a1', recipe=None):
        return coordinated.make_programs(self.base_recipe() if recipe is None else recipe,
            'defended', attacker=attacker, source=self.source, reference=self.reference)

    def test_prefix_polynomials_attack_and_time_grid_are_exact_in_both_slots(self):
        for attacker in ('a1', 'a2'):
            with self.subTest(attacker=attacker):
                base, bh, _, bs = self.base(attacker)
                new, nh, metadata, ns = self.make(attacker=attacker)
                defender = 'a2' if attacker == 'a1' else 'a1'
                self.assertEqual(bh, nh)
                self.assertEqual(bs, ns)
                for key in ('knot_time_s', 'knot_qpos', 'coefficients'):
                    np.testing.assert_array_equal(getattr(base[attacker].curve, key), getattr(new[attacker].curve, key))
                np.testing.assert_array_equal(base[defender].curve.knot_time_s, new[defender].curve.knot_time_s)
                np.testing.assert_array_equal(base[defender].curve.knot_qpos[:, :4], new[defender].curve.knot_qpos[:, :4])
                np.testing.assert_array_equal(base[defender].curve.coefficients[:, :, :4], new[defender].curve.coefficients[:, :, :4])
                np.testing.assert_array_equal(base[defender].curve.bezier_controls[:, :, :4], new[defender].curve.bezier_controls[:, :, :4])
                self.assertEqual(base[defender].start_s, new[defender].start_s)
                self.assertTrue(metadata['base_J0_J3_reference_exactly_unchanged'])

    def test_baseline_holds_original_guard_with_same_reset_and_full_horizon(self):
        for attacker in ('a1', 'a2'):
            base, bh, bm, bs = self.make(variant='baseline', attacker=attacker)
            new, nh, nm, ns = self.make(attacker=attacker)
            defender = 'a2' if attacker == 'a1' else 'a1'
            self.assertEqual(bh, nh)
            self.assertEqual(bs, ns)
            self.assertEqual({k: v for k, v in bm.items() if k != 'variant'},
                             {k: v for k, v in nm.items() if k != 'variant'})
            for slot in ('a1', 'a2'):
                np.testing.assert_array_equal(base[slot].target(0.), new[slot].target(0.))
            for t in np.linspace(0., nh, 23):
                np.testing.assert_allclose(base[defender].target(t), self.guard, atol=1e-12, rtol=0.)
            self.assertGreaterEqual(nh-max(p.end_s for p in new.values()), 2.-1e-12)
            self.assertAlmostEqual(nh/.001, round(nh/.001))

    def test_wrist_overlay_returns_to_guard_with_continuous_c2_and_whole_interval_bound(self):
        base, _, _, _ = self.base()
        new, horizon, metadata, _ = self.make()
        curve = new['a2'].curve
        np.testing.assert_allclose(curve.knot_qpos[:, 4:]-base['a2'].curve.knot_qpos[:, 4:],
                                   self.recipe()['wrist_offsets_rad'], atol=1e-15, rtol=0.)
        self.assertLessEqual(np.max(np.abs(curve.bezier_controls[:, :, 4:]-base['a2'].curve.bezier_controls[:, :, 4:])), .35)
        for endpoint in (0., curve.duration_s):
            np.testing.assert_array_equal(curve.sample(endpoint), self.guard)
            for derivative in curve.derivatives(endpoint):
                np.testing.assert_array_equal(derivative, np.zeros(7))
        for knot in curve.knot_time_s[1:-1]:
            np.testing.assert_allclose(curve.sample(knot-1e-7), curve.sample(knot+1e-7), atol=1e-6, rtol=0.)
            for left, right in zip(curve.derivatives(knot-1e-7), curve.derivatives(knot+1e-7)):
                np.testing.assert_allclose(left, right, atol=1e-5, rtol=0.)
        np.testing.assert_array_equal(new['a2'].target(horizon), self.guard)
        self.assertTrue(metadata['downward_force_or_success_not_inferred_from_command'])
        self.assertFalse(metadata['policy_reaction_demonstrated'])
        self.assertFalse(metadata['admitted_to_training'])

    def test_node_bound_alone_cannot_pass_oversized_interior_wrist_difference(self):
        recipe = self.recipe()
        recipe['wrist_offsets_rad'] = [[0., value, 0.] for value in [0., -.35, -.35, .35, .35, 0.]]
        # This exact tracking-v2 pattern has legal nodes but > .35 between them.
        for variant in ('baseline', 'defended'):
            with self.subTest(variant=variant), self.assertRaisesRegex(ValueError, 'Whole-curve'):
                self.make(recipe, variant)

    def test_bad_recipe_types_shapes_numbers_and_endpoint_pulses_are_rejected(self):
        bad = [[], {}, None]
        for key, value in (('schema_version', True), ('candidate_id', ''), ('base_recipe', []),
                           ('status', 'trained'), ('wrist_offsets_rad', [[0., 0.]]*6),
                           ('wrist_offsets_rad', [[0., 0., 0.]]*5),
                           ('wrist_offsets_rad', 'not numeric')):
            recipe = self.recipe()
            recipe[key] = value
            bad.append(recipe)
        for row, col, value in ((0, 0, .01), (5, 2, .01), (1, 0, .350001),
                                 (1, 1, True), (1, 1, False), (1, 2, float('nan')),
                                 (1, 0, float('inf')), (1, 0, '0.1')):
            recipe = self.recipe()
            recipe['wrist_offsets_rad'][row][col] = value
            bad.append(recipe)
        recipe = self.recipe()
        recipe['wrist_offsets_rad'] = [[0., 0., 0.]]*6
        bad.append(recipe)
        recipe = self.recipe()
        recipe['unexpected'] = True
        bad.append(recipe)
        for variant in ('baseline', 'defended'):
            for recipe in bad:
                with self.subTest(variant=variant, recipe=recipe), self.assertRaises(ValueError):
                    trial.make_programs(recipe, variant, base_recipe=self.base_recipe(),
                        source=self.source, reference=self.reference)

    def test_actual_joint_limit_not_overridden_by_allowed_wrist_increment(self):
        display = self.display.copy()
        display[6] = np.deg2rad(88.)
        self.reference_mock.return_value = ({'facing_variant': {'base_yaw_delta_rad': 0.}}, display, None)
        recipe = self.recipe()
        for row in recipe['wrist_offsets_rad'][1:4]:
            row[2] = .1  # Under increment cap, but pushes J6 beyond +90 degrees.
        for variant in ('baseline', 'defended'):
            with self.subTest(variant=variant), self.assertRaisesRegex(ValueError, 'limits|range'):
                self.make(recipe, variant)

    def test_speed_ceiling_and_variant_slot_checks_are_not_bypassed(self):
        base = self.base_recipe()
        base['time_s'] = [0., 1., 1.01, 3.]
        base['offsets_rad'] = [[0.]*7, [-.02, 0., 0., .001, 0., 0., 0.],
                               [-.02, 0., 0., .001, 0., 0., 0.], [0.]*7]
        recipe = self.recipe()
        recipe['wrist_offsets_rad'] = [[0., 0., 0.], [.2, 0., 0.], [-.2, 0., 0.], [0., 0., 0.]]
        for variant in ('baseline', 'defended'):
            with self.subTest(variant=variant), self.assertRaisesRegex(ValueError, 'speed'):
                self.make(recipe, variant, base=base)
        with self.assertRaises(ValueError):
            self.make(variant='other')
        with self.assertRaises(ValueError):
            self.make(attacker='other')

    def test_screen_passes_attacker_and_uses_the_opposite_defender_curve(self):
        for module, recipe, planned in ((trial, self.recipe(), self.make(attacker='a2')),
                                       (coordinated, self.base_recipe(), self.base(attacker='a2'))):
            with self.subTest(module=module.__name__):
                programs, _, metadata, _ = planned
                output = self.root / module.__name__.rsplit('.', 1)[-1]
                model = object()
                with patch.object(module, 'make_programs', return_value=planned) as make, \
                        patch('sim.limited_model.build_limited_floor', return_value=SimpleNamespace(model=model)), \
                        patch('sim.a1_dynamics_screen.screen_curve', return_value={'fixture_screen': True}) as screen:
                    result = module.screen(output, recipe, attacker='a2')
                make.assert_called_once_with(recipe, 'defended', attacker='a2')
                screen.assert_called_once_with(model, programs['a1'].curve, .002)
                self.assertEqual(metadata['defender_slot'], 'a1')
                self.assertFalse(result['native_dynamics_tested'])
                self.assertTrue((output/'screen.json').is_file())

    def test_screen_cli_forwards_selected_attacker_slot(self):
        for module, recipe in ((trial, self.recipe()), (coordinated, self.base_recipe())):
            with self.subTest(module=module.__name__):
                recipe_path = self.root / (module.__name__.rsplit('.', 1)[-1]+'.json')
                recipe_path.write_text(json.dumps(recipe))
                output = self.root / 'cli_output'
                argv = [module.__name__, '--output', str(output), '--variant', 'screen',
                        '--recipe', str(recipe_path), '--attacker-slot', 'a2']
                with patch('sys.argv', argv), patch('builtins.print'), \
                        patch.object(module, 'screen', return_value={}) as screen:
                    module.main()
                screen.assert_called_once_with(output, recipe, attacker='a2')


if __name__ == '__main__':
    unittest.main()
