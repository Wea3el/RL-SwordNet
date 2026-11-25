"""Small coordinated reference contracts without native physics or GUI."""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from sim import manual_coordinated_trial as trial
from sim.joint_constraints import load_joint_constraints
from sim.motion_spline import C2JointSpline


class ManualCoordinatedTrialTests(unittest.TestCase):
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
        self.source = self.root / 'source.json'
        self.scoring = {'base_distance': 1.75, 'regions': [], 'fixture_only': True}
        self.source.write_text(json.dumps({'scoring': self.scoring, 'actions': {
            'fixture_cut': {'kind': 'attack', 'target_region': 'upper',
                'knot_time_s': self.attack.knot_time_s.tolist(),
                'knot_qpos': self.attack.knot_qpos.tolist()}}}))
        self.reference = self.root / 'reference.json'
        self.reference.write_text('{"fixture_only":true}')
        self.reference_mock = self.stack.enter_context(patch.object(trial, 'load_reference',
            return_value=({'facing_variant': {'base_yaw_delta_rad': 0.}}, self.display, self.guard)))
        self.stack.enter_context(patch('mujoco.mj_step', side_effect=AssertionError('No native physics')))

    @staticmethod
    def recipe():
        return {'schema_version': 1, 'candidate_id': 'fixture_small_retraction',
                'status': 'offline_candidate_not_for_training', 'attack': 'fixture_cut',
                'start_s': 1.25, 'time_s': [0., 1., 2., 3.],
                'offsets_rad': [[0.]*7, [-.10, .08, 0., -.06, 0., .04, 0.],
                                [-.15, .04, 0., -.02, 0., .02, 0.], [0.]*7]}

    def make(self, recipe=None, variant='defended', attacker='a1'):
        return trial.make_programs(self.recipe() if recipe is None else recipe, variant,
            attacker=attacker, source=self.source, reference=self.reference)

    def test_matched_attack_reset_clock_duration_and_constant_baseline_both_slots(self):
        for attacker in ('a1', 'a2'):
            with self.subTest(attacker=attacker):
                baseline, bh, bm, bs = self.make(variant='baseline', attacker=attacker)
                defended, dh, dm, ds = self.make(attacker=attacker)
                defender = 'a2' if attacker == 'a1' else 'a1'
                self.assertEqual(bh, dh)
                self.assertEqual(bs, ds)
                self.assertEqual({k: v for k, v in bm.items() if k != 'variant'},
                                 {k: v for k, v in dm.items() if k != 'variant'})
                self.assertEqual(defended[attacker].start_s, .75)
                self.assertEqual(defended[defender].start_s, 1.25)
                for slot in ('a1', 'a2'):
                    np.testing.assert_array_equal(baseline[slot].target(0.), defended[slot].target(0.))
                for field in ('knot_time_s', 'knot_qpos', 'coefficients'):
                    np.testing.assert_array_equal(getattr(defended[attacker].curve, field), getattr(self.attack, field))
                    np.testing.assert_array_equal(getattr(baseline[attacker].curve, field), getattr(self.attack, field))
                for t in np.linspace(0., bh, 21):
                    np.testing.assert_allclose(baseline[defender].target(t), self.guard, atol=1e-12, rtol=0.)
                np.testing.assert_array_equal(defended[defender].target(1.249), self.guard)
                self.assertGreaterEqual(dh-max(p.end_s for p in defended.values()), 2.-1e-12)

    def test_only_declared_axes_move_in_cad_coordinates_and_return_at_rest(self):
        recipe = self.recipe()
        programs, horizon, metadata, _ = self.make(recipe)
        curve = programs['a2'].curve
        np.testing.assert_allclose(curve.knot_qpos, self.guard+np.asarray(recipe['offsets_rad']), atol=1e-15)
        np.testing.assert_array_equal(curve.knot_qpos[:, [2, 4, 6]],
                                     np.tile(self.guard[[2, 4, 6]], (4, 1)))
        for endpoint in (0., curve.duration_s):
            np.testing.assert_array_equal(curve.sample(endpoint), self.guard)
            for derivative in curve.derivatives(endpoint):
                np.testing.assert_array_equal(derivative, np.zeros(7))
        for knot in curve.knot_time_s[1:-1]:
            np.testing.assert_allclose(curve.sample(knot-1e-7), curve.sample(knot+1e-7), atol=1e-6, rtol=0.)
            for left, right in zip(curve.derivatives(knot-1e-7), curve.derivatives(knot+1e-7)):
                np.testing.assert_allclose(left, right, atol=1e-5, rtol=0.)
        np.testing.assert_array_equal(programs['a2'].target(horizon), self.guard)
        self.assertEqual(metadata['offset_coordinate_system'], 'original_CAD_radians')
        self.assertEqual(metadata['allowed_auxiliary_joints'], ['J1', 'J3', 'J5'])
        np.testing.assert_allclose(metadata['planned_peak_absolute_offsets_rad'],
                                   np.max(np.abs(recipe['offsets_rad']), axis=0))

    def test_exact_auxiliary_cap_is_allowed_but_over_cap_and_unauthorized_axes_are_not(self):
        recipe = self.recipe()
        recipe['offsets_rad'][1][3] = -.25
        self.make(recipe)
        bad = []
        for joint in (1, 3, 5):
            recipe = self.recipe()
            recipe['offsets_rad'][1][joint] = .250001
            bad.append(recipe)
        for joint in (2, 4, 6):
            recipe = self.recipe()
            recipe['offsets_rad'][1][joint] = .001
            bad.append(recipe)
        recipe = self.recipe()
        recipe['offsets_rad'][1][0] = 1.000001
        bad.append(recipe)
        for variant in ('baseline', 'defended'):
            for recipe in bad:
                with self.subTest(variant=variant, recipe=recipe), self.assertRaises(ValueError):
                    self.make(recipe, variant)

    def test_bad_shapes_numbers_timing_and_endpoints_are_rejected_for_both_variants(self):
        bad = []
        for key, value in (('schema_version', True), ('start_s', True), ('start_s', .749),
                           ('start_s', float('nan')), ('time_s', [0., 1., 1., 3.]),
                           ('time_s', [0., 2., 1., 3.]), ('time_s', [0., .001, .002, .003]),
                           ('time_s', [0., 20., 40., 60.]), ('time_s', [[0., 1., 2., 3.]]),
                           ('time_s', [0., 1., float('inf'), 3.]),
                           ('offsets_rad', [[0.]*6]*4)):
            recipe = self.recipe()
            recipe[key] = value
            bad.append(recipe)
        for row, col, value in ((0, 0, .01), (3, 3, .01), (1, 1, float('nan'))):
            recipe = self.recipe()
            recipe['offsets_rad'][row][col] = value
            bad.append(recipe)
        recipe = self.recipe()
        recipe['unexpected'] = 1
        bad.append(recipe)
        for variant in ('baseline', 'defended'):
            for recipe in bad:
                with self.subTest(variant=variant, recipe=recipe), self.assertRaises(ValueError):
                    self.make(recipe, variant)

    def test_mixed_boolean_arrays_are_not_silently_coerced_to_numbers(self):
        bad = []
        recipe = self.recipe()
        recipe['time_s'][1] = True  # In a float list, numpy normally silently coerces this to 1.
        bad.append(recipe)
        recipe = self.recipe()
        recipe['offsets_rad'][1][2] = False  # Physically zero, but not valid numeric input.
        bad.append(recipe)
        for variant in ('baseline', 'defended'):
            for recipe in bad:
                with self.subTest(variant=variant, recipe=recipe), self.assertRaises(ValueError):
                    self.make(recipe, variant)

    def test_actual_joint_ranges_checked_even_when_auxiliary_offsets_are_small(self):
        display = self.display.copy()
        display[1] = .01
        self.reference_mock.return_value = ({'facing_variant': {'base_yaw_delta_rad': 0.}}, display, None)
        recipe = self.recipe()
        recipe['offsets_rad'][1][1] = -.02
        for variant in ('baseline', 'defended'):
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                self.make(recipe, variant)

    def test_requires_existing_attack_j0_auxiliary_motion_and_nontraining_metadata(self):
        source = json.loads(self.source.read_text())
        source['actions']['fixture_guard'] = dict(source['actions']['fixture_cut'], kind='defence')
        self.source.write_text(json.dumps(source))
        for name in ('missing', False, 'fixture_guard'):
            recipe = self.recipe()
            recipe['attack'] = name
            with self.assertRaises(ValueError):
                self.make(recipe)
        for cols in ((0,), (1, 3, 5)):
            recipe = self.recipe()
            for row in recipe['offsets_rad']:
                for col in cols:
                    row[col] = 0.
            with self.assertRaises(ValueError):
                self.make(recipe)
        _, _, metadata, _ = self.make()
        self.assertTrue(metadata['manual_reference_is_rounded_screenshot'])
        self.assertTrue(metadata['uses_known_clock_for_offline_probe_only'])
        self.assertFalse(metadata['policy_reaction_demonstrated'])
        self.assertFalse(metadata['admitted_to_training'])
        self.assertEqual(metadata['source_attack_sha256'], hashlib.sha256(self.source.read_bytes()).hexdigest())

    def _review_fixture(self, label):
        root = self.root / label
        summaries = []
        for variant in ('baseline', 'defended'):
            programs, horizon, metadata, scoring = self.make(variant=variant)
            folder = root / variant
            native = folder / 'native'
            native.mkdir(parents=True)
            (folder/'plan.json').write_text(json.dumps(dict(metadata,
                requested_duration_s=horizon, scoring=scoring)))
            summary = {'entry_ready': {'a1': True, 'a2': True},
                       'final_settled': {'a1': True, 'a2': True},
                       'qpos_resets_after_initialization': 0}
            (native/'summary.json').write_text(json.dumps(summary))
            summaries.append(summary)
            if variant == 'baseline':
                np.savez(native/'references.npz', a1_knot_time_s=self.attack.knot_time_s,
                         a2_knot_time_s=programs['a2'].curve.knot_time_s,
                         a2_knot_qpos=programs['a2'].curve.knot_qpos)
        return root, summaries

    @staticmethod
    def _qualified(hit=1.5):
        return {'status': 'qualified', 'successful_physical_defence': True,
                'matching_checks': {'mocked_native_compare': True},
                'baseline_first_triangle_hit_s': hit}

    def test_review_does_not_promote_unsettled_reset_or_late_hit_evidence(self):
        mutations = (('entry', 'entry_ready', {'a1': False, 'a2': True}),
                     ('final', 'final_settled', {'a1': True, 'a2': False}),
                     ('reset', 'qpos_resets_after_initialization', 1),
                     ('late', None, None))
        for label, key, value in mutations:
            root, summaries = self._review_fixture(label)
            if key:
                summaries[1][key] = value
                (root/'defended/native/summary.json').write_text(json.dumps(summaries[1]))
            with patch.object(trial, 'compare', return_value=self._qualified(3. if label == 'late' else 1.5)):
                result = trial.review(root)
            self.assertEqual(result['status'], 'invalid')
            self.assertFalse(result['successful_physical_defence'])
            self.assertFalse(result['admitted_to_training'])

    def test_review_checks_constant_baseline_and_refuses_overwriting_comparison(self):
        root, _ = self._review_fixture('nonconstant')
        q = np.tile(self.guard, (2, 1))
        q[1, 0] += .01
        np.savez(root/'baseline/native/references.npz', a1_knot_time_s=self.attack.knot_time_s,
                 a2_knot_time_s=[0., 3.], a2_knot_qpos=q)
        with patch.object(trial, 'compare', return_value=self._qualified()):
            result = trial.review(root)
        self.assertEqual(result['status'], 'invalid')
        self.assertFalse(result['matching_checks']['baseline_defender_entire_reference_is_original_guard'])
        with self.assertRaises(FileExistsError):
            trial.review(root)


if __name__ == '__main__':
    unittest.main()
