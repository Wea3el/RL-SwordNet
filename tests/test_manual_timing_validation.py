"""Timing contracts use small fixtures, never native physics or saved writes."""
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

from sim import manual_adaptive_trial as adaptive
from sim import manual_timing_validation as trial
from sim.joint_constraints import hardware_to_model_rad, load_joint_constraints
from sim.motion_spline import C2JointSpline


class ManualTimingValidationTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        constraints = load_joint_constraints()
        display = np.deg2rad(np.asarray(constraints.bounds_deg).mean(axis=1))
        self.guard = hardware_to_model_rad(np.asarray(constraints.bounds_deg).mean(axis=1), constraints)
        self.ready = self.guard.copy()
        self.ready[0] += .1
        middle = self.guard.copy()
        middle[1] += .1
        self.attack = C2JointSpline([0., 1., 2.], [self.guard, middle, self.guard])
        self.source = self.root/'source.json'
        self.scoring = {'base_distance': 1.75, 'regions': [], 'fixture_only': True}
        self.source.write_text(json.dumps({'scoring': self.scoring, 'actions': {adaptive.ATTACK: {
            'kind': 'attack', 'target_region': 'upper',
            'knot_time_s': self.attack.knot_time_s.tolist(),
            'knot_qpos': self.attack.knot_qpos.tolist()}}}))
        self.reference = self.root/'reference.json'
        self.reference.write_text('{"fixture_only":true}')
        self.stack.enter_context(patch.object(adaptive, 'load_reference', return_value=(
            {'facing_variant': {'base_yaw_delta_rad': 0.}}, display, self.guard)))
        self.stack.enter_context(patch('mujoco.mj_step', side_effect=AssertionError('No native physics')))

    def recipe(self):
        middle = self.ready.copy()
        middle[0] += .25
        middle[3] -= .2
        return {'schema_version': 1, 'candidate_id': 'timing_fixture',
                'status': 'offline_candidate_not_for_training', 'start_s': .75,
                'ready_qpos_cad_rad': self.ready.tolist(),
                'time_s': [0., 1.8, 2.8, 4.25, 5.5, 13.25],
                'qpos_cad_rad': [self.ready.tolist(), self.ready.tolist(), middle.tolist(),
                                 middle.tolist(), self.ready.tolist(), self.ready.tolist()]}

    def make(self, delta=.1, attacker='a1', recipe=None):
        with patch.object(trial, 'bind_baseline', return_value={'fixture_only': True}):
            return trial.make_programs(delta, attacker=attacker,
                recipe=self.recipe() if recipe is None else recipe,
                source=self.source, reference=self.reference)

    def baseline(self, attacker='a1'):
        programs, horizon, metadata, scoring = self.make(attacker=attacker)
        defender = metadata['defender_slot']
        root = self.root/('baseline_'+attacker)
        folder = root/'baseline/native'
        folder.mkdir(parents=True)
        plan = dict(metadata, requested_duration_s=horizon, scoring=scoring)
        (folder.parent/'plan.json').write_text(json.dumps(plan))
        dependencies = {'AGENTS.md': trial._sha(trial.ROOT/'AGENTS.md')}
        summary = {'requested_duration_s': horizon, 'native_execution_pass': True,
            'source_unchanged': True, 'failure': None, 'qpos_resets_after_initialization': 0,
            'entry_ready': {'a1': True, 'a2': True}, 'final_settled': {'a1': True, 'a2': True},
            'program_start_s': {attacker: .75, defender: .75},
            'source_sha256': dependencies, 'model_source_sha256': dependencies,
            'runtime_config': {'physics_timestep_s': .1}}
        (folder/'summary.json').write_text(json.dumps(summary))
        (folder/'scene.xml').write_text('<fixture/>')
        (folder/'contacts.json').write_text('[]')
        hold = C2JointSpline([0., 13.25], [self.ready, self.ready])
        curves = {attacker: programs[attacker].curve, defender: hold}
        np.savez(folder/'references.npz', **{slot+'_'+key: getattr(curve, key)
            for slot, curve in curves.items() for key in ('knot_time_s', 'knot_qpos', 'coefficients')})
        initial = np.r_[programs['a1'].target(0.), programs['a2'].target(0.)]
        times = np.arange(161)*.1
        np.savez(folder/'trace.npz', time_s=times, qpos=np.tile(initial, (len(times), 1)),
                 qvel=np.zeros((len(times), 14)))
        previous = {'baseline_valid_threat': True, 'attacker_slot': attacker,
            'baseline_first_triangle_hit_s': 1.5, 'baseline_triangle_audit': {
                'native_trace_source_sha256': {name: trial._sha(folder/name)
                    for name in ('summary.json', 'trace.npz', 'scene.xml')}}}
        (root/'comparison.json').write_text(json.dumps(previous))
        return root, programs, horizon, metadata, scoring

    def test_all_requested_offsets_keep_attack_reset_motion_prefix_and_complete_tail(self):
        source = self.recipe()
        before = copy.deepcopy(source)
        original = C2JointSpline(source['time_s'], source['qpos_cad_rad'])
        for attacker in ('a1', 'a2'):
            for delta in trial.REQUESTED_DELTAS_S:
                with self.subTest(attacker=attacker, delta=delta):
                    programs, horizon, metadata, _ = self.make(delta, attacker, source)
                    defender = metadata['defender_slot']
                    attack, defence = programs[attacker], programs[defender]
                    self.assertEqual(attack.start_s, .75)
                    self.assertEqual(defence.start_s, .75+delta)
                    self.assertEqual(defence.end_s, 14.)
                    self.assertEqual(horizon, 16.)
                    self.assertGreaterEqual(horizon-max(p.end_s for p in programs.values()), 2.)
                    for key in ('knot_time_s', 'knot_qpos', 'coefficients'):
                        np.testing.assert_array_equal(getattr(attack.curve, key), getattr(self.attack, key))
                    np.testing.assert_array_equal(defence.curve.coefficients[:-1], original.coefficients[:-1])
                    np.testing.assert_array_equal(defence.curve.bezier_controls[:-1], original.bezier_controls[:-1])
                    np.testing.assert_array_equal(defence.curve.knot_time_s[:-1], original.knot_time_s[:-1])
                    np.testing.assert_array_equal(defence.curve.knot_qpos, original.knot_qpos)
                    np.testing.assert_array_equal(defence.target(0.), self.ready)
                    np.testing.assert_array_equal(defence.target(horizon), self.ready)
                    for endpoint in (0., defence.curve.duration_s):
                        for derivative in defence.curve.derivatives(endpoint):
                            np.testing.assert_array_equal(derivative, np.zeros(7))
                    self.assertTrue(metadata['uses_known_clock_for_offline_probe_only'])
                    self.assertFalse(metadata['policy_reaction_demonstrated'])
                    self.assertFalse(metadata['admitted_to_training'])
                    self.assertEqual(metadata['timing_source_recipe'], source)
        self.assertEqual(source, before)

    def test_only_flat_final_hold_can_be_resized_and_original_full_tail_is_required(self):
        for mutate in ('nonflat', 'missing_tail', 'original_clock', 'nonzero_local_start'):
            recipe = self.recipe()
            if mutate == 'nonflat':
                recipe['qpos_cad_rad'][-2][0] += .01
            elif mutate == 'missing_tail':
                recipe['time_s'][-1] = 13.
            elif mutate == 'original_clock':
                recipe['start_s'] = 1.
            else:
                recipe['time_s'][0] = .01
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                self.make(recipe=recipe)

    def test_bad_offsets_cannot_produce_negative_start_or_reverse_final_hold(self):
        for delta in (True, False, '0.1', None, [0.1], float('nan'), float('inf'), -.75001, 8.):
            with self.subTest(delta=delta), self.assertRaises(ValueError):
                self.make(delta)
        with self.assertRaises(ValueError):
            self.make(attacker='wrong')
        with self.assertRaises(ValueError):
            trial.shift_recipe([], .1)

    def test_shared_joint_limits_speed_and_sixteen_second_horizon_still_apply(self):
        for mode in ('limit', 'speed', 'bool'):
            recipe = self.recipe()
            if mode == 'limit':
                recipe['qpos_cad_rad'][2][3] = 20.
            elif mode == 'speed':
                recipe['time_s'][2] = 1.800001
            else:
                recipe['qpos_cad_rad'][2][0] = True
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.make(recipe=recipe)
        planned = self.make()
        with patch.object(trial, 'adaptive_programs', return_value=(planned[0], 17., planned[2], planned[3])), \
                self.assertRaisesRegex(ValueError, '16-second'):
            self.make()

    def test_matching_immutable_waiting_baseline_binds_in_both_slots(self):
        for attacker in ('a1', 'a2'):
            args = self.baseline(attacker)
            before = {p: p.read_bytes() for p in args[0].rglob('*') if p.is_file()}
            bound = trial.bind_baseline(*args)
            self.assertTrue(bound['qualified_waiting_threat_bound'])
            self.assertTrue(bound['baseline_read_only'])
            self.assertEqual(set(bound['sha256']), set(trial.BOUND_FILES))
            self.assertEqual(trial._check_binding(bound), args[0])
            for path, content in before.items():
                self.assertEqual(path.read_bytes(), content)

    def test_baseline_with_changed_ready_clock_source_or_window_is_rejected(self):
        for mode in ('ready', 'clock', 'source', 'window', 'not_ready', 'reset', 'stale', 'no_threat'):
            args = self.baseline('a2')
            root = args[0]
            plan_path, summary_path = root/'baseline/plan.json', root/'baseline/native/summary.json'
            plan, summary = json.loads(plan_path.read_text()), json.loads(summary_path.read_text())
            if mode == 'ready':
                plan['initial_guard_cad_rad'][0] += .01
            elif mode == 'clock':
                summary['program_start_s']['a2'] = .8
            elif mode == 'source':
                plan['source_attack_sha256'] = 'different'
            elif mode == 'window':
                summary['requested_duration_s'] = 15.
            elif mode == 'not_ready':
                summary['final_settled']['a1'] = False
            elif mode == 'reset':
                summary['qpos_resets_after_initialization'] = 1
            elif mode == 'stale':
                summary['source_sha256']['AGENTS.md'] = 'stale'
            else:
                previous_path = root/'comparison.json'
                previous = json.loads(previous_path.read_text())
                previous['baseline_valid_threat'] = False
                previous_path.write_text(json.dumps(previous))
            plan_path.write_text(json.dumps(plan))
            summary_path.write_text(json.dumps(summary))
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                trial.bind_baseline(*args)
            # Each mutation gets a fresh immutable fixture, without overwriting prior evidence.
            root.rename(self.root/('invalid_'+mode))

    def test_baseline_attack_reference_and_initial_trace_must_match(self):
        for mode in ('attack', 'moving_waiter', 'initial', 'truncated', 'unbound_audit'):
            args = self.baseline()
            root, folder = args[0], args[0]/'baseline/native'
            if mode in ('attack', 'moving_waiter'):
                with np.load(folder/'references.npz') as refs:
                    data = {k: refs[k].copy() for k in refs.files}
                data[('a1' if mode == 'attack' else 'a2')+'_knot_qpos'][0, 0] += .01
                np.savez(folder/'references.npz', **data)
            elif mode in ('initial', 'truncated'):
                with np.load(folder/'trace.npz') as trace:
                    data = {k: trace[k].copy() for k in trace.files}
                if mode == 'initial':
                    data['qpos'][0, 0] += .01
                else:
                    data = {k: v[:-1] for k, v in data.items()}
                np.savez(folder/'trace.npz', **data)
            else:
                (folder/'scene.xml').write_text('<changed_fixture/>')
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                trial.bind_baseline(*args)
            root.rename(self.root/('invalid_'+mode))

    def test_review_rechecks_all_frozen_files_before_delegating(self):
        args = self.baseline()
        binding = trial.bind_baseline(*args)
        output = self.root/'review'
        (output/'defended').mkdir(parents=True)
        (output/'defended/plan.json').write_text(json.dumps({'baseline_binding': binding}))
        with patch.object(trial, 'coordinated_review', return_value={'status': 'fixture'}) as review:
            self.assertEqual(trial.review(output)['status'], 'fixture')
        review.assert_called_once_with(output, args[0])
        (args[0]/'baseline/native/contacts.json').write_text('[{"changed":true}]')
        with patch.object(trial, 'coordinated_review') as review, self.assertRaises(ValueError):
            trial.review(output)
        review.assert_not_called()

    def test_screen_and_cli_select_actual_defender_and_preserve_old_outputs(self):
        planned = self.make(attacker='a2')
        output = self.root/'screen'
        model = object()
        with patch.object(trial, 'make_programs', return_value=planned) as make, \
                patch('sim.limited_model.build_limited_floor', return_value=SimpleNamespace(model=model)), \
                patch('sim.a1_dynamics_screen.screen_curve', return_value={}) as screen:
            result = trial.screen(output, .25, attacker='a2')
        make.assert_called_once_with(.25, attacker='a2')
        screen.assert_called_once_with(model, planned[0]['a1'].curve, .002)
        self.assertFalse(result['native_dynamics_tested'])
        before = (output/'screen.json').read_bytes()
        with self.assertRaises(FileExistsError):
            trial.screen(output, .25)
        self.assertEqual(before, (output/'screen.json').read_bytes())
        with patch('sys.argv', ['manual_timing_validation', '--output', str(self.root/'cli'),
                    '--variant', 'defended', '--delta=-0.1', '--attacker-slot', 'a2']), \
                patch.object(trial, 'run', return_value={}) as run, patch('builtins.print'):
            trial.main()
        run.assert_called_once_with(self.root/'cli', -.1, attacker='a2')

    def test_existing_native_candidate_is_not_overwritten_or_executed(self):
        output = self.root/'existing'
        (output/'defended').mkdir(parents=True)
        sentinel = output/'defended/keep'
        sentinel.write_text('keep')
        with patch.object(trial, 'run_constrained_trial') as native, self.assertRaises(FileExistsError):
            trial.run(output, .1)
        native.assert_not_called()
        self.assertEqual(sentinel.read_text(), 'keep')


if __name__ == '__main__':
    unittest.main()
