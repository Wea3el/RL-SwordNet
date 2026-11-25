"""Reference preparation and mocked orchestration; never integrate physics."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from sim import a1_release_case as case_runner
from sim.a1_release import DEFAULT_CONSTRAINTS_PATH, DEFAULT_PACKAGE, MANIFEST_STATUS
from sim.a1_runtime import load_runtime_profile
from sim.motion_spline import C2JointSpline


def write_json(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False), encoding='utf-8')


def curve_row(times, qpos):
    curve = C2JointSpline(times, qpos)
    return dict(knot_time_s=curve.knot_time_s.tolist(),
                knot_qpos=curve.knot_qpos.tolist(), coefficients=curve.coefficients.tolist())


class ReleaseCaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'package'
        self.root.mkdir()
        # An accidental native call is a test failure, not a hidden simulation.
        self.step_patch = patch('mujoco.mj_step', side_effect=AssertionError('native forbidden'))
        self.step_patch.start()
        self.addCleanup(self.step_patch.stop)
        q = [1., .5, 1., .5, 1., .5, 0.]
        qa, qd = list(q), list(q)
        qa[0] += .1
        qd[2] += .1
        self.case = dict(duration_s=4., attacker_slot='a1', defender_slot='a2',
            initial_qpos={'a1': q, 'a2': list(q)},
            programs={
                'a1': dict(start_s=.2, curve=curve_row([0., 1., 2.], [q, qa, q])),
                'a2': dict(start_s=.4, curve=curve_row([0., .8, 1.8], [q, qd, q]))},
            target_region='left', admitted_to_training=False,
            proof_sha256='a' * 64, native_references_sha256='b' * 64)
        self.runtime = load_runtime_profile(
            case_runner.ROOT / 'config/a1_runtime_rk4_candidate.json').runtime_config()
        self.scoring = dict(base_distance=1.75, regions={
            'left': dict(center=[.32, .2, .93], radius=.1),
            'right': dict(center=[.32, -.2, .93], radius=.1),
            'upper': dict(center=[.32, 0., 1.22], radius=.1)})
        self.cases = dict(schema_version=1, cases={'example': self.case},
            runtime_config=self.runtime, scoring=self.scoring,
            native_steps=0, admitted_to_training=False)
        catalog = dict(schema_version=1, admitted_to_training=False,
            initial_guard='ready', guards={'ready': {'qpos_rad': q}},
            policy_order=['hold'], actions={'hold': dict(kind='defence',
                start_guard='ready', end_guard='ready', **curve_row([0., 1.], [q, q]))})
        write_json(self.root / 'catalog.json', catalog)
        (self.root / 'joint_constraints_observed.json').write_bytes(DEFAULT_CONSTRAINTS_PATH.read_bytes())
        self.manifest = dict(schema_version=1, status=MANIFEST_STATUS,
            catalog_file='catalog.json', cases_file='cases.json',
            joint_constraints_file='joint_constraints_observed.json',
            legacy_actor_compatible=False, admitted_to_training=False, files_sha256={})
        self.update_package()

    def update_package(self):
        write_json(self.root / 'cases.json', self.cases)
        self.manifest['files_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in self.root.iterdir() if p.name != 'manifest.json'}
        write_json(self.root / 'manifest.json', self.manifest)

    def prepared(self, variant='defended'):
        return case_runner.prepare_case('example', variant, self.root)

    def test_defended_exact_knots_coefficients_start_and_input_unchanged(self):
        before = copy.deepcopy(self.case)
        programs, duration = case_runner.prepare_references(self.case, 'defended')
        self.assertEqual(duration, 4.)
        for slot, program in programs.items():
            self.assertEqual(program.start_s, self.case['programs'][slot]['start_s'])
            for key in ('knot_time_s', 'knot_qpos', 'coefficients'):
                self.assertTrue(np.array_equal(getattr(program.curve, key), self.case['programs'][slot]['curve'][key]))
            self.assertTrue(np.array_equal(program.target(0), self.case['initial_qpos'][slot]))
        self.assertEqual(self.case, before)

    def test_baseline_only_defender_becomes_exact_same_initial_hold(self):
        defended, duration = case_runner.prepare_references(self.case, 'defended')
        baseline, baseline_duration = case_runner.prepare_references(self.case, 'baseline')
        self.assertEqual(duration, baseline_duration)
        for key in ('knot_time_s', 'knot_qpos', 'coefficients'):
            np.testing.assert_array_equal(getattr(defended['a1'].curve, key), getattr(baseline['a1'].curve, key))
        self.assertEqual(baseline['a1'].start_s, defended['a1'].start_s)
        self.assertEqual(baseline['a2'].start_s, defended['a2'].start_s)
        self.assertEqual(baseline['a2'].end_s, defended['a2'].end_s)
        for time in (-1., 0., .4, 1., 4., 10.):
            np.testing.assert_array_equal(baseline['a2'].target(time), self.case['initial_qpos']['a2'])

    def test_swapped_roles_hold_the_explicit_defender_not_slot_name(self):
        case = copy.deepcopy(self.case)
        case.update(attacker_slot='a2', defender_slot='a1')
        programs, _ = case_runner.prepare_references(case, 'baseline')
        self.assertEqual(len(programs['a1'].curve.knot_time_s), 2)
        np.testing.assert_array_equal(programs['a2'].curve.coefficients, case['programs']['a2']['curve']['coefficients'])

    def test_tampered_coefficients_rejected_even_in_baseline(self):
        for variant in case_runner.VARIANTS:
            case = copy.deepcopy(self.case)
            case['programs']['a2']['curve']['coefficients'][0][1][0] += 1e-12
            with self.subTest(variant=variant), self.assertRaisesRegex(ValueError, 'coefficients differ'):
                case_runner.prepare_references(case, variant)

    def test_bad_reference_schema_and_clock_rejected(self):
        changes = (
            lambda c: c.update(attacker_slot='a2'),
            lambda c: c['programs'].pop('a2'),
            lambda c: c['programs']['a1'].update(start_s=-1.),
            lambda c: c['programs']['a1'].update(start_s=True),
            lambda c: c['programs']['a1']['curve']['knot_time_s'].__setitem__(0, .1),
            lambda c: c['initial_qpos']['a1'].__setitem__(0, 1.0000000001),
            lambda c: c['programs']['a1']['curve'].pop('coefficients'),
            lambda c: c.update(duration_s=2.25),
            lambda c: c.update(duration_s=61.),
            lambda c: c.update(admitted_to_training=True),
            lambda c: c['programs']['a1']['curve']['knot_qpos'][0].__setitem__(0, '1'),
            lambda c: c['programs']['a1']['curve']['knot_qpos'][0].__setitem__(0, True))
        for change in changes:
            case = copy.deepcopy(self.case)
            change(case)
            with self.subTest(change=change), self.assertRaises((ValueError, TypeError)):
                case_runner.prepare_references(case, 'defended')
        for variant in ('other', None, True):
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                case_runner.prepare_references(self.case, variant)

    def test_package_binding_and_nonadmission_metadata_without_native(self):
        result = self.prepared()
        self.assertEqual(result.target_region, 'left')
        self.assertEqual(result.metadata['release_manifest_sha256'],
                         hashlib.sha256((self.root / 'manifest.json').read_bytes()).hexdigest())
        for field in ('autonomous_reaction', 'trained_policy', 'admitted_to_training', 'pair_comparison_performed'):
            self.assertIs(result.metadata[field], False)
        self.assertFalse(result.metadata['physical_binding']['physical_contract_bound'])
        self.assertFalse(result.metadata['physical_binding_verified'])
        self.assertFalse((Path(self.temp.name) / 'native').exists())

    def add_physical_fixture(self):
        repo = Path(self.temp.name) / 'repo'
        (repo / 'config').mkdir(parents=True)
        (repo / 'config/a1_runtime_rk4_candidate.json').write_bytes(
            (case_runner.ROOT / 'config/a1_runtime_rk4_candidate.json').read_bytes())
        (repo / 'config/assembly.json').write_bytes((case_runner.ROOT / 'config/assembly.json').read_bytes())
        (repo / 'model.xml').write_bytes(b'authored-test-model-not-compiled')
        contract = dict(runtime_config=self.runtime,
            model_source_sha256={'model.xml': hashlib.sha256((repo / 'model.xml').read_bytes()).hexdigest()},
            torque_caps_nm=[5, 80, 15, 20, 3, 3, 2])
        write_json(self.root / 'physical_contract.json', contract)
        self.manifest['physical_contract_file'] = 'physical_contract.json'
        self.update_package()
        return repo, contract

    def test_physical_contract_binds_model_and_caps_without_native(self):
        repo, contract = self.add_physical_fixture()
        with patch.object(case_runner, 'ROOT', repo):
            prepared = self.prepared()
        binding = prepared.metadata['physical_binding']
        self.assertTrue(binding['physical_contract_bound'])
        self.assertTrue(prepared.metadata['physical_binding_verified'])
        self.assertEqual(binding['model_source_sha256'], contract['model_source_sha256'])
        self.assertEqual(binding['torque_caps_nm'], contract['torque_caps_nm'])

    def test_changed_model_source_rejected_before_native(self):
        repo, _ = self.add_physical_fixture()
        (repo / 'model.xml').write_bytes(b'changed-model')
        with patch.object(case_runner, 'ROOT', repo), \
                self.assertRaisesRegex(ValueError, 'model source differs'):
            self.prepared()

    def test_changed_torque_cap_rejected_before_native(self):
        repo, _ = self.add_physical_fixture()
        assembly_path = repo / 'config/assembly.json'
        assembly = json.loads(assembly_path.read_text())
        assembly['runtime']['torque_caps_nm'][4] = 3.1
        write_json(assembly_path, assembly)
        with patch.object(case_runner, 'ROOT', repo), \
                self.assertRaisesRegex(ValueError, 'torque caps differ'):
            self.prepared()

    def test_physical_source_path_cannot_escape_repository(self):
        repo, contract = self.add_physical_fixture()
        contract['model_source_sha256'] = {'../model.xml': '0' * 64}
        write_json(self.root / 'physical_contract.json', contract)
        self.update_package()
        with patch.object(case_runner, 'ROOT', repo), \
                self.assertRaisesRegex(ValueError, 'escapes the repository'):
            self.prepared()

    def test_stale_runtime_and_unphysical_case_are_rejected(self):
        self.cases['runtime_config']['base_distance_m'] += .01
        self.update_package()
        with self.assertRaisesRegex(ValueError, 'runtime differs'):
            self.prepared()
        self.cases['runtime_config'] = load_runtime_profile(
            case_runner.ROOT / 'config/a1_runtime_rk4_candidate.json').runtime_config()
        self.case['initial_qpos']['a2'][0] = -1.
        self.case['programs']['a2']['curve'] = curve_row([0., 1.],
            [self.case['initial_qpos']['a2']] * 2)
        self.update_package()
        # Baseline cannot hide an invalid source defender.
        with self.assertRaises(ValueError):
            self.prepared('baseline')

    def test_missing_unknown_case_or_nominal_region_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unknown packaged case'):
            case_runner.prepare_case('absent', 'baseline', self.root)
        del self.case['target_region']
        self.update_package()
        with self.assertRaisesRegex(ValueError, 'target_region'):
            self.prepared()

    def test_all_six_actual_package_cases_exact_for_both_variants(self):
        cases = json.loads((DEFAULT_PACKAGE / 'cases.json').read_text())['cases']
        self.assertEqual(len(cases), 6)
        for name, source in cases.items():
            for variant in case_runner.VARIANTS:
                with self.subTest(case=name, variant=variant):
                    result = case_runner.prepare_case(name, variant)
                    self.assertEqual(result.duration_s, source['duration_s'])
                    attacker = source['attacker_slot']
                    for key in ('knot_time_s', 'knot_qpos', 'coefficients'):
                        np.testing.assert_array_equal(getattr(result.programs[attacker].curve, key),
                            source['programs'][attacker]['curve'][key])
                    self.assertEqual(result.programs[attacker].start_s, source['programs'][attacker]['start_s'])

    def fake_native(self, programs, output, *, duration_s, metadata, progress):
        output = Path(output)
        output.mkdir()
        summary = dict(runtime_config=metadata['runtime_config'],
            program_start_s={s: p.start_s for s, p in programs.items()},
            requested_duration_s=duration_s, native_execution_pass=True, failure=None,
            geometry_first_touch_diagnostic={'not_a_three_region_verdict': True})
        write_json(output / 'summary.json', summary)
        np.savez(output / 'references.npz', **{
            s + '_' + key: getattr(p.curve, key) for s, p in programs.items()
            for key in ('knot_time_s', 'knot_qpos', 'coefficients')})
        return summary

    def test_exact_saved_reference_binding_detects_changed_arrays(self):
        prepared = self.prepared()
        output = Path(self.temp.name) / 'fake_native'
        self.fake_native(prepared.programs, output, duration_s=prepared.duration_s,
                         metadata=prepared.metadata, progress=None)
        self.assertTrue(case_runner.verify_native_references(output, prepared))
        with np.load(output / 'references.npz') as saved:
            arrays = {k: saved[k].copy() for k in saved.files}
        arrays['a2_coefficients'][0, 0, 0] += 1e-12
        np.savez(output / 'references.npz', **arrays)
        with self.assertRaisesRegex(ValueError, 'reference differs'):
            case_runner.verify_native_references(output, prepared)

    def test_run_orchestration_preserves_duration_and_saves_fresh_all_region_audit(self):
        output = Path(self.temp.name) / 'trial'
        audit = dict(full_recorded_duration_s=4., initial_overlap={}, sweep_failures={},
                     all_entries=[dict(agent='a1', region='right', time_s=2.)])
        with patch.object(case_runner, 'run_constrained_trial', side_effect=self.fake_native) as native, \
                patch.object(case_runner, 'audit_recorded_triangle', return_value=audit) as auditor:
            result = case_runner.run_case('example', 'baseline', output, self.root, progress=None)
        self.assertEqual(native.call_args.kwargs['duration_s'], 4.)
        self.assertEqual(auditor.call_args.args, (output / 'native', self.scoring, 'left'))
        self.assertEqual(auditor.call_args.kwargs, {'attacker': 'a1'})
        self.assertEqual(json.loads((output / 'triangle_audit.json').read_text()), audit)
        self.assertEqual(result['attacker_entry_count'], 1)
        self.assertFalse(result['zero_entries_observed'])
        self.assertFalse(result['attack_or_defence_certified'])
        self.assertFalse(result['single_target_native_result_used_for_triangle_success'])
        self.assertFalse(result['packaged_physical_binding_verified'])
        self.assertTrue(result['whole_window_three_region_audit_valid'])
        self.assertTrue((output / 'request.json').is_file())

    def test_audit_failure_preserved_not_replaced_by_native_single_target_result(self):
        output = Path(self.temp.name) / 'failed_audit'
        with patch.object(case_runner, 'run_constrained_trial', side_effect=self.fake_native), \
                patch.object(case_runner, 'audit_recorded_triangle', side_effect=ValueError('incomplete trace')):
            result = case_runner.run_case('example', 'defended', output, self.root, progress=None)
        self.assertFalse(result['whole_window_three_region_audit_valid'])
        self.assertFalse(result['zero_entries_observed'])
        self.assertEqual(result['errors'][0]['stage'], 'triangle_audit')
        self.assertTrue((output / 'audit_errors.json').is_file())
        self.assertEqual(result['status'], 'incomplete_or_invalid_diagnostic')

    def test_existing_output_refused_before_native_or_preparation(self):
        output = Path(self.temp.name) / 'already_exists'
        output.mkdir()
        with patch.object(case_runner, 'run_constrained_trial') as native, \
                patch.object(case_runner, 'prepare_case') as prepare:
            with self.assertRaises(FileExistsError):
                case_runner.run_case('example', 'defended', output, self.root)
        native.assert_not_called()
        prepare.assert_not_called()

    def test_cli_forwards_arguments_without_native(self):
        result = dict(whole_window_three_region_audit_valid=True)
        with patch.object(case_runner, 'run_case', return_value=result) as run, \
                contextlib.redirect_stdout(io.StringIO()):
            code = case_runner.main(['--case', 'example', '--variant', 'defended',
                '--output', '/tmp/not-created-by-this-test', '--package', str(self.root)])
        self.assertEqual(code, 0)
        run.assert_called_once_with('example', 'defended', Path('/tmp/not-created-by-this-test'), self.root)


if __name__ == '__main__':
    unittest.main()
