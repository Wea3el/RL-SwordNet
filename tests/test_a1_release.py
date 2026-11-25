"""Package integrity and explicit graph semantics; no integration or native steps."""
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

from sim import a1_release
from sim.motion_spline import C2JointSpline


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class A1ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'package'
        self.root.mkdir()
        ready = [1., .5, 1., .5, 1., .5, 0.]
        other = [1.1, .5, 1., .5, 1., .5, 0.]
        self.catalog = dict(schema_version=1, admitted_to_training=False,
            guards={'ready': {'qpos_rad': ready}, 'other': {'qpos_rad': other}},
            initial_guard='ready', policy_order=['go', 'back'],
            exact_guard_aliases={'old_ready_name': 'ready'},
            actions={
                'go': dict(kind='attack', start_guard='ready', end_guard='other',
                           knot_time_s=[0., 1.], knot_qpos=[ready, other]),
                'back': dict(kind='transition', start_guard='other', end_guard='ready',
                             knot_time_s=[0., 1.], knot_qpos=[other, ready])})
        # JSON files do not alias a guard's list with an action's endpoint list.
        self.catalog = json.loads(json.dumps(self.catalog))
        self.write_json('catalog.json', self.catalog)
        self.write_json('cases.json', {'schema_version': 1, 'cases': {}})
        (self.root / 'joint_constraints_observed.json').write_bytes(
            a1_release.DEFAULT_CONSTRAINTS_PATH.read_bytes())
        self.manifest = dict(schema_version=1, status=a1_release.MANIFEST_STATUS,
            legacy_actor_compatible=False, admitted_to_training=False,
            catalog_file='catalog.json', cases_file='cases.json',
            joint_constraints_file='joint_constraints_observed.json',
            files_sha256={}, source_fingerprints={'outputs/absent_original.json': '0' * 64},
            counts=dict(catalog_edges=2, named_nodes=2, unique_q=2, policy_outputs=3),
            limitations=['Not training admission.'])
        self.rehash()

    def write_json(self, name, value):
        (self.root / name).write_text(json.dumps(value, allow_nan=False), encoding='utf-8')

    def rehash(self):
        self.manifest['files_sha256'] = {
            path.relative_to(self.root).as_posix(): sha(path.read_bytes())
            for path in self.root.rglob('*') if path.is_file() and path.name != 'manifest.json'}
        self.write_json('manifest.json', self.manifest)

    def change_catalog(self, change):
        cfg = copy.deepcopy(self.catalog)
        change(cfg)
        self.write_json('catalog.json', cfg)
        self.rehash()
        return a1_release.load_release(self.root)

    def test_inspect_has_no_admission_and_ordered_actions(self):
        release = a1_release.load_release(self.root)
        report = release.inspect()
        self.assertEqual(release.action_names, ('wait', 'go', 'back'))
        self.assertEqual([r['action_id'] for r in report['actions']], [1, 2])
        self.assertEqual(report['policy_outputs_including_wait'], 3)
        self.assertFalse(report['legacy_actor_compatible'])
        self.assertFalse(report['admitted_to_training'])
        self.assertEqual(report['native_steps'], 0)
        self.assertIn('candidate_not_training_admission', report['status'])
        self.assertEqual(report['manifest_sha256'], sha((self.root / 'manifest.json').read_bytes()))
        # Original provenance files need not exist, and are never opened.
        self.assertFalse((a1_release.ROOT / 'outputs/absent_original.json').exists())

    def test_loading_inspection_and_executor_never_step_physics(self):
        with patch('mujoco.mj_step', side_effect=AssertionError('native forbidden')):
            release = a1_release.load_release(self.root)
            release.inspect()
            release.new_executor()

    def test_executor_requires_actual_100ms_gate_and_explicit_edge(self):
        executor = a1_release.load_release(self.root).new_executor()
        self.assertFalse(executor.ready)
        self.assertEqual(executor.available(), ())
        with self.assertRaises(ValueError):
            executor.request('go', 0.)
        for time in (0., .099):
            executor.observe(time, executor.initial_q, np.zeros(7))
            self.assertEqual(executor.available(), ())
        executor.observe(.1, executor.initial_q, np.zeros(7))
        self.assertEqual(executor.available(), ('go',))
        with self.assertRaises(ValueError):
            executor.request('back', .1)
        executor.request('go', .1)
        self.assertEqual(executor.available(), ())
        qend = executor.nodes['other']
        executor.observe(1.1, qend, np.zeros(7))
        self.assertEqual(executor.available(), ())
        executor.observe(1.2, qend, np.zeros(7))
        self.assertEqual(executor.available(), ('back',))
        self.assertEqual(executor.node, 'other')

    def test_independent_state_and_fresh_manifest_copies(self):
        release = a1_release.load_release(self.root)
        first, second = release.new_executor(), release.new_executor()
        first.observe(0., first.initial_q, np.zeros(7))
        first.observe(.1, first.initial_q, np.zeros(7))
        first.request('go', .1)
        self.assertIsNone(second.active)
        self.assertEqual(second.events, [])
        first.nodes['ready'][0] = 2.
        self.assertEqual(second.nodes['ready'][0], 1.)
        release.manifest['admitted_to_training'] = True
        self.assertFalse(release.manifest['admitted_to_training'])

    def test_equal_coordinates_do_not_infer_node_or_alias_bridges(self):
        self.manifest['counts'].update(named_nodes=3)
        release = self.change_catalog(lambda c: c['guards'].update(
            context_ready={'qpos_rad': list(c['guards']['ready']['qpos_rad'])}))
        executor = release.new_executor('context_ready')
        executor.observe(0., executor.initial_q, np.zeros(7))
        executor.observe(.1, executor.initial_q, np.zeros(7))
        self.assertTrue(executor.ready)
        self.assertEqual(executor.available(), ())
        for initial in ('old_ready_name', 'missing', [], None):
            if initial is None:  # Default is the explicit initial node, not an alias.
                self.assertEqual(release.new_executor().node, 'ready')
            else:
                with self.subTest(initial=initial), self.assertRaises(ValueError):
                    release.new_executor(initial)

    def test_missing_tampered_and_unhashed_content_rejected(self):
        for kind in ('tampered', 'missing', 'extra'):
            with self.subTest(kind=kind):
                path = self.root / ('extra.json' if kind == 'extra' else 'catalog.json')
                old = path.read_bytes() if path.exists() else None
                if kind == 'missing':
                    path.unlink()
                else:
                    path.write_bytes(b'{}')
                with self.assertRaises(ValueError):
                    a1_release.load_release(self.root)
                if old is None:
                    path.unlink()
                else:
                    path.write_bytes(old)

    def test_required_hashes_and_valid_digests(self):
        original = copy.deepcopy(self.manifest)
        for change in (
                lambda m: m['files_sha256'].pop('cases.json'),
                lambda m: m['files_sha256'].update({'catalog.json': 'bad'}),
                lambda m: m['files_sha256'].update({'manifest.json': '0' * 64}),
                lambda m: m.update(catalog_file='cases.json')):
            self.manifest = copy.deepcopy(original)
            change(self.manifest)
            self.write_json('manifest.json', self.manifest)
            with self.subTest(change=change), self.assertRaises(ValueError):
                a1_release.load_release(self.root)

    def test_package_paths_and_provenance_cannot_escape(self):
        original = copy.deepcopy(self.manifest)
        for name in ('../escape.json', '/tmp/escape.json', 'a/../../escape',
                     'a\\escape', 'a//escape', './escape', '.'):
            for field in ('files_sha256', 'source_fingerprints'):
                self.manifest = copy.deepcopy(original)
                self.manifest[field][name] = '0' * 64
                self.write_json('manifest.json', self.manifest)
                with self.subTest(field=field, name=name), self.assertRaises(ValueError):
                    a1_release.load_release(self.root)

    def test_symlink_to_outside_is_not_hashable_package_content(self):
        outside = Path(self.temp.name) / 'outside.json'
        outside.write_bytes(b'{}')
        (self.root / 'link.json').symlink_to(outside)
        self.manifest['files_sha256']['link.json'] = sha(b'{}')
        self.write_json('manifest.json', self.manifest)
        with self.assertRaises(ValueError):
            a1_release.load_release(self.root)

    def test_legacy_and_admitted_formats_are_rejected(self):
        original = copy.deepcopy(self.manifest)
        for change in (lambda m: m.update(status='qualified'),
                       lambda m: m.update(legacy_actor_compatible=True),
                       lambda m: m.update(admitted_to_training=True),
                       lambda m: m.update(schema_version=True)):
            self.manifest = copy.deepcopy(original)
            change(self.manifest)
            self.write_json('manifest.json', self.manifest)
            with self.subTest(change=change), self.assertRaises(ValueError):
                a1_release.load_release(self.root)
        self.write_json('bank.json', {'schema': 'a1_closed_numeric_bank_v1'})
        (self.root / 'manifest.json').unlink()
        with self.assertRaises(ValueError):
            a1_release.load_release(self.root)
        with self.assertRaises(ValueError):
            a1_release.load_release(self.root / 'bank.json')

    def test_catalog_endpoints_policy_order_and_counts_are_exact(self):
        changes = (
            lambda c: c['actions']['go']['knot_qpos'][0].__setitem__(0, 1.00000000000001),
            lambda c: c['actions']['go'].update(start_guard='missing'),
            lambda c: c.update(policy_order=['go', 'go']),
            lambda c: c.update(initial_guard=[]),
            lambda c: c.update(admitted_to_training=True))
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.change_catalog(change)
        self.write_json('catalog.json', self.catalog)
        self.manifest['counts']['catalog_edges'] = 99
        self.rehash()
        with self.assertRaises(ValueError):
            a1_release.load_release(self.root)

    def test_joint_limits_speed_and_numeric_coercion_rejected(self):
        changes = (
            lambda c: c['guards']['ready']['qpos_rad'].__setitem__(0, -0.001),
            lambda c: c['actions']['go'].update(knot_time_s=[0., .01]),
            lambda c: c['actions']['go']['knot_time_s'].__setitem__(0, False),
            lambda c: c['actions']['go']['knot_qpos'][0].__setitem__(0, True),
            lambda c: c['actions']['go']['knot_qpos'][0].__setitem__(0, '1'),
            lambda c: c['guards']['ready']['qpos_rad'].__setitem__(0, True))
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.change_catalog(change)

    def test_stored_coefficients_require_exact_equality(self):
        coeff = C2JointSpline([0., 1.], self.catalog['actions']['go']['knot_qpos']).coefficients.tolist()
        release = self.change_catalog(lambda c: c['actions']['go'].update(coefficients=coeff))
        self.assertEqual(release.action_names, ('wait', 'go', 'back'))
        coeff[0][0][0] += 1e-14
        with self.assertRaises(ValueError):
            self.change_catalog(lambda c: c['actions']['go'].update(coefficients=coeff))

    def test_nonfinite_and_duplicate_json_rejected_even_with_new_hash(self):
        for payload in (b'{"schema_version":1,"schema_version":1}',
                        b'{"schema_version":1,"bad":NaN}'):
            (self.root / 'cases.json').write_bytes(payload)
            self.rehash()
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                a1_release.load_release(self.root)

    def test_executor_rejects_changed_current_constraints(self):
        release = a1_release.load_release(self.root)
        different = Path(self.temp.name) / 'different_constraints.json'
        different.write_bytes(a1_release.DEFAULT_CONSTRAINTS_PATH.read_bytes() + b'\n')
        with patch.object(a1_release, 'DEFAULT_CONSTRAINTS_PATH', different), self.assertRaises(ValueError):
            release.new_executor()
        # Read-only inspection still reports the packaged constraints, not a modified runtime.
        self.assertEqual(release.inspect()['named_nodes'], 2)

    def test_loaded_package_cannot_silently_change(self):
        release = a1_release.load_release(self.root)
        self.manifest['limitations'].append('changed')
        self.write_json('manifest.json', self.manifest)
        for method in (release.inspect, release.new_executor):
            with self.subTest(method=method), self.assertRaises(ValueError):
                method()

    def test_cli_default_and_explicit_package(self):
        for argv in (['inspect'], ['inspect', '--package', str(self.root)]):
            output = io.StringIO()
            with patch.object(a1_release, 'DEFAULT_PACKAGE', self.root), contextlib.redirect_stdout(output):
                self.assertEqual(a1_release.main(argv), 0)
            self.assertEqual(json.loads(output.getvalue())['policy_outputs_including_wait'], 3)

    def test_packaged_release_smoke_without_original_outputs(self):
        """This public package is self-contained; copying it drops all original outputs."""
        actual = a1_release.DEFAULT_PACKAGE
        if not actual.exists():
            self.skipTest('Development asset package is not present in this checkout')
        target = Path(self.temp.name) / 'public_package_copy'
        target.mkdir()
        for path in actual.iterdir():
            if path.is_file():
                (target / path.name).write_bytes(path.read_bytes())
        with patch('mujoco.mj_step', side_effect=AssertionError('native forbidden')):
            release = a1_release.load_release(target)
            report = release.inspect()
            self.assertEqual(report['named_nodes'], 13)
            self.assertEqual(report['distinct_joint_poses'], 12)
            self.assertEqual(report['policy_outputs_including_wait'], 35)
            self.assertEqual(len(release.action_names), 35)
            self.assertFalse(release.new_executor().ready)
            self.assertFalse(report['admitted_to_training'])


if __name__ == '__main__':
    unittest.main()
