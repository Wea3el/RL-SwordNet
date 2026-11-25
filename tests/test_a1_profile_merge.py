import copy
import json
from pathlib import Path
import tempfile
import unittest

from sim.a1_profile_merge import merge


class CandidateMergeTests(unittest.TestCase):
    def sources(self, path):
        q = [1., 1., 1., 1., 1., 1., 0.]
        row = {'start_guard': 'g', 'end_guard': 'g', 'knot_time_s': [0., 1.],
               'knot_qpos': [q, q], 'candidate_pass': True}
        data = {'schema_version': 1, 'guards': {'g': {'qpos_rad': q}},
                'actions': {'a': row, 'b': copy.deepcopy(row)}, 'scoring': {'id': 'v1'},
                'blade_geometry': {'sha256': 'geom'},
                'retiming_provenance': {'all_qpos_exactly_preserved': True}}
        base = path/'base.json'
        source = path/'replacement.json'
        base.write_text(json.dumps(data))
        data['actions']['a']['knot_time_s'] = [0., 2.]
        source.write_text(json.dumps(data))
        return base, source

    def test_merge_exact_reference_without_copying_certificate(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            base, source = self.sources(path)
            before = {p: p.read_bytes() for p in (base, source)}
            result = merge(base, [('a', source)], path/'out')
            self.assertEqual(result['actions']['a']['knot_time_s'], [0., 2.])
            self.assertEqual(result['actions']['b']['knot_time_s'], [0., 1.])
            self.assertFalse(result['admitted_to_training'])
            self.assertNotIn('retiming_provenance', result)
            self.assertEqual(before, {p: p.read_bytes() for p in (base, source)})
            with self.assertRaises(FileExistsError):
                merge(base, [('a', source)], path/'out')

    def test_rejects_scoring_or_geometry_change(self):
        for key in ('scoring', 'blade_geometry'):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as folder:
                path = Path(folder)
                base, source = self.sources(path)
                data = json.loads(source.read_text())
                data[key]['changed'] = True
                source.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    merge(base, [('a', source)], path/'out')

    def test_rejects_guard_conflict_and_unmatched_endpoint(self):
        for conflict in (True, False):
            with self.subTest(conflict=conflict), tempfile.TemporaryDirectory() as folder:
                path = Path(folder)
                base, source = self.sources(path)
                data = json.loads(source.read_text())
                if conflict:
                    data['guards']['g']['qpos_rad'][0] += .1
                else:
                    data['actions']['a']['knot_qpos'][0][0] += .1
                source.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    merge(base, [('a', source)], path/'out')

    def test_rejects_duplicate_replacement(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            base, source = self.sources(path)
            with self.assertRaises(ValueError):
                merge(base, [('a', source), ('a', source)], path/'out')

    def test_explicit_addition_preserves_existing_and_requires_migration(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder);base,source=self.sources(path)
            data=json.loads(source.read_text())
            data['actions']['parry_right_thrust']=dict(data['actions']['a'],kind='defence',family='parry_right')
            source.write_text(json.dumps(data))
            result=merge(base,[],path/'out',additions=[('parry_right_thrust',source)])
            self.assertEqual(result['actions']['a']['knot_time_s'],[0.,1.])
            self.assertIn('parry_right_thrust',result['actions'])
            self.assertTrue(result['composition']['runtime_and_snapshot_migration_required'])
            self.assertFalse(result['admitted_to_training'])

    def test_addition_cannot_silently_replace_or_invent_family(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder);base,source=self.sources(path)
            with self.assertRaises(ValueError):
                merge(base,[],path/'exists',additions=[('a',source)])
            data=json.loads(source.read_text())
            data['actions']['bad']=dict(data['actions']['a'],kind='defence',family='new_family')
            source.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                merge(base,[],path/'bad',additions=[('bad',source)])

    def test_native_research_permission_cannot_be_merged_or_used_as_base(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder);base,source=self.sources(path)
            data=json.loads(source.read_text())
            data['actions']['a']['candidate_stage']='authorized_native_research_hypothesis'
            source.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                merge(base,[('a',source)],path/'out')
            with self.assertRaises(ValueError):
                merge(source,[],path/'base_out')
