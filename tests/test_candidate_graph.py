"""Numerical loader/executor interfaces only, never physical action evidence.

Positive fixtures are authored here inside the current confirmed limits.
Historical robot-motion configs remain unchanged and are rejection controls.
"""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from sim.assembly import ROOT
from sim.candidate_graph import load_candidate_graph


class CandidateGraphTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'synthetic_interface_fixture.json'
        self.cfg = {
            'schema_version': 1, 'policy_family': 'A1',
            'status': 'synthetic_interface_test_not_physical_evidence',
            'admitted_to_training': False, 'legacy_actor_compatible': False,
            'joint_names': [f'J{i}' for i in range(7)], 'angle_unit': 'radian',
            'joint_constraints': 'config/joint_constraints_observed.json',
            'reference_generator': 'sim.motion_spline.C2JointSpline',
            'initial_node': 'ready',
            'nodes': {
                'ready': [1., .5, 1., .5, 1., .5, 0.],
                'strike': [1.2, .6, 1.1, .6, 1., .6, .1],
                'glancing_intercept': [.9, .55, 1.1, .4, 1.1, .6, -.1]},
            'actions': {
                'attack': {'start_node': 'ready', 'end_node': 'ready',
                           'knot_time_s': [0., 3., 6.],
                           'waypoint_nodes': ['ready', 'strike', 'ready']},
                'parry': {'start_node': 'ready', 'end_node': 'ready',
                          'knot_time_s': [0., 1., 3., 4.5],
                          'waypoint_nodes': ['ready', 'glancing_intercept',
                                             'glancing_intercept', 'ready']}}}
        self.path.write_text(json.dumps(self.cfg), encoding='utf-8')

    def load_changed(self, change):
        cfg = copy.deepcopy(self.cfg)
        change(cfg)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'candidate.json'
            path.write_text(json.dumps(cfg))
            return load_candidate_graph(path)

    def test_reference_curves_and_independent_executors(self):
        candidate = load_candidate_graph(self.path)
        self.assertEqual(candidate.source_sha256, hashlib.sha256(self.path.read_bytes()).hexdigest())
        first, second = candidate.new_executor(), candidate.new_executor()
        self.assertIsNot(first, second)
        self.assertFalse(first.ready)
        self.assertFalse(second.ready)
        np.testing.assert_array_equal(first.initial_q, self.cfg['nodes']['ready'])
        self.assertEqual(tuple(first.actions), ('attack', 'parry'))
        self.assertEqual(first.actions['attack'].curve.duration_s, 6.)
        self.assertEqual(first.actions['parry'].curve.duration_s, 4.5)
        first.observe(0., first.initial_q, np.zeros(7))
        first.observe(.1, first.initial_q, np.zeros(7))
        first.request('attack', .1)
        self.assertIsNone(second.active)
        self.assertEqual(second.events, [])

    def test_metadata_is_not_training_permission(self):
        with self.assertRaises(ValueError):
            self.load_changed(lambda c: c.update(admitted_to_training=True))

    def test_historical_configs_rejected_by_current_J3_mapping(self):
        for name in ('a1_shared_ready_glancing_seed.json', 'a1_direction_candidates_v1.json'):
            path = ROOT / 'config' / name
            with self.subTest(config=name), self.assertRaisesRegex(ValueError, r'J3.*128\.151'):
                load_candidate_graph(path)

    def test_joint_order_and_units_cannot_be_guessed(self):
        for change in (lambda c: c.update(angle_unit='degree'),
                       lambda c: c['joint_names'].reverse(),
                       lambda c: c.update(joint_names={n: None for n in c['joint_names']}),
                       lambda c: c.update(joint_constraints='another.json')):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.load_changed(change)

    def test_initial_node_requires_a_known_string(self):
        for initial in ([], {}, None, 'missing'):
            with self.subTest(initial=initial), self.assertRaises(ValueError):
                self.load_changed(lambda c: c.update(initial_node=initial))

    def test_missing_waypoint_or_wrong_endpoint_rejected(self):
        for change in (lambda c: c['actions']['attack']['waypoint_nodes'].__setitem__(1, 'missing'),
                       lambda c: c['actions']['attack'].update(end_node='strike')):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.load_changed(change)

    def test_out_of_range_no_clip_or_modulo(self):
        with self.assertRaises(ValueError):
            self.load_changed(lambda c: c['nodes']['strike'].__setitem__(1, -.01))

    def test_boolean_or_string_coordinates_and_times_rejected(self):
        for change in (lambda c: c['nodes']['strike'].__setitem__(1, True),
                       lambda c: c['nodes']['strike'].__setitem__(1, '0.6'),
                       lambda c: c['actions']['attack']['knot_time_s'].__setitem__(0, False)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.load_changed(change)

    def test_disconnected_entry_is_not_a_selectable_skill(self):
        def change(cfg):
            cfg['actions']['attack'].update(start_node='strike', end_node='strike',
                waypoint_nodes=['strike', 'strike'], knot_time_s=[0., 1.])
        with self.assertRaises(ValueError):
            self.load_changed(change)

    def test_duplicate_names_are_not_silently_overwritten(self):
        source = self.path.read_text().replace('"schema_version": 1,',
                                              '"schema_version": 2, "schema_version": 1,')
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'duplicate.json'
            path.write_text(source)
            with self.assertRaises(ValueError):
                load_candidate_graph(path)


if __name__ == '__main__':
    unittest.main()
