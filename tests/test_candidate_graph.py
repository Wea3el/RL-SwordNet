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
        self.path = ROOT / 'config/a1_shared_ready_glancing_seed.json'
        self.cfg = json.loads(self.path.read_text())

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

    def test_direction_candidates_keep_closed_ready_paths_without_old_actor_compatibility(self):
        path = ROOT / 'config/a1_direction_candidates_v1.json'
        cfg = json.loads(path.read_text())
        self.assertIs(cfg['legacy_actor_compatible'], False)
        self.assertIs(cfg['admitted_to_training'], False)
        executor = load_candidate_graph(path).new_executor()
        self.assertEqual(set(executor.actions),
                         {'attack_original', 'parry_original', 'sweep_down', 'sweep_up', 'parry_low'})
        for action in executor.actions.values():
            self.assertEqual((action.start_node, action.end_node), ('ready', 'ready'))
            np.testing.assert_array_equal(action.curve.sample(0.), executor.initial_q)
            np.testing.assert_array_equal(action.curve.sample(action.curve.duration_s), executor.initial_q)

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
