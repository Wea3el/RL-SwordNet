"""Contract tests use synthetic in-range curves, not fencing certification."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from sim.multiguard_bank import (ACTION_NAMES, GUARD_NAMES, MultiGuardController,
                                bank_summary, export_geometry_bank,
                                geometry_report_to_payload, load_multiguard_bank)


def fixture():
    origin = np.asarray([3., .7, 3., 1., 2., 1., 0.])
    nodes = {name: (origin+np.asarray([i*.02, i*.01, 0., 0., 0., 0., 0.])).tolist()
             for i, name in enumerate(GUARD_NAMES)}
    cfg = {'schema_version': 1, 'bank_kind': 'multiguard_v1_candidate',
           'policy_family': 'A1', 'admitted_to_training': False, 'legacy_actor_compatible': False,
           'joint_names': [f'J{i}' for i in range(7)], 'angle_unit': 'radian',
           'joint_constraints': 'config/joint_constraints_observed.json',
           'reference_generator': 'sim.motion_spline.C2JointSpline', 'initial_node': 'center',
           'action_ids': {str(i): name for i, name in enumerate(ACTION_NAMES)},
           'guard_nodes': {name: name for name in GUARD_NAMES}, 'nodes': nodes,
           'actions': {}, 'families': {}, 'control_transitions': [], 'source_sha256': {}}

    def edge(name, start, end):
        cfg['actions'][name] = {'start_node': start, 'end_node': end,
                                'waypoint_nodes': [start, end], 'knot_time_s': [0., 1.]}

    for i, name in enumerate(ACTION_NAMES[1:], 1):
        start, end = GUARD_NAMES[(i-1) % 5], GUARD_NAMES[i % 5]
        edge(name, start, end)
        cfg['families'][name] = {'id': i, 'kind': 'defence' if i >= 7 else 'attack',
                                 'routes': {start: name}}
    # Same policy family can name a different explicitly authored motion at a
    # different guard. This is graph routing, not geometry validation.
    edge('cut_down_repeat', 'right_high', 'left_low')
    cfg['families'][ACTION_NAMES[1]]['routes']['right_high'] = 'cut_down_repeat'
    for i, start in enumerate(GUARD_NAMES):
        end = GUARD_NAMES[(i+1) % 5]
        name = f'reposition_{start}'
        edge(name, start, end)
        cfg['control_transitions'].append(name)
    return cfg


class MultiGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)/'bank.json'
        self.cfg = fixture()
        self.save()

    def tearDown(self):
        self.temp.cleanup()

    def save(self):
        self.path.write_text(json.dumps(self.cfg))

    def settle(self, controller, start=0.):
        q = controller.executor.target(start)
        controller.observe(start, q, np.zeros(7))
        controller.observe(start+.1, q, np.zeros(7))

    def finish(self, controller, start):
        destination = controller.executor.actions[controller.executor.active].end_node
        q = controller.executor.nodes[destination]
        controller.observe(start+1., q, np.zeros(7))
        self.assertIsNotNone(controller.executor.active)
        controller.observe(start+1.1, q, np.zeros(7))

    def test_full_registry_and_immutable_independent_execution(self):
        bank = load_multiguard_bank(self.path, require_complete=True)
        self.assertEqual(bank.missing_families, ())
        self.assertEqual(bank.unreachable_families, ())
        self.assertEqual(bank.policy_unreachable_families, ())
        self.assertEqual(bank.policy_dead_end_guards, frozenset())
        self.assertTrue(all(not lost for _, lost in bank.policy_limited_guards))
        self.assertEqual(bank.source_sha256, hashlib.sha256(self.path.read_bytes()).hexdigest())
        a, b = bank.new_executor(), bank.new_executor()
        a.observe(0, a.initial_q, np.zeros(7)); a.observe(.1, a.initial_q, np.zeros(7))
        a.request('reposition_center', .1)
        self.assertIsNone(b.active)
        self.assertEqual(b.events, [])
        with self.assertRaises(Exception):
            bank.source_bytes = b'changed'

    def test_current_guard_routing_not_same_ready_reset(self):
        c = MultiGuardController(self.path)
        self.assertEqual(c.mask(0).tolist(), [True]+[False]*9)
        self.settle(c)
        self.assertEqual(c.executor.node, 'center')
        with self.assertRaises(ValueError):
            c.request_id(1, .1)
        c.request_control_transition('reposition_center', .1)
        self.assertEqual(c.executor.active, 'reposition_center')
        self.assertFalse(c.actor_decision_allowed(.1))
        self.finish(c, .1)
        self.assertEqual(c.executor.node, 'left_high')
        self.assertTrue(c.mask(1.2)[1])
        c.request_id(1, 1.2)
        self.assertEqual(c.executor.active, ACTION_NAMES[1])
        self.finish(c, 1.2)
        self.assertEqual(c.executor.node, 'right_high')
        self.assertTrue(c.mask(2.3)[1])
        c.request_id(1, 2.3)
        self.assertEqual(c.executor.active, 'cut_down_repeat')

    def test_busy_wait_keeps_active_curve_and_is_not_policy_decision(self):
        c = MultiGuardController(self.path)
        self.settle(c)
        c.request_control_transition('reposition_center', .1)
        expected = c.target(.3)
        c.request_id(0, .3)
        np.testing.assert_array_equal(c.target(.3), expected)
        self.assertEqual(c.executor.active, 'reposition_center')
        self.assertEqual(c.mask(.3).tolist(), [True]+[False]*9)
        self.assertFalse(c.actor_decision_allowed(.3))
        with self.assertRaises(ValueError):
            c.request_control_transition('reposition_center', .3)
        with self.assertRaises(ValueError):
            c.request_id(5, .3)

    def test_reference_end_is_not_actual_settling(self):
        c = MultiGuardController(self.path)
        self.settle(c)
        c.request_control_transition('reposition_center', .1)
        q = c.executor.nodes['left_high']
        c.observe(1.1, q, np.ones(7))
        self.assertIsNotNone(c.executor.active)
        self.assertFalse(c.actor_decision_allowed(1.1))
        c.observe(1.2, q, np.zeros(7))
        self.assertIsNotNone(c.executor.active)
        c.observe(1.3, q, np.zeros(7))
        self.assertIsNone(c.executor.active)
        self.assertEqual(c.executor.node, 'left_high')

    def test_stale_observation_does_not_authorize_new_choice(self):
        c = MultiGuardController(self.path, max_observation_age_s=.02)
        self.settle(c)
        self.assertTrue(c.actor_decision_allowed(.11))
        self.assertFalse(c.actor_decision_allowed(.13))
        self.assertEqual(c.mask(.13).tolist(), [True]+[False]*9)
        with self.assertRaises(ValueError):
            c.request_control_transition('reposition_center', .13)
        c.request_id(0, .13)
        with self.assertRaises(ValueError):
            c.observe(.12, c.executor.initial_q, np.zeros(7))
        for bad in (True, -1, float('nan'), float('inf')):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                c.mask(bad)

    def test_partial_registry_preserves_missing_and_disconnected_without_bridges(self):
        self.cfg['actions'] = {ACTION_NAMES[1]: self.cfg['actions'][ACTION_NAMES[1]]}
        self.cfg['families'][ACTION_NAMES[1]]['routes'] = {'left_high': ACTION_NAMES[1]}
        for name in ACTION_NAMES[2:]:
            self.cfg['families'][name]['routes'] = {}
        self.cfg['control_transitions'] = []
        self.save()
        bank = load_multiguard_bank(self.path)
        self.assertEqual(bank.missing_families, ACTION_NAMES[2:])
        self.assertEqual(bank.unreachable_families, (ACTION_NAMES[1],))
        self.assertEqual(bank.reachable_nodes, frozenset({'center'}))
        self.assertEqual(bank.policy_dead_end_guards, frozenset({'center'}))
        c = MultiGuardController(self.path)
        self.settle(c)
        self.assertEqual(c.mask(.1).tolist(), [True]+[False]*9)
        self.assertEqual(c.available_control_transitions(.1), ())
        with self.assertRaises(ValueError):
            load_multiguard_bank(self.path, require_complete=True)

    def test_manual_reposition_cannot_pass_policy_reachability(self):
        # Control transitions still connect the full graph, but the policy has
        # no way out of center if its only local family becomes a self loop.
        name = ACTION_NAMES[5]
        self.cfg['actions'][name].update(end_node='center', waypoint_nodes=['center', 'center'])
        self.save()
        bank = load_multiguard_bank(self.path)
        self.assertEqual(bank.unreachable_families, ())
        self.assertEqual(bank.policy_reachable_nodes, frozenset({'center'}))
        self.assertEqual(set(bank.policy_unreachable_families), set(ACTION_NAMES[1:])-{name})
        with self.assertRaises(ValueError):
            load_multiguard_bank(self.path, require_complete=True)

    def test_closed_cycle_can_lose_families_despite_initial_reachability(self):
        endpoints = [('left_high', 'right_low'), ('right_high', 'left_low'),
                     ('left_low', 'right_high'), ('right_low', 'left_high'),
                     ('center', 'left_high'), ('center', 'right_high'),
                     ('center', 'center'), ('center', 'center'), ('center', 'center')]
        del self.cfg['actions']['cut_down_repeat']
        for name, (start, end) in zip(ACTION_NAMES[1:], endpoints):
            self.cfg['actions'][name].update(start_node=start, end_node=end,
                                            waypoint_nodes=[start, end])
            self.cfg['families'][name]['routes'] = {start: name}
        self.save()
        bank = load_multiguard_bank(self.path)
        self.assertEqual(bank.missing_families, ())
        self.assertEqual(bank.policy_unreachable_families, ())
        self.assertEqual(bank.policy_dead_end_guards, frozenset())
        limited = dict(bank.policy_limited_guards)
        self.assertEqual(limited['center'], ())
        self.assertIn('thrust_center', limited['left_high'])
        self.assertIn('parry_high', limited['right_low'])
        self.assertNotIn('cut_down_left_to_right', limited['left_high'])
        self.assertEqual(bank_summary(bank)['policy_limited_guards']['left_high'],
                         list(limited['left_high']))
        controller = MultiGuardController(self.path)
        self.assertEqual(controller.own_status(0.)['policy_limited_guards'], limited)
        with self.assertRaises(ValueError):
            load_multiguard_bank(self.path, require_complete=True)

    def test_no_unknown_id_wrong_owner_or_wrong_entry(self):
        for change in (
            lambda c: c['families'][ACTION_NAMES[1]].update(id=True),
            lambda c: c['families'][ACTION_NAMES[1]].update(kind='defence'),
            lambda c: c['families'][ACTION_NAMES[1]]['routes'].update(center=ACTION_NAMES[1]),
            lambda c: c['control_transitions'].append(ACTION_NAMES[1]),
            lambda c: c.update(admitted_to_training=True),
            lambda c: c['nodes']['left_high'].__setitem__(1, -.1),
            lambda c: c['actions'][ACTION_NAMES[1]]['knot_time_s'].__setitem__(0, True),
            lambda c: c['action_ids'].update({'2': 'other'}),
        ):
            self.cfg = fixture(); change(self.cfg); self.save()
            with self.subTest(change=change), self.assertRaises(ValueError):
                load_multiguard_bank(self.path)
        self.cfg = fixture(); self.save()
        c = MultiGuardController(self.path)
        for bad in (True, -1, 10, 1., '1'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                c.request_id(bad, 0)

    def test_changed_bank_or_source_cannot_be_executed(self):
        source = Path(self.temp.name)/'evidence.txt'; source.write_text('original')
        self.cfg['source_sha256'] = {str(source): hashlib.sha256(source.read_bytes()).hexdigest()}
        self.save()
        c = MultiGuardController(self.path)
        self.settle(c)
        source.write_text('changed')
        with self.assertRaises(ValueError): c.request_id(0, .1)
        source.write_text('original')
        self.cfg['notes'] = 'changed'; self.save()
        with self.assertRaises(ValueError): c.observe(.2, c.executor.initial_q, np.zeros(7))

    def test_own_status_exposes_no_opponent_or_future_labels(self):
        c = MultiGuardController(self.path)
        self.settle(c)
        status = c.own_status(.1)
        self.assertFalse(status['training_admitted'])
        self.assertTrue(status['own_observation_fresh'])
        self.assertFalse(any('opponent' in name or 'future' in name for name in status))
        status['available_mask'][0] = False
        self.assertTrue(c.mask(.1)[0])

    def geometry_report(self):
        cfg = fixture()
        return {'guards': {n: {'qpos_rad': cfg['nodes'][n]} for n in GUARD_NAMES},
                'actions': {'cut_down_lr': {'start_guard': 'left_high', 'end_guard': 'right_low',
                    'kind': 'attack', 'candidate_pass': True, 'knot_time_s': [0., 1.],
                    'knot_qpos': [cfg['nodes']['left_high'][:], cfg['nodes']['right_low'][:]]},
                    'parry_left': {'kind': 'defence', 'candidate_pass': False}}}

    def test_geometry_import_keeps_native_gates_pending_and_does_not_fill_missing(self):
        path = Path(self.temp.name)/'report.json'; path.write_text(json.dumps(self.geometry_report()))
        payload = geometry_report_to_payload(path)
        self.assertEqual(len(payload['actions']), 1)
        self.assertFalse(payload['admitted_to_training'])
        self.assertEqual(payload['control_transitions'], [])
        self.assertEqual(payload['families'][ACTION_NAMES[1]]['routes'], {'left_high': 'cut_down_lr'})
        self.assertEqual(payload['families']['parry_left']['routes'], {})
        self.assertEqual(payload['actions']['cut_down_lr']['waypoint_nodes'], ['left_high', 'right_low'])
        self.path.write_text(json.dumps(payload))
        bank = load_multiguard_bank(self.path)
        self.assertEqual(len(bank.missing_families), 8)
        self.assertEqual(bank.unreachable_families, (ACTION_NAMES[1],))
        path.write_text('{}')
        with self.assertRaises(ValueError): bank.assert_identity()

    def test_optional_context_routes_are_reported_but_not_implicitly_imported(self):
        report = self.geometry_report()
        report['context_routes'] = {'extra_parry': {'family': 'parry_left', 'candidate_pass': True}}
        path = Path(self.temp.name)/'context_report.json'
        path.write_text(json.dumps(report))
        payload = geometry_report_to_payload(path)
        self.assertEqual(payload['context_routes_not_imported'], ['extra_parry'])
        self.assertNotIn('extra_parry', payload['actions'])
        self.assertEqual(payload['families']['parry_left']['routes'], {})

        # Explicit selection through actions is not an automatic context import.
        q = report['guards']
        report['actions']['extra_parry'] = {
            'family': 'parry_left', 'kind': 'defence', 'candidate_pass': True,
            'start_guard': 'center', 'end_guard': 'left_high',
            'knot_time_s': [0., 1.],
            'knot_qpos': [q['center']['qpos_rad'], q['left_high']['qpos_rad']]}
        path.write_text(json.dumps(report))
        payload = geometry_report_to_payload(path)
        self.assertEqual(payload['context_routes_not_imported'], [])
        self.assertEqual(payload['families']['parry_left']['routes'], {'center': 'extra_parry'})

    def test_geometry_import_rejects_endpoint_repair_and_ambiguous_routes(self):
        for mutate in (
            lambda r: r['actions']['cut_down_lr']['knot_qpos'][0].__setitem__(0, 3.1),
            lambda r: r['actions']['cut_down_lr'].update(kind='defence'),
            lambda r: r['actions'].update(second={**copy.deepcopy(r['actions']['cut_down_lr']),
                                                   'family': 'cut_down_lr'}),
        ):
            report = self.geometry_report(); mutate(report)
            path = Path(self.temp.name)/'report.json'; path.write_text(json.dumps(report))
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                geometry_report_to_payload(path)

    def test_export_is_candidate_only_and_preserves_existing_artifact(self):
        source = Path(self.temp.name)/'report.json'
        source.write_text(json.dumps(self.geometry_report()))
        destination = Path(self.temp.name)/'export'/'bank.json'
        bank = export_geometry_bank(source, destination)
        summary = bank_summary(bank)
        self.assertFalse(summary['training_admitted'])
        self.assertEqual(summary['family_route_counts']['cut_down_left_to_right'], 1)
        self.assertIn('parry_left', summary['missing_families'])
        before = destination.read_bytes()
        with self.assertRaises(FileExistsError):
            export_geometry_bank(source, destination)
        self.assertEqual(destination.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
