"""Fast command-contract fixtures; no simulator integration or fencing claims."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from sim.a1_variant_bank import load_variant_bank, NamedEdgeController


def fixture():
    q = [3., .7, 3., 1., 2., 1., 0.]
    other = q.copy(); other[0] += .1
    guards = {'right_low_negative_robust': {'qpos_rad': q}, 'another_real_guard': {'qpos_rad': other}}
    actions = {}
    for name, amount in (('parry_right_thrust', .1), ('parry_right_descending', .2)):
        middle = q.copy(); middle[0] += amount
        actions[name] = {'kind': 'defence', 'family': 'parry_right', 'candidate_pass': True,
            'start_guard': 'right_low_negative_robust', 'end_guard': 'right_low_negative_robust',
            'knot_time_s': [0., 1., 2.], 'knot_qpos': [q.copy(), middle, q.copy()]}
    actions['transfer'] = {'kind': 'transition', 'family': 'explicit_guard_transfer', 'candidate_pass': True,
        'start_guard': 'right_low_negative_robust', 'end_guard': 'another_real_guard',
        'knot_time_s': [0., 1.], 'knot_qpos': [q.copy(), other.copy()]}
    actions['return'] = {'kind': 'transition', 'family': 'explicit_guard_transfer', 'candidate_pass': True,
        'start_guard': 'another_real_guard', 'end_guard': 'right_low_negative_robust',
        'knot_time_s': [0., 1.], 'knot_qpos': [other.copy(), q.copy()]}
    return {'schema_version': 1, 'guards': guards, 'actions': actions, 'admitted_to_training': False}


ORDER = ('parry_right_thrust', 'parry_right_descending', 'transfer')


class NamedEdgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)/'report.json'
        self.report = fixture()
        self.save()

    def tearDown(self):
        self.temp.cleanup()

    def save(self):
        self.path.write_text(json.dumps(self.report))

    def bank(self, order=ORDER):
        return load_variant_bank(self.path, policy_order=order, initial_guard='right_low_negative_robust')

    def ready(self, controller, time=0.):
        q = controller.executor.nodes[controller.executor.node]
        controller.observe(time, q, np.zeros(7))
        controller.observe(time+.1, q, np.zeros(7))

    def test_same_family_same_entry_are_distinct_choices(self):
        bank = self.bank(); controller = NamedEdgeController(bank)
        self.ready(controller)
        self.assertEqual(bank.action_names, ('wait',)+ORDER)
        self.assertEqual(controller.mask(.1).tolist(), [True]*4)
        controller.request_id(2, .1)
        state = controller.own_status(.1)
        self.assertEqual(state['own_active_action_id'], 2)
        self.assertEqual(state['own_active_action_name'], 'parry_right_descending')
        self.assertEqual(state['own_active_family'], 'parry_right')
        self.assertFalse(any('opponent' in key for key in state))

    def test_busy_wait_never_interrupts_or_forges_completion(self):
        controller = NamedEdgeController(self.bank()); self.ready(controller)
        controller.request_id(1, .1)
        self.assertEqual(controller.mask(.1).tolist(), [True, False, False, False])
        controller.request_id(0, .1)
        self.assertEqual(controller.executor.active, ORDER[0])
        with self.assertRaises(ValueError): controller.request_id(2, .1)
        q = controller.executor.nodes['right_low_negative_robust']
        controller.observe(2.1, q, np.zeros(7))
        self.assertIsNotNone(controller.executor.active)
        controller.observe(2.2, q, np.zeros(7))
        self.assertIsNone(controller.executor.active)
        self.assertTrue(controller.actor_decision_allowed(2.2))

    def test_transfer_must_be_explicit_and_no_automatic_return(self):
        bank = self.bank(); c = NamedEdgeController(bank); self.ready(c)
        c.request_id(3, .1)
        q = c.executor.nodes['another_real_guard']
        c.observe(1.1, q, np.zeros(7)); c.observe(1.2, q, np.zeros(7))
        self.assertEqual(c.mask(1.2).tolist(), [True, False, False, False])
        self.assertFalse(c.actor_decision_allowed(1.2))
        self.assertNotIn('return', c.executor.actions)
        self.assertEqual(bank.summary()['wait_only_reachable_guards'], ['another_real_guard'])
        self.assertIn('return', bank.summary()['unselected_report_edges_not_callable'])

    def test_unsettled_wrong_pose_and_stale_observation_disable_choices(self):
        c = NamedEdgeController(self.bank(), max_observation_age_s=.01)
        self.assertEqual(c.mask(0.).tolist(), [True, False, False, False])
        self.ready(c)
        self.assertTrue(c.actor_decision_allowed(.1))
        self.assertEqual(c.mask(.12).tolist(), [True, False, False, False])
        with self.assertRaises(ValueError): c.request_id(1, .12)
        with self.assertRaises(ValueError): c.observe(.11, c.executor.initial_q, np.zeros(7))
        c.observe(.12, c.executor.initial_q+.2, np.zeros(7))
        self.assertFalse(c.actor_decision_allowed(.12))

    def test_independent_executors(self):
        bank = self.bank(); a = NamedEdgeController(bank); b = NamedEdgeController(bank)
        self.ready(a); self.ready(b); a.request_id(1, .1)
        self.assertIsNone(b.executor.active)
        self.assertEqual(b.executor.events, [])

    def test_reordered_ids_change_signature_and_legacy_actor_rejected(self):
        first = self.bank(); other = self.bank(tuple(reversed(ORDER)))
        self.assertNotEqual(first.actor_signature(), other.actor_signature())
        first.assert_actor_compatible(first.actor_signature(), output_size=4)
        for signature, size in (({}, 5), ({}, 10), (other.actor_signature(), 4),
                                (first.actor_signature(), 5)):
            with self.subTest(size=size), self.assertRaises(ValueError):
                first.assert_actor_compatible(signature, output_size=size)

    def test_source_mutation_rejected_before_use(self):
        bank = self.bank(); c = NamedEdgeController(bank)
        self.report['note'] = 'changed'; self.save()
        with self.assertRaises(ValueError): c.observe(0., c.executor.initial_q, np.zeros(7))
        with self.assertRaises(ValueError): bank.assert_actor_compatible(bank.actor_signature(), output_size=4)

    def test_invalid_selected_reference_cannot_enter_adapter(self):
        for modification in ('bounds', 'endpoint', 'clock', 'candidate', 'kind', 'boolean'):
            with self.subTest(modification=modification):
                self.report = fixture(); row = self.report['actions'][ORDER[0]]
                if modification == 'bounds': row['knot_qpos'][1][1] = -.1
                elif modification == 'endpoint': row['knot_qpos'][0][0] += .01
                elif modification == 'clock': row['knot_time_s'][0] = .1
                elif modification == 'candidate': row['candidate_pass'] = False
                elif modification == 'kind': row['kind'] = 'fake'
                elif modification == 'boolean': row['knot_qpos'][0][0] = True
                self.save()
                with self.assertRaises(ValueError): self.bank()

    def test_explicit_order_and_initial_node_required(self):
        for order in (list(ORDER), (), (ORDER[0], ORDER[0]), ('wait',), ('missing',)):
            with self.subTest(order=order), self.assertRaises(ValueError): self.bank(order)
        with self.assertRaises(ValueError):
            load_variant_bank(self.path, policy_order=ORDER, initial_guard='automatic_center')

    def test_non_admission_must_be_explicit(self):
        self.report.pop('admitted_to_training');self.save()
        with self.assertRaises(ValueError):self.bank()

    def test_native_research_hypothesis_is_not_a_policy_choice(self):
        row = self.report['actions'][ORDER[0]]
        row['candidate_stage'] = 'authorized_native_research_hypothesis'
        row['candidate_pass'] = True  # Permission to probe physics, not to select in a library.
        self.save()
        with self.assertRaises(ValueError): self.bank()

    def test_summary_never_promotes_candidate_or_physics_identity(self):
        result = self.bank().summary()
        self.assertFalse(result['admitted_to_training'])
        self.assertFalse(result['actual_physics_identity_bound'])
        self.assertFalse(result['automatic_bridge_or_opponent_label_selection'])
        self.assertEqual([a['family'] for a in result['actions'][:2]], ['parry_right']*2)


if __name__ == '__main__':
    unittest.main()
