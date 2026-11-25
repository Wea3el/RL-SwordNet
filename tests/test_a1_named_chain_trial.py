"""Synthetic command/catalog evidence only; never executes native physics."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from sim.a1_named_chain_trial import (CatalogSequence, plan_named_chain,
    recorded_catalog_checks, recorded_choice_checks, run)
from sim.a1_variant_bank import load_variant_bank
from sim.assembly import _finite_json


def fixture():
    q = [3., .7, 3., 1., 2., 1., 0.]
    r = q.copy(); r[0] += .1
    actions = {}
    for name, start, end in (('go', 'right', 'left'), ('back', 'left', 'right'),
                              ('parry_right_thrust', 'right', 'right'),
                              ('parry_right_descending', 'right', 'right')):
        nodes = {'right': q, 'left': r}
        actions[name] = {'candidate_pass': True, 'kind': 'defence' if name.startswith('parry') else 'transition',
            'family': 'parry_right' if name.startswith('parry') else 'explicit_guard_transfer',
            'start_guard': start, 'end_guard': end, 'knot_time_s': [0., 1.],
            'knot_qpos': [nodes[start][:], nodes[end][:]]}
    return {'admitted_to_training': False, 'guards': {'right': {'qpos_rad': q}, 'left': {'qpos_rad': r}},
            'actions': actions, 'scoring': {'fixture_only': True}}


class NamedChainTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.path = self.folder/'report.json'
        self.path.write_text(json.dumps(fixture()))

    def tearDown(self):
        self.temp.cleanup()

    def bank(self, order=('go', 'back', 'parry_right_thrust', 'parry_right_descending'), initial='right'):
        return load_variant_bank(self.path, policy_order=order, initial_guard=initial)

    def test_unselected_transfer_cannot_be_called_from_source_report(self):
        bank = self.bank(('go', 'parry_right_thrust'))
        with self.assertRaises(ValueError): plan_named_chain(bank, ['go', 'back'])
        self.assertNotIn('back', bank.new_executor().actions)

    def test_exact_initial_and_continuous_nodes_required(self):
        bank = self.bank()
        with self.assertRaises(ValueError): plan_named_chain(bank, ['back'])
        with self.assertRaises(ValueError): plan_named_chain(bank, ['go', 'go'])
        with self.assertRaises(ValueError): plan_named_chain(bank, ['go'], repeats=2)
        plan = plan_named_chain(bank, ['go', 'back'], repeats=2)
        self.assertEqual(plan['names'], ['go', 'back', 'go', 'back'])
        self.assertEqual(plan['action_ids'], [1, 2, 1, 2])
        self.assertEqual(plan['initial_guard'], plan['final_guard'])

    def test_catalog_executor_contains_all_selected_not_only_scripted_edges(self):
        bank = self.bank(); plan = plan_named_chain(bank, ['parry_right_thrust'])
        self.assertEqual(plan['names'], ['parry_right_thrust'])
        self.assertEqual(set(bank.new_executor().actions), set(bank.policy_order))

    def test_same_family_choices_have_distinct_ids(self):
        bank = self.bank(); plan = plan_named_chain(bank, ['parry_right_thrust', 'parry_right_descending'])
        policy = CatalogSequence(bank, plan)
        executor = bank.new_executor()
        q = executor.initial_q
        obs = {'own_q_rad': q, 'own_qvel_rad_s': np.zeros(7)}
        self.assertIsNone(policy.choose(obs, executor.available()))
        executor.observe(0., q, np.zeros(7)); executor.observe(.1, q, np.zeros(7))
        self.assertEqual(policy.choose(obs, executor.available()), 'parry_right_thrust')
        executor.request('parry_right_thrust', .1)
        self.assertIsNone(policy.choose(obs, executor.available()))
        executor.observe(1.1, q, np.zeros(7)); executor.observe(1.2, q, np.zeros(7))
        self.assertEqual(policy.choose(obs, executor.available()), 'parry_right_descending')
        self.assertEqual([e['action_id'] for e in policy.events], [3, 4])

    def test_policy_initial_state_is_native_snapshot_json_serializable(self):
        bank = self.bank()
        policy = CatalogSequence(bank, plan_named_chain(bank, ['parry_right_thrust']))
        saved = json.loads(json.dumps(_finite_json(copy.deepcopy(policy.__dict__))))
        self.assertEqual(saved['catalog_signature'], bank.actor_signature())
        self.assertEqual(saved['action_names'], list(bank.action_names))
        self.assertEqual(saved['entry_qpos']['parry_right_thrust'], fixture()['guards']['right']['qpos_rad'])
        self.assertNotIn('bank', saved)

    def test_run_reaches_native_start_with_serializable_policy_state(self):
        calls = []
        def stop_after_snapshot(executors, policies, output, **kwargs):
            snapshots = {slot: json.loads(json.dumps(_finite_json(copy.deepcopy(policy.__dict__))))
                         for slot, policy in policies.items()}
            calls.append(snapshots)
            self.assertEqual(set(executors['a1'].actions), set(self.bank().policy_order))
            self.assertEqual(snapshots['a1']['catalog_signature'], kwargs['metadata']['catalog_signature'])
            raise RuntimeError('mock-native-stop-after-policy-snapshot')
        with patch('sim.a1_named_chain_trial.run_feedback_trial', side_effect=stop_after_snapshot):
            with self.assertRaisesRegex(RuntimeError, 'mock-native-stop-after-policy-snapshot'):
                run(self.path, self.folder/'mock_start', policy_order=self.bank().policy_order,
                    initial_guard='right', sequence=['parry_right_thrust'], actor='a1', opponent_guard='left')
        self.assertEqual(len(calls), 1)
        self.assertTrue((self.folder/'mock_start/catalog_snapshot.json').is_file())

    def test_exposed_unselected_edge_and_bad_measured_pose_are_rejected(self):
        bank = self.bank(('go',)); policy = CatalogSequence(bank, plan_named_chain(bank, ['go']))
        obs = {'own_q_rad': np.zeros(7), 'own_qvel_rad_s': np.zeros(7)}
        with self.assertRaises(ValueError): policy.choose(obs, ('back',))
        with self.assertRaises(ValueError): policy.choose(obs, ('go',))

    def write_native_catalog(self, bank):
        folder = self.folder/'native'; folder.mkdir()
        executor = bank.new_executor()
        graph = {'initial_node': bank.initial_guard,
            'nodes': {n: q.tolist() for n, q in executor.nodes.items()},
            'actions': {n: {'start_node': a.start_node, 'end_node': a.end_node}
                        for n, a in executor.actions.items()}}
        summary = {'graph': {'a1': graph}, 'metadata': {'catalog_signature': bank.actor_signature()}}
        (folder/'summary.json').write_text(json.dumps(summary))
        arrays = {f'a1__{n}__{key}': getattr(a.curve, key) for n, a in executor.actions.items()
                  for key in ('knot_time_s', 'knot_qpos', 'coefficients')}
        np.savez_compressed(folder/'graph_references.npz', **arrays)
        return folder, summary, arrays

    def test_recorded_catalog_and_exact_references_bound(self):
        bank = self.bank(); folder, summary, arrays = self.write_native_catalog(bank)
        self.assertTrue(all(recorded_catalog_checks(folder, bank, 'a1').values()))
        arrays['a1__go__knot_qpos'] = arrays['a1__go__knot_qpos'].copy()
        arrays['a1__go__knot_qpos'][0, 0] += .01
        np.savez_compressed(folder/'graph_references.npz', **arrays)
        self.assertFalse(recorded_catalog_checks(folder, bank, 'a1')['actual_frozen_references_match_catalog'])

    def test_added_native_edge_and_wrong_initial_node_fail(self):
        bank = self.bank(); folder, summary, _ = self.write_native_catalog(bank)
        summary['graph']['a1']['actions']['hidden_bridge'] = {'start_node': 'left', 'end_node': 'right'}
        summary['graph']['a1']['initial_node'] = 'left'
        (folder/'summary.json').write_text(json.dumps(summary))
        checks = recorded_catalog_checks(folder, bank, 'a1')
        self.assertFalse(checks['actor_executor_has_exact_catalog_edges'])
        self.assertFalse(checks['actor_executor_has_catalog_initial_node'])

    def test_native_request_time_name_and_id_are_bound(self):
        bank = self.bank(); policy = CatalogSequence(bank, plan_named_chain(bank, ['parry_right_thrust']))
        q = bank.new_executor().initial_q
        policy.choose({'own_q_rad': q, 'own_qvel_rad_s': np.zeros(7)}, ('parry_right_thrust',))
        folder = self.folder/'requests'; folder.mkdir()
        rows = [{'time_s': .1, 'requests': {'a1': 'parry_right_thrust'},
                 'physical': {'a1': {'own_q_rad': q.tolist(), 'own_qvel_rad_s': [0.]*7}},
                 'available': {'a1': ['parry_right_thrust']}}]
        (folder/'observations.json').write_text(json.dumps(rows))
        events = [{'event': 'started', 'time_s': .1, 'action': 'parry_right_thrust'}]
        checks, log = recorded_choice_checks(folder, bank, 'a1', policy, events)
        self.assertTrue(all(checks.values())); self.assertEqual(log[0]['action_id'], 3)
        events[0]['time_s'] = .2
        self.assertFalse(recorded_choice_checks(folder, bank, 'a1', policy, events)[0][
            'native_requests_match_real_executor_start_times'])
        rows[0]['physical']['a1']['own_q_rad'][0] += .1
        (folder/'observations.json').write_text(json.dumps(rows))
        self.assertFalse(recorded_choice_checks(folder, bank, 'a1', policy, events)[0][
            'choices_bind_current_public_state_and_availability'])

    def test_wait_unknown_and_excessive_windows_rejected(self):
        bank = self.bank()
        for sequence in ([], ['wait'], ['missing']):
            with self.subTest(sequence=sequence), self.assertRaises(ValueError):
                plan_named_chain(bank, sequence)
        with self.assertRaises(ValueError): plan_named_chain(bank, ['go', 'back'], repeats=40)


if __name__ == '__main__':
    unittest.main()
