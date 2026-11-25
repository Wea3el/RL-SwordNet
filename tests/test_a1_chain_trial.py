"""Pure chain-contract tests; synthetic curves prove no fencing ability."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from sim.a1_chain_trial import (AvailableSequence, chain_verdict, event_chain_checks,
                                declared_opponent_qpos, hashes_unchanged, make_executor, plan_chain)


def fixture():
    q = [3., .7, 3., 1., 2., 1., 0.]
    r = q.copy(); r[0] += .1
    return {'guards': {'left': {'qpos_rad': q}, 'right': {'qpos_rad': r}},
            'actions': {
                'go': {'candidate_pass': True, 'start_guard': 'left', 'end_guard': 'right',
                       'knot_time_s': [0., 1.], 'knot_qpos': [q.copy(), r.copy()]},
                'back': {'candidate_pass': True, 'start_guard': 'right', 'end_guard': 'left',
                         'knot_time_s': [0., 1.], 'knot_qpos': [r.copy(), q.copy()]}}}


class ChainTests(unittest.TestCase):
    def test_public_opponent_condition_does_not_replace_center(self):
        report = {'guards': {'center': {'qpos_rad': [0.]*7},
                             'side': {'qpos_rad': [1.]*7}}}
        q = declared_opponent_qpos(report, 'side')
        np.testing.assert_array_equal(q, np.ones(7))
        q[0] = 99.
        self.assertEqual(report['guards']['side']['qpos_rad'][0], 1.)
        np.testing.assert_array_equal(declared_opponent_qpos(report), np.zeros(7))
        with self.assertRaises(ValueError): declared_opponent_qpos(report, 'missing')

    def test_hashes_accept_serializable_string_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'source.py'; p.write_text('original')
            hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()}
            self.assertTrue(hashes_unchanged(hashes))
            p.write_text('changed')
            self.assertFalse(hashes_unchanged(hashes))

    def test_incomplete_horizon_is_not_physics_failure(self):
        policy = AvailableSequence(['go'])
        self.assertEqual(chain_verdict({'all_done': False}, {'completion': {'is_truncation': True}}, policy),
                         'incomplete_not_a_physical_failure')
        self.assertEqual(chain_verdict({'all_done': False}, {'failure': None}, policy),
                         'incomplete_pending_sequence_not_a_physical_failure')
        self.assertEqual(chain_verdict({'all_done': False}, {'failure': ['actual_limit_violation']}, policy),
                         'not_passed')

    def test_exact_bidirectional_chain_and_repeat(self):
        p = plan_chain(fixture(), ['go', 'back'], repeats=2)
        self.assertEqual(p['names'], ['go', 'back', 'go', 'back'])
        self.assertEqual(p['initial_guard'], p['final_guard'])
        self.assertGreater(p['duration_s'], 4.4)

    def test_repeat_without_return_route_is_rejected(self):
        with self.assertRaises(ValueError): plan_chain(fixture(), ['go'], repeats=2)

    def test_missing_bridge_is_not_generated(self):
        with self.assertRaises(ValueError): plan_chain(fixture(), ['go', 'go'])

    def test_failed_candidate_is_not_run(self):
        r = fixture(); r['actions']['go']['candidate_pass'] = False
        with self.assertRaises(ValueError): plan_chain(r, ['go'])

    def test_endpoint_alias_is_not_repaired(self):
        r = copy.deepcopy(fixture()); r['actions']['go']['knot_qpos'][0][0] += .01
        with self.assertRaises(ValueError): plan_chain(r, ['go'])

    def test_bad_sequence_and_repeat(self):
        for seq, repeat in [([], 1), ('go', 1), ([True], 1), (['go'], True), (['go'], 0)]:
            with self.subTest(seq=seq, repeat=repeat), self.assertRaises(ValueError):
                plan_chain(fixture(), seq, repeats=repeat)

    def test_horizon_cannot_drop_tail(self):
        with self.assertRaises(ValueError): plan_chain(fixture(), ['go', 'back'], repeats=25)

    def test_policy_waits_for_actual_availability(self):
        r = fixture(); p = plan_chain(r, ['go', 'back']); e = make_executor(r, p)
        policy = AvailableSequence(p['names'])
        obs = {'own_q_rad': e.initial_q, 'own_qvel_rad_s': np.zeros(7)}
        self.assertIsNone(policy.choose(obs, e.available()))
        e.observe(0., e.initial_q, np.zeros(7)); e.observe(.1, e.initial_q, np.zeros(7))
        self.assertEqual(policy.choose(obs, e.available()), 'go'); e.request('go', .1)
        self.assertIsNone(policy.choose(obs, e.available()))
        qr = np.asarray(r['guards']['right']['qpos_rad'])
        e.observe(1.1, qr, np.ones(7))
        self.assertIsNone(policy.choose(obs, e.available()))
        e.observe(1.2, qr, np.zeros(7)); e.observe(1.3, qr, np.zeros(7))
        obs['own_q_rad'] = qr
        self.assertEqual(policy.choose(obs, e.available()), 'back')

    def test_event_checks_require_all_completions(self):
        p = plan_chain(fixture(), ['go', 'back']); policy = AvailableSequence(p['names'])
        events = []
        for i, (name, row) in enumerate(zip(p['names'], p['actions'])):
            obs = {'own_q_rad': p['curves'][name].knot_qpos[0], 'own_qvel_rad_s': np.zeros(7)}
            policy.choose(obs, [name])
            events += [{'event': 'started', 'action': name, 'node': row['start_guard'], 'time_s': 2.*i},
                       {'event': 'completed', 'action': name, 'node': row['end_guard'], 'time_s': 2.*i+1.1}]
        self.assertTrue(all(event_chain_checks(p, events, policy).values()))
        self.assertFalse(all(event_chain_checks(p, events[:-1], policy).values()))

    def test_bad_observed_state_cannot_count_as_trigger(self):
        p = plan_chain(fixture(), ['go']); policy = AvailableSequence(p['names'])
        policy.choose({'own_q_rad': np.zeros(7), 'own_qvel_rad_s': np.zeros(7)}, ['go'])
        events = [{'event': 'started', 'action': 'go', 'node': 'left', 'time_s': 0.},
                  {'event': 'completed', 'action': 'go', 'node': 'right', 'time_s': 1.2}]
        self.assertFalse(event_chain_checks(p, events, policy)['requests_use_observed_entry_state'])


if __name__ == '__main__':
    unittest.main()
