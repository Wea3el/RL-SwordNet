import unittest

from sim.a1_chain_effectiveness import assign_entries


class ChainEffectivenessTests(unittest.TestCase):
    def fixture(self):
        actions = {'move': {'kind': 'transition'},
                   'cut': {'kind': 'attack', 'target_region': 'left', 'intended_direction': [0, -1, -1]}}
        events = [{'event': 'started', 'action': 'move', 'time_s': .1},
                  {'event': 'completed', 'action': 'move', 'time_s': 1.},
                  {'event': 'started', 'action': 'cut', 'time_s': 1.1},
                  {'event': 'completed', 'action': 'cut', 'time_s': 2.}]
        return actions, events

    def test_hits_belong_to_their_actual_execution_interval(self):
        a, e = self.fixture()
        rows, extra = assign_entries(a, e, [{'agent': 'a1', 'time_s': 1.5, 'region': 'left'}], 'a1')
        self.assertTrue(rows[1]['intended_region_first']); self.assertEqual(extra, [])
        self.assertEqual(rows[0]['entries'], [])

    def test_movement_score_is_not_an_attack_success(self):
        a, e = self.fixture()
        rows, _ = assign_entries(a, e, [{'agent': 'a1', 'time_s': .5, 'region': 'left'}], 'a1')
        self.assertEqual(len(rows[0]['entries']), 1); self.assertFalse(rows[1]['intended_region_first'])

    def test_late_or_other_agent_hit_cannot_fill_a_missing_attack(self):
        a, e = self.fixture()
        rows, extra = assign_entries(a, e, [{'agent': 'a1', 'time_s': 2.5, 'region': 'left'},
                                           {'agent': 'a2', 'time_s': 1.5, 'region': 'left'}], 'a1')
        self.assertFalse(rows[1]['intended_region_first']); self.assertEqual(len(extra), 1)

    def test_wrong_region_first_is_not_corrected_by_a_later_hit(self):
        a, e = self.fixture()
        rows, _ = assign_entries(a, e, [{'agent': 'a1', 'time_s': 1.3, 'region': 'right'},
                                       {'agent': 'a1', 'time_s': 1.5, 'region': 'left'}], 'a1')
        self.assertFalse(rows[1]['intended_region_first'])

    def test_missing_completion_rejected(self):
        a, e = self.fixture()
        with self.assertRaises(ValueError): assign_entries(a, e[:-1], [], 'a1')

    def test_shared_boundary_entry_is_not_counted_twice(self):
        a, e = self.fixture(); e[2]['time_s'] = 1.
        rows, _ = assign_entries(a, e, [{'agent': 'a1', 'time_s': 1., 'region': 'left'}], 'a1')
        self.assertEqual(len(rows[0]['entries']), 1)
        self.assertEqual(rows[1]['entries'], [])


if __name__ == '__main__':
    unittest.main()
