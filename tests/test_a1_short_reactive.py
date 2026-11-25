import copy
import unittest

from sim.a1_short_reactive import cue_checks
from sim.a1_triangle_reactive import PUBLIC_FIELDS


class CueChecksTests(unittest.TestCase):
    def fixture(self):
        physical = {key:[0., 0., 0.] for key in PUBLIC_FIELDS}
        physical[PUBLIC_FIELDS[0]] = [0.] * 7
        physical[PUBLIC_FIELDS[1]] = [0.] * 7
        rows = [{'time_s':1., 'physical':{'a1':copy.deepcopy(physical), 'a2':copy.deepcopy(physical)},
                 'requests':{'a1':None, 'a2':'parry'}, 'available':{'a1':(), 'a2':('parry',)}}]
        summary = {'executor_events':{'a1':[{'event':'started', 'time_s':.8}],
                   'a2':[{'event':'started', 'time_s':1.}, {'event':'completed', 'time_s':10.}]},
                   'policy_events':{'a2':[{'event':'observed_threat', 'observed_tip_m':[0.,0.,0.],
                                           'observed_tip_velocity_m_s':[0.,0.,0.]}]}}
        return summary, rows

    def test_bound_snapshot_and_completed_action(self):
        summary, rows = self.fixture()
        self.assertTrue(all(cue_checks(summary, rows, 'a1').values()))

    def test_preemptive_or_unavailable_trigger_rejected(self):
        summary, rows = self.fixture(); summary['executor_events']['a1'][0]['time_s'] = 1.2
        self.assertFalse(cue_checks(summary, rows, 'a1')['no_defence_before_actual_attack'])
        rows[0]['available']['a2'] = ()
        self.assertFalse(cue_checks(summary, rows, 'a1')['requests_only_when_available'])

    def test_future_field_or_different_snapshot_rejected(self):
        summary, rows = self.fixture(); rows[0]['physical']['a2']['opponent_action'] = 'attack'
        self.assertFalse(cue_checks(summary, rows, 'a1')['shared_physical_fields_only'])
        summary['policy_events']['a2'][0]['observed_tip_m'] = [1.,2.,3.]
        self.assertFalse(cue_checks(summary, rows, 'a1')['cue_matches_actual_shared_snapshot'])

    def test_stationary_control_requires_no_trigger(self):
        summary, rows = self.fixture()
        self.assertFalse(cue_checks(summary, rows, 'a1', stationary=True)['no_stationary_false_trigger'])
        summary['executor_events']['a2'] = []; summary['policy_events']['a2'] = []
        rows[0]['requests']['a2'] = None
        self.assertTrue(all(cue_checks(summary, rows, 'a1', stationary=True).values()))


if __name__ == '__main__':
    unittest.main()
