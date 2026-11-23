import copy
import unittest
from sim.constrained_validation import defence_evidence_gate


class ConstrainedDefenceTests(unittest.TestCase):
    def setUp(self):
        self.base = {'native_execution_pass':True, 'source_unchanged':True,
            'all_target_entries':[{'agent':'a1','time_s':2.}],
            'runtime_config':{'minimum_blade_force_n':.1,'minimum_blade_impulse_ns':.02}}
        self.defended = dict(self.base, all_target_entries=[])
        self.contacts = [{'interval_start_s':1.,'interval_end_s':1.1,
            'contacts':[{'category':'cross_blade_blade','normal_force_n':1.}]}]

    def gate(self, **kwargs):
        return defence_evidence_gate(kwargs.get('baseline',self.base),kwargs.get('defended',self.defended),
            kwargs.get('contacts',self.contacts),attacker='a1',matched=kwargs.get('matched',True))

    def test_only_matched_meaningful_pre_hit_contact_full_window_can_pass(self):
        self.assertTrue(self.gate()['successful_physical_defence'])
        self.assertFalse(self.gate(matched=False)['successful_physical_defence'])
        self.assertFalse(self.gate(contacts=[])['successful_physical_defence'])

    def test_late_hit_is_failure_even_after_an_earlier_counter_hit(self):
        d = dict(self.defended, all_target_entries=[{'agent':'a2','time_s':1.}, {'agent':'a1','time_s':4.}])
        self.assertFalse(self.gate(defended=d)['successful_physical_defence'])

    def test_contact_after_would_be_hit_cannot_explain_a_block(self):
        c = copy.deepcopy(self.contacts); c[0].update(interval_start_s=2.1,interval_end_s=2.2)
        self.assertFalse(self.gate(contacts=c)['successful_physical_defence'])

    def test_failed_native_run_or_missing_baseline_cannot_pass(self):
        self.assertFalse(self.gate(defended=dict(self.defended,native_execution_pass=False))['successful_physical_defence'])
        self.assertFalse(self.gate(baseline=dict(self.base,all_target_entries=[]))['successful_physical_defence'])
        self.assertFalse(self.gate(defended=dict(self.defended,source_unchanged=False))['successful_physical_defence'])

    def test_partial_interval_is_clipped_at_baseline_hit(self):
        c = copy.deepcopy(self.contacts); c[0].update(interval_start_s=1.99,interval_end_s=2.2)
        result = self.gate(contacts=c)
        self.assertAlmostEqual(result['legal_weapon_impulse_before_baseline_hit_ns'],.01)
        self.assertFalse(result['successful_physical_defence'])


if __name__ == '__main__': unittest.main()
