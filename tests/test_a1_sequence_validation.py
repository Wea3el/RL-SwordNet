"""Continuous defence cannot borrow the first attack's contact evidence."""
import copy
import unittest

from sim.a1_sequence_validation import sequence_defence_gate
from sim.skill_validation import V2_CONFIG


def fixtures(attacker='a1'):
    defender = 'a2' if attacker == 'a1' else 'a1'
    base = dict(status='completed', requested_duration_s=12., completed_time_s=12.,
        runtime_config=copy.deepcopy(V2_CONFIG), command_sha256={attacker:'same', defender:'guard'},
        initial_qpos=[0.]*14, scoring_radius_m=.06, targets_by_attacker={},
        unchanged_force_caps_nm=[1.]*14, source_hashes={}, geometry_sha256='same',
        controller_implementation_sha256={'a1_control.py':'same_actual_controller_sha'},
        action_events={attacker:[dict(time_s=1.,action_id=1),dict(time_s=6.,action_id=2)],defender:[]},
        controller_diagnostics={attacker:dict(completions=[dict(action_id=1),dict(action_id=2)]),
            defender:dict(completions=[])},
        hits=[dict(agent=attacker,time_s=4.),dict(agent=attacker,time_s=9.)],
        first_positive_blade_contact_time_s=None,blade_contact_normal_impulse_ns=0.)
    defended = copy.deepcopy(base)
    defended.update(hits=[], first_positive_blade_contact_time_s=3.,
        blade_contact_impulse_samples=[dict(time_s=3.,normal_force_n=20.,interval_s=.002),
            dict(time_s=8.,normal_force_n=20.,interval_s=.002)])
    defended['action_events'][defender]=[dict(time_s=2.,action_id=4),dict(time_s=7.,action_id=3)]
    defended['controller_diagnostics'][defender]['completions']=[dict(action_id=4),dict(action_id=3)]
    return base, defended


class SequenceGateTests(unittest.TestCase):
    def gate(self, base, defended, attacker='a1'):
        return sequence_defence_gate(base,defended,attacker,[1,2],[4,3])

    def test_each_attack_has_real_contact_and_whole_window_completion(self):
        base, defended=fixtures()
        before=copy.deepcopy((base,defended))
        result=self.gate(base,defended)
        self.assertTrue(result['successful_continuous_physical_defence'])
        self.assertEqual((base,defended),before)
        self.assertFalse(result['admitted_to_training'])

    def test_same_evidence_in_either_physical_slot(self):
        self.assertTrue(self.gate(*fixtures('a2'),attacker='a2')['successful_continuous_physical_defence'])

    def test_first_attack_contact_cannot_certify_second(self):
        base,defended=fixtures()
        defended['blade_contact_impulse_samples'].pop()
        r=self.gate(base,defended)
        self.assertTrue(r['matched_whole_window_physical_comparison']['successful_physical_defence'])
        self.assertFalse(r['successful_continuous_physical_defence'])

    def test_every_baseline_attack_must_really_hit(self):
        base,defended=fixtures()
        base['hits'].pop()
        self.assertFalse(self.gate(base,defended)['successful_continuous_physical_defence'])

    def test_second_contact_after_baseline_entry_does_not_count(self):
        base,defended=fixtures()
        defended['blade_contact_impulse_samples'][1]['time_s']=9.
        self.assertFalse(self.gate(base,defended)['successful_continuous_physical_defence'])

    def test_full_shadow_entry_overrides_early_contacts(self):
        base,defended=fixtures()
        defended['hits']=[dict(agent='a1',time_s=10.)]
        self.assertFalse(self.gate(base,defended)['successful_continuous_physical_defence'])

    def test_cached_success_cannot_hide_missing_actual_action(self):
        base,defended=fixtures()
        defended['controller_diagnostics']['a2']['completions'].pop()
        defended['a1_execution_gate']={'native_execution_pass':True}
        self.assertFalse(self.gate(base,defended)['successful_continuous_physical_defence'])

    def test_different_fixture_or_attack_start_is_not_a_pair(self):
        for key in ('geometry_sha256','command_sha256','action_events'):
            base,defended=fixtures()
            if key=='geometry_sha256':defended[key]='changed'
            elif key=='command_sha256':defended[key]['a1']='changed'
            else:defended[key]['a1'][1]['time_s']+=.1
            self.assertFalse(self.gate(base,defended)['successful_continuous_physical_defence'])

    def test_fractional_interval_is_clipped_at_each_cutoff(self):
        base,defended=fixtures()
        base['hits'][1]['time_s']=9.001
        defended['blade_contact_impulse_samples'][1]=dict(time_s=9.,interval_s=.002,normal_force_n=11.)
        r=self.gate(base,defended)
        self.assertAlmostEqual(r['attack_windows'][1]['clipped_prehit_blade_impulse_ns'],.011)
        self.assertFalse(r['successful_continuous_physical_defence'])

    def test_missing_interval_evidence_or_truncation_cannot_pass(self):
        for field in ('blade_contact_impulse_samples','completed_time_s','status'):
            base,defended=fixtures()
            if field=='blade_contact_impulse_samples':defended.pop(field)
            elif field=='completed_time_s':defended[field]=10.
            else:defended[field]='invalid'
            self.assertFalse(self.gate(base,defended)['successful_continuous_physical_defence'])

    def test_malformed_force_evidence_is_error_not_normal_defence(self):
        for value in (float('nan'),-1.,True):
            base,defended=fixtures()
            defended['blade_contact_impulse_samples'][0]['normal_force_n']=value
            with self.assertRaises(ValueError):self.gate(base,defended)

    def test_expected_actions_are_explicit_and_nonzero(self):
        for expected in ([],[True,2],[0,2]):
            with self.assertRaises(ValueError):
                sequence_defence_gate(*fixtures(),'a1',expected,[4,3])

    def test_recorded_thresholds_cannot_be_replaced_by_default_or_caller(self):
        base,defended=fixtures()
        for r in (base,defended):r['runtime_config']['minimum_blade_impulse_ns']=.04
        base['hits'][1]['time_s']=9.001
        defended['blade_contact_impulse_samples'][1]=dict(time_s=9.,interval_s=.002,normal_force_n=30.)
        self.assertFalse(self.gate(base,defended)['successful_continuous_physical_defence'])
        with self.assertRaises(ValueError):
            sequence_defence_gate(base,defended,'a1',[1,2],[4,3],
                cfg={'minimum_blade_impulse_ns':.02})

    def test_recorded_thresholds_may_not_be_silently_inferred(self):
        for key in ('minimum_blade_force_n','minimum_blade_impulse_ns','physics_timestep_s'):
            base,defended=fixtures()
            for r in (base,defended):r['runtime_config'].pop(key)
            with self.assertRaises(ValueError):self.gate(base,defended)

    def test_actual_controllers_must_match_and_be_recorded(self):
        for change in ('different','missing'):
            base,defended=fixtures()
            if change=='different':defended['controller_implementation_sha256']={'other.py':'different'}
            else:
                base.pop('controller_implementation_sha256')
                defended.pop('controller_implementation_sha256')
            self.assertFalse(self.gate(base,defended)['successful_continuous_physical_defence'])

    def test_long_duplicate_overlapping_or_reordered_samples_are_rejected(self):
        for change in ('long','duplicate','overlap','reordered'):
            base,defended=fixtures()
            samples=defended['blade_contact_impulse_samples']
            if change=='long':samples[0]['interval_s']=6.
            elif change=='duplicate':samples.append(copy.deepcopy(samples[1]))
            elif change=='overlap':samples[1]['time_s']=3.001
            else:samples.reverse()
            with self.assertRaises(ValueError):self.gate(base,defended)


if __name__=='__main__':unittest.main()
