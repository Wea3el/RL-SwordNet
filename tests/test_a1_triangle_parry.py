import copy
import unittest

import numpy as np

from sim.a1_triangle_parry import parry_program,triangle_defence_gate


class TriangleParryTests(unittest.TestCase):
    def setUp(self):
        self.base={'native_execution_pass':True,'source_unchanged':True,'failure':None,
                   'runtime_config':{'minimum_blade_force_n':.1,'minimum_blade_impulse_ns':.02}}
        self.base_audit={'all_entries':[{'agent':'a1','time_s':3.,'region':'upper'}],
                         'intended_region_first':True,'initial_overlap':{},'sweep_failures':{}}
        self.defended=copy.deepcopy(self.base)
        self.audit={'all_entries':[],'initial_overlap':{},'sweep_failures':{}}
        self.contacts=[{'interval_start_s':2.0,'interval_end_s':2.1,'contacts':[
            {'category':'cross_blade_blade','geom1':'a1_shinai_blade_collision_2',
             'geom2':'a2_shinai_blade_collision_0','normal_force_n':1.}]}]
        self.match={'same_attack':True,'same_start':True,'complete_trace':True}

    def gate(self):
        return triangle_defence_gate(self.base,self.base_audit,self.defended,self.audit,
                                     self.contacts,matching_checks=self.match)

    def test_matched_contact_and_full_window_empty_is_conditional_success(self):
        self.assertEqual(self.gate()['status'],'qualified')
        self.assertFalse(self.gate()['admitted_to_training'])

    def test_no_contact_is_not_a_block(self):
        self.contacts=[]
        self.assertEqual(self.gate()['status'],'failed')

    def test_late_contact_does_not_prove_interception(self):
        self.contacts[0].update(interval_start_s=3.1,interval_end_s=3.2)
        self.assertEqual(self.gate()['status'],'failed')

    def test_straddling_contact_force_is_not_backdated_before_hit(self):
        self.contacts[0].update(interval_start_s=2.99,interval_end_s=3.01)
        self.contacts[0]['contacts'][0]['normal_force_n']=3.
        result=self.gate()
        self.assertEqual(result['status'],'failed')
        self.assertEqual(result['pre_hit_weapon_impulse_ns'],0.)

    def test_baseline_hit_frame_start_is_conservative_cutoff(self):
        self.base_audit['all_entries'][0]['step_t0_s']=2.99
        self.contacts[0].update(interval_start_s=2.99,interval_end_s=3.)
        self.contacts[0]['contacts'][0]['normal_force_n']=100.
        self.assertEqual(self.gate()['status'],'failed')

    def test_off_region_and_recovery_hit_are_failures(self):
        for region in ('upper','left','right'):
            self.audit['all_entries']=[{'agent':'a1','region':region,'time_s':10.}]
            self.assertEqual(self.gate()['status'],'failed')

    def test_counter_does_not_hide_later_hit(self):
        self.audit['all_entries']=[{'agent':'a2','region':'left','time_s':1.},
                                   {'agent':'a1','region':'upper','time_s':10.}]
        self.assertEqual(self.gate()['status'],'failed')

    def test_counter_is_separate_not_pure_parry(self):
        self.audit['all_entries']=[{'agent':'a2','region':'left','time_s':2.5}]
        self.assertEqual(self.gate()['status'],'defence_with_counter_candidate')

    def test_weak_contact_fails_existing_evidence_threshold(self):
        self.contacts[0]['contacts'][0]['normal_force_n']=.001
        self.assertEqual(self.gate()['status'],'failed')

    def test_fault_or_truncation_is_invalid_not_success(self):
        self.defended['native_execution_pass']=False
        self.defended['failure']=['excessive_proxy_penetration']
        self.match['complete_trace']=False
        self.assertEqual(self.gate()['status'],'invalid')

    def test_different_attack_is_unmatched(self):
        self.match['same_attack']=False
        self.assertEqual(self.gate()['status'],'unmatched')

    def test_wrong_attacker_audit_is_unmatched(self):
        self.base_audit['attacker_slot']='a2'
        self.assertEqual(self.gate()['status'],'unmatched')

    def test_mirrored_attacker_requires_matching_metadata(self):
        self.base_audit['all_entries'][0]['agent']='a2'
        self.base_audit['attacker_slot']='a2'
        self.audit['attacker_slot']='a2'
        def mirrored():
            return triangle_defence_gate(self.base,self.base_audit,self.defended,
                self.audit,self.contacts,matching_checks=self.match,attacker='a2')
        self.assertEqual(mirrored()['status'],'unmatched')
        self.base['metadata']={'attacker_slot':'a2'}
        self.defended['metadata']={'attacker_slot':'a2'}
        self.assertEqual(mirrored()['status'],'qualified')

    def test_missing_baseline_threat_does_not_credit_defence(self):
        self.base_audit['all_entries']=[]
        self.assertEqual(self.gate()['status'],'no_threat')

    def test_initial_overlap_and_sweep_error_are_invalid(self):
        for key in ('initial_overlap','sweep_failures'):
            self.audit[key]={'upper':'invalid evidence'}
            self.assertEqual(self.gate()['status'],'invalid')
            self.audit[key]={}

    def test_non_cross_robot_contact_is_not_used(self):
        self.contacts[0]['contacts'][0]['geom2']='a1_shinai_blade_collision_0'
        self.assertEqual(self.gate()['status'],'failed')

    def test_program_includes_full_recovery_and_tail(self):
        q=np.ones(7)
        spec={'start_s':1.,'approach_s':.7,'release_s':8.,'return_s':1.}
        p=parry_program(q,q+.1,spec,12.)
        np.testing.assert_array_equal(p.target(0),q)
        np.testing.assert_array_equal(p.target(12),q)
        self.assertLessEqual(p.end_s+2.,12.)
        with self.assertRaises(ValueError):
            parry_program(q,q+.1,spec,10.)


if __name__=='__main__':
    unittest.main()
