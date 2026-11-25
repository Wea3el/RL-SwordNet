"""Synthetic evidence-contract tests, never claims about fencing performance."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import math
from unittest.mock import patch

from sim.a1_coverage_audit import (ATTACKS,DEFENCES,REQUIRED_MATCHING,CHAIN_CHECKS,action_contract,
    candidate_reachability,chain_reached,generate,geometry_identity,pair_verdict,summarize_cell,
    same_pair_definition,trace_files_bound)


GEOMETRY='a'*64


def bank():
    q=[3.,1.,3.,1.,2.,1.,0.]
    r=q.copy();r[0]+=.1
    return {'scoring':{'fixture':True},'geometry_contract_sha256':GEOMETRY,
        'guards':{'left':{'qpos_rad':q},'right':{'qpos_rad':r}},'initial_guard':'left',
        'actions':{
            'down':{'kind':'attack','family':'cut_down_left_to_right','candidate_pass':True,
                'start_guard':'left','end_guard':'right','target_region':'left',
                'knot_time_s':[0.,1.],'knot_qpos':[q,r]},
            'parry':{'kind':'defence','family':'parry_left','candidate_pass':True,
                'start_guard':'right','end_guard':'left','target_region':'left',
                'active_interval_s':[.2,.8],
                'knot_time_s':[0.,1.],'knot_qpos':[r,q]}}}


def record():
    return {'attack_name':'down','defence_name':'parry','source_report_sha256':'BANK',
        'geometry_contract_sha256':GEOMETRY,'matching_checks':{k:True for k in REQUIRED_MATCHING},
        'baseline_valid_threat':True,'all_regions_checked':True,'attacker_slot':'a1',
        'defence_offset_s':0.,'defence_start_s':.75,'baseline_first_triangle_hit_s':2.,
        'first_pre_hit_weapon_contact_s':1.5,'successful_physical_defence':True,
        'meaningful_pre_hit_contact':True,'numerical_and_full_execution_clean':True,
        'geometric_evidence_clean':True,'attacker_entries_entire_window':[],
        'defender_entries_entire_window':[]}


def upright():
    sample={'pass':True,'sample_count':10,'minimum_elevation_deg':30.,'minimum_grip_height_m':.5}
    return {'qualified_upright_example':True,'actual_orientation_pass':True,
        'reference_or_command_used':False,'source_unchanged':True,'comparison_unchanged':True,
        'source_native_execution_pass':True,'independent_qualified_defence_bound':True,
        'all_active_recorded_frames':sample.copy(),'all_nonzero_loaded_weapon_contact_frames':sample.copy(),
        'defender_slot':'a2','active_interval_s':[.95,1.55],
        'min_elevation_deg':25.,'min_grip_height_m':.35}


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.b=bank()
        self.actions={n:action_contract(n,a,self.b['guards'])for n,a in self.b['actions'].items()}

    def verdict(self,r=None,**kw):
        defaults=dict(bank_sha='BANK',expected_geometry=GEOMETRY,actions=self.actions,
            native_records=[{'geometry_contract_sha256':GEOMETRY,'native_execution_pass':True,
                             'qpos_resets_after_initialization':0,'final_settled':{'a1':True,'a2':True}}]*2,upright=upright(),
            snapshot_matches_bank=True,scoring_matches_bank=True,
            current_implementation_verified=True,upright_sources_verified=True,
            current_evidence_verified=True)
        defaults.update(kw)
        return pair_verdict(r or record(),**defaults)

    def test_exact_current_pair_is_scoped_not_general(self):
        v=self.verdict()
        self.assertEqual(v['status'],'qualified_upright_pair')
        self.assertFalse(v['arbitrary_A2_defence_proven'])
        self.assertFalse(v['runtime_entry_reachable_proven'])

    def test_unchanged_bank_cannot_bypass_current_implementation(self):
        v=self.verdict(current_implementation_verified=False)
        self.assertFalse(v['qualified_design_pair_pass'])
        self.assertIn('current_implementation_not_verified',v['reasons'])

    def test_mutated_native_files_cannot_reuse_historical_checks(self):
        v=self.verdict(current_evidence_verified=False)
        self.assertFalse(v['qualified_design_pair_pass'])
        self.assertIn('current_native_artifacts_not_bound',v['reasons'])

    def test_trace_hash_check_detects_a_changed_file(self):
        with tempfile.TemporaryDirectory()as tmp:
            path=Path(tmp)
            for name in ('summary.json','trace.npz','scene.xml'):
                (path/name).write_bytes(name.encode())
            audit={'native_trace_source_sha256':{
                p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in path.iterdir()}}
            self.assertTrue(trace_files_bound(path,audit))
            (path/'trace.npz').write_bytes(b'changed')
            self.assertFalse(trace_files_bound(path,audit))
            audit['native_trace_source_sha256'].pop('trace.npz')
            self.assertFalse(trace_files_bound(path,audit))

    def test_generator_checks_implementation_for_exact_same_bank(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);p=root/'bank.json';p.write_text(json.dumps(self.b))
            pair=root/'pair';pair.mkdir()
            r=record();r['source_report_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
            (pair/'source_report_snapshot.json').write_bytes(p.read_bytes())
            (pair/'scoring_snapshot.json').write_text(json.dumps(self.b['scoring']))
            (pair/'comparison.json').write_text(json.dumps(r))
            for label in ('baseline','defended'):
                (pair/label).mkdir()
                (pair/label/'summary.json').write_text(json.dumps({
                    'geometry_contract_sha256':GEOMETRY,'native_execution_pass':True,
                    'qpos_resets_after_initialization':0,'final_settled':{'a1':True,'a2':True}}))
            with patch('sim.a1_coverage_audit.native_dependencies_current',return_value=False):
                result=generate(p,[pair])
        row=result['all_evidence'][0]
        self.assertFalse(row['qualified_design_pair_pass'])
        self.assertIn('current_implementation_not_verified',row['reasons'])

    def test_upright_cannot_use_shorter_phase_or_wrong_slot(self):
        for key,value in (('active_interval_s',[1.,1.2]),('defender_slot','a1')):
            u=upright();u[key]=value
            self.assertFalse(self.verdict(upright=u)['qualified_design_pair_pass'])
        self.assertFalse(self.verdict(upright_sources_verified=False)['qualified_design_pair_pass'])

    def test_unrelated_action_change_does_not_change_exact_pair(self):
        revised=copy.deepcopy(self.b)
        revised['actions']['unrelated']={'new_candidate':True}
        self.assertTrue(same_pair_definition(self.b,revised,'down','parry'))
        revised['actions']['parry']['knot_time_s'][1]=1.1
        self.assertFalse(same_pair_definition(self.b,revised,'down','parry'))

    def test_bound_unchanged_pair_never_bypasses_geometry(self):
        r=record();r['source_report_sha256']='OLDER_BANK'
        self.assertTrue(self.verdict(r,unchanged_bound_pair=True)['physical_defence_pass'])
        self.assertFalse(self.verdict(r,unchanged_bound_pair=False)['physical_defence_pass'])
        r['geometry_contract_sha256']='different'
        self.assertFalse(self.verdict(r,unchanged_bound_pair=True)['physical_defence_pass'])

    def test_missing_family_never_counts_as_pass(self):
        cell=summarize_cell([],['parry'],[self.verdict()])
        self.assertEqual(cell['status'],'missing_attack_family')
        self.assertEqual(summarize_cell(['down'],[],[])['status'],'missing_defence_family')

    def test_no_threat_cannot_be_successful_defence(self):
        r=record();r['baseline_valid_threat']=False
        v=self.verdict(r)
        self.assertFalse(v['valid_matched_baseline'])
        self.assertFalse(v['physical_defence_pass'])

    def test_wrong_entry_or_truncated_tail_rejected(self):
        for key in ('baseline_holds_exact_defence_entry','complete_defence_plus_two_second_tail',
                    'complete_attack_plus_two_second_tail','same_initial_qpos'):
            with self.subTest(key=key):
                r=record();r['matching_checks'][key]=False
                self.assertFalse(self.verdict(r)['valid_matched_baseline'])

    def test_old_geometry_is_not_inherited(self):
        v=self.verdict(native_records=[{'geometry_contract_sha256':'OLD'}]*2)
        self.assertFalse(v['valid_matched_baseline'])
        self.assertIn('each_native_summary_must_bind_geometry',v['reasons'])

    def test_no_frozen_geometry_cannot_qualify(self):
        self.assertFalse(self.verdict(expected_geometry=None)['physical_defence_pass'])
        self.assertFalse(self.verdict(native_records=[])['physical_defence_pass'])

    def test_bank_or_scoring_changes_invalidate_pair(self):
        r=record();r['source_report_sha256']='OTHER'
        self.assertFalse(self.verdict(r)['physical_defence_pass'])
        self.assertFalse(self.verdict(scoring_matches_bank=False)['physical_defence_pass'])
        self.assertFalse(self.verdict(snapshot_matches_bank=False)['physical_defence_pass'])

    def test_after_contact_hit_is_failed_not_success(self):
        r=record();r['attacker_entries_entire_window']=[{'time_s':3.}]
        v=self.verdict(r)
        self.assertTrue(v['valid_matched_baseline'])
        self.assertEqual(v['status'],'valid_threat_defence_failed')

    def test_loaded_contact_orientation_cannot_hide_outside_active(self):
        u=upright();u['all_nonzero_loaded_weapon_contact_frames']['minimum_elevation_deg']=19.
        self.assertEqual(self.verdict(upright=u)['status'],'physical_pair_only_upright_unverified')
        u=upright();u['all_nonzero_loaded_weapon_contact_frames']['sample_count']=0
        self.assertFalse(self.verdict(upright=u)['upright_defence_pass'])

    def test_two_slots_not_two_copies_of_same_slot(self):
        first=self.verdict();again=self.verdict()
        cell=summarize_cell(['down'],['parry'],[first,again])
        self.assertEqual(cell['status'],'one_slot_scoped_pair_pass')
        other=copy.deepcopy(first);other['attacker_slot']='a2'
        cell=summarize_cell(['down'],['parry'],[first,other])
        self.assertEqual(cell['status'],'both_slots_scoped_pair_pass')
        self.assertFalse(cell['continuous_timing_window_proven'])

    def test_distinct_variants_or_offsets_cannot_form_a_mirrored_pair(self):
        first=self.verdict();second=copy.deepcopy(first);second['attacker_slot']='a2'
        second['defence_offset_s']=.1
        cell=summarize_cell(['down'],['parry'],[first,second])
        self.assertNotEqual(cell['status'],'both_slots_scoped_pair_pass')
        self.assertEqual(cell['same_variant_and_offset_both_slots'],[])
        second['defence_offset_s']=0.;second['defence']='different'
        cell=summarize_cell(['down'],['parry','different'],[first,second])
        self.assertNotEqual(cell['status'],'both_slots_scoped_pair_pass')

    def test_no_implicit_bridge_or_pose_alias(self):
        self.assertEqual(candidate_reachability(self.actions,[]),[])
        bad=copy.deepcopy(self.b['actions']['down']);bad['knot_qpos'][0][0]+=.01
        self.assertFalse(action_contract('down',bad,self.b['guards'])['candidate_contract_pass'])
        self.assertEqual(candidate_reachability(self.actions,['left']),['left','right'])

    def test_action_limits_use_current_cad_zero_mapping_not_hardware_numbers(self):
        for j3_deg, should_pass in ((-30., True), (110., False)):
            b=bank()
            for guard in b['guards'].values(): guard['qpos_rad'][3]=math.radians(j3_deg)
            row=b['actions']['down']
            result=action_contract('down',row,b['guards'])
            self.assertEqual(result['candidate_contract_pass'],should_pass)
            self.assertEqual('joint_bounds_exceeded' in result['reasons'],not should_pass)

    def test_native_reachability_requires_current_bound_chain(self):
        r={'source_report_sha256':'BANK','chain_feasibility_pass':True,'checks':{k:True for k in CHAIN_CHECKS},
           'initial_guard':'left','final_guard':'right','scripted_sequence':['down'],
           'executor_events':[{'event':'started','action':'down','node':'left','time_s':.1},
                              {'event':'completed','action':'down','node':'right','time_s':1.1}]}
        native={'native_execution_pass':True,'geometry_contract_sha256':GEOMETRY,
                'qpos_resets_after_initialization':0}
        self.assertEqual(chain_reached(r,bank_sha='BANK',expected_geometry=GEOMETRY,
            native_summary=native,actions=self.actions,current_implementation_verified=True,
            current_evidence_verified=True),['left','right'])
        self.assertEqual(chain_reached(r,bank_sha='BANK',expected_geometry=GEOMETRY,
            native_summary=native,actions=self.actions,current_implementation_verified=False,
            current_evidence_verified=True),[])
        native['geometry_contract_sha256']='OLD'
        self.assertEqual(chain_reached(r,bank_sha='BANK',expected_geometry=GEOMETRY,
            native_summary=native,actions=self.actions,current_implementation_verified=True,
            current_evidence_verified=True),[])

    def test_manifest_is_descriptive_and_never_admits_training(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp)/'bank.json';p.write_text(json.dumps(self.b))
            result=generate(p,[])
        self.assertFalse(result['admitted_to_training'])
        self.assertEqual(len(result['missing_families']),len(ATTACKS)+len(DEFENCES)-2)
        self.assertFalse(result['guard_choices']['right']['native_runtime_entry_reachable'])

    def test_conflicting_geometry_tokens_fail_closed(self):
        self.assertIsNone(geometry_identity({'geometry_contract_sha256':GEOMETRY},
                                           {'geometry_contract_sha256':'OTHER'}))

    def test_high_parry_does_not_invent_a_25_degree_requirement(self):
        self.actions['parry']['family']='parry_high'
        self.actions['parry']['upright_requirement']=False
        v=self.verdict(upright=None)
        self.assertTrue(v['qualified_design_pair_pass'])
        self.assertFalse(v['upright_defence_pass'])
        self.assertFalse(v['upright_requirement'])

    def test_profile_bank_blade_identity_matches_runtime_token(self):
        self.assertEqual(geometry_identity({'blade_geometry':{'sha256':GEOMETRY}}),GEOMETRY)
        self.assertEqual(geometry_identity({'metadata':{'collision_geometry_sha256':GEOMETRY}}),GEOMETRY)

    def test_present_but_failed_action_is_not_hidden_as_absent(self):
        fixture=bank();fixture['actions']['down']['candidate_pass']=False
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp)/'bank.json';p.write_text(json.dumps(fixture))
            r=generate(p,[])
        self.assertNotIn('cut_down_left_to_right',r['missing_families'])
        self.assertIn('cut_down_left_to_right',r['unavailable_families'])
        self.assertEqual(r['attack_defence_matrix']['cut_down_left_to_right']['parry_left']['status'],
                         'attack_family_not_diagnostic_eligible')


if __name__=='__main__':unittest.main()
