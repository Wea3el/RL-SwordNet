"""Strict read-only A1 family/interaction coverage manifest generator.

This never starts a simulator, authorizes training, or promotes old geometry
evidence. Missing identity, threats, entries or complete windows stay missing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from .motion_spline import C2JointSpline
from .assembly import ROOT
from .joint_constraints import load_joint_constraints, model_joint_ranges_rad


ATTACKS=('cut_down_left_to_right','cut_down_right_to_left','cut_up_left_to_right',
         'cut_up_right_to_left','cut_down_center','thrust_center')
DEFENCES=('parry_left','parry_right','parry_high')
REQUIRED_MATCHING=('runtime_config','joint_constraints','model_source_sha256','attack_reference',
    'same_initial_qpos','same_initial_qvel','baseline_holds_exact_defence_entry',
    'complete_attack_plus_two_second_tail','complete_defence_plus_two_second_tail',
    'baseline_complete_native_trace','defended_complete_native_trace',
    'baseline_both_slots_finally_settled','defended_both_slots_finally_settled',
    'source_report_unchanged','scoring_source_unchanged','baseline_hit_within_attack_curve')
CHAIN_CHECKS=('all_actions_started_in_order','all_actions_completed_in_order','all_script_requests_consumed',
    'requests_use_observed_entry_state','each_edge_runs_its_complete_clock','no_post_initialization_resets',
    'native_execution_pass','evidence_complete','source_unchanged','implementation_unchanged',
    'scoring_unchanged','both_finally_settled','triangle_evidence_valid',
    'all_starts_at_declared_nodes','all_completions_at_declared_nodes',
    'next_action_never_precedes_actual_completion','final_node_matches_plan')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geometry_identity(*records):
    """Explicit runtime geometry tokens only; do not infer identity from names."""
    found=set()
    for record in records:
        if not isinstance(record,dict):continue
        for mapping in (record,record.get('metadata',{}),record.get('runtime_config',{})):
            for key in ('geometry_contract_sha256','collision_geometry_sha256','shared_geometry_sha256'):
                value=mapping.get(key)
                if isinstance(value,str)and value:found.add(value)
            for key in ('geometry_binding','geometry_contract','blade_geometry'):
                value=mapping.get(key)
                if isinstance(value,dict)and isinstance(value.get('sha256'),str):found.add(value['sha256'])
    return next(iter(found))if len(found)==1 else None


def family(name,row):
    explicit=row.get('family',name)
    if explicit in ATTACKS and row.get('kind')=='attack':return explicit
    if explicit in DEFENCES and row.get('kind')=='defence':return explicit
    return None


def action_contract(name,row,guards):
    reasons=[]
    if row.get('candidate_pass')is not True:reasons.append('source_candidate_not_passed')
    start,end=row.get('start_guard'),row.get('end_guard')
    if start not in guards or end not in guards:reasons.append('missing_exact_guard')
    try:
        curve=C2JointSpline(row['knot_time_s'],row['knot_qpos'])
        if curve.knot_time_s[0]!=0.:reasons.append('reference_must_begin_at_zero')
        if start in guards and list(curve.knot_qpos[0])!=guards[start]['qpos_rad']:reasons.append('wrong_entry_qpos')
        if end in guards and list(curve.knot_qpos[-1])!=guards[end]['qpos_rad']:reasons.append('wrong_exit_qpos')
        # References remain CAD q, not hardware angles: apply the current
        # confirmed zero/sign map rather than reusing the table as raw q.
        bounds=model_joint_ranges_rad(load_joint_constraints())
        if any(not low<=q<=high for controls in curve.bezier_controls.reshape(-1,7)
               for q,(low,high)in zip(controls,bounds)):
            reasons.append('joint_bounds_exceeded')
        duration=curve.duration_s
    except (KeyError,ValueError,TypeError,IndexError):
        reasons.append('invalid_finite_c2_reference');duration=None
    return {'name':name,'family':family(name,row),'kind':row.get('kind'),
        'start_guard':start,'end_guard':end,'target_region':row.get('target_region'),
        'duration_s':duration,'candidate_contract_pass':not reasons,'reasons':reasons,
        'upright_requirement':(family(name,row)in ('parry_left','parry_right') or
                               row.get('min_elevation_deg') is not None),
        'active_interval_s':row.get('active_upright_interval_s',row.get('active_interval_s')),
        'minimum_elevation_deg':float(row.get('orientation_requirement',{}).get(
            'minimum_elevation_deg',row.get('min_elevation_deg',25.))),
        'minimum_grip_height_m':float(row.get('orientation_requirement',{}).get(
            'minimum_grip_height_m',row.get('minimum_grip_height_m',row.get('min_grip_height',.35)))),
        'policy_callable_declared':row.get('policy_callable') is True}


def same_pair_definition(saved, current, attack, defence):
    """Only the two tested references can transfer across bank revisions.

    A changed unrelated action cannot invalidate a physical pair experiment,
    but this check never transfers graph, policy or action-mask conclusions.
    Exact entry/exit names AND joint states are required; no guard aliases.
    """
    fields=('kind','family','direction_intent_attacker_frame',
            'knot_time_s','knot_qpos','start_guard','end_guard','target_region',
            'active_interval_s','active_upright_interval_s','min_elevation_deg',
            'min_grip_height','minimum_grip_height_m','orientation_requirement')
    try:
        for name in (attack,defence):
            a,b=saved['actions'][name],current['actions'][name]
            if any(a.get(k)!=b.get(k)for k in fields):return False
            for key in ('start_guard','end_guard'):
                if saved['guards'][a[key]]['qpos_rad']!=current['guards'][b[key]]['qpos_rad']:return False
        return saved.get('scoring')==current.get('scoring')
    except (KeyError,TypeError):return False


def hashes_current(hashes):
    if not isinstance(hashes,dict)or not hashes:return False
    for relative,expected in hashes.items():
        path=(ROOT/relative).resolve()
        if not path.is_relative_to(ROOT)or not path.is_file()or digest(path)!=expected:return False
    return True


def native_dependencies_current(records):
    """Reject stale implementations even when the bank itself is unchanged."""
    return bool(records)and all(hashes_current(r.get('source_sha256'))for r in records)


def upright_contract_matches(upright,record,action):
    """The checked phase must be the whole phase declared before the trial."""
    try:
        local=action['active_interval_s']
        absolute=[record['defence_start_s']+local[0],record['defence_start_s']+local[1]]
        actual=upright['active_interval_s']
        return (len(local)==2 and len(actual)==2 and local[0]<local[1] and
            all(math.isfinite(v)for v in absolute+list(actual))and
            all(abs(a-b)<=1e-9 for a,b in zip(absolute,actual))and
            upright['defender_slot']==('a2'if record['attacker_slot']=='a1'else'a1')and
            upright['min_elevation_deg']>=max(25.,action['minimum_elevation_deg'])and
            upright['min_grip_height_m']>=max(.35,action['minimum_grip_height_m']))
    except (KeyError,TypeError,ValueError,IndexError):return False


def trace_files_bound(folder,audit):
    expected=audit.get('native_trace_source_sha256',{})
    return (set(expected)=={'scene.xml','trace.npz','summary.json'}and
            all((folder/name).is_file()and digest(folder/name)==value
                for name,value in expected.items()))


def pair_files_bound(folder,record,snapshot):
    """Bind current artifacts, not only historical 'unchanged' booleans."""
    try:
        if not all(trace_files_bound(folder/label,record[label+'_triangle_audit'])
                   for label in ('baseline','defended')):return False
        if digest(folder/'defended/contacts.json')!=record['native_contact_source_sha256']:return False
        from .multiguard_defence_probe import plan_pair,recorded_plan_checks
        plan=plan_pair(snapshot,record['attack_name'],record['defence_name'],
            defence_offset_s=record['defence_offset_s'],attacker=record['attacker_slot'])
        return all(all(recorded_plan_checks(folder/label,plan[label]).values())
                   for label in ('baseline','defended'))
    except (KeyError,ValueError,TypeError,OSError):return False


def pair_verdict(record,*,bank_sha,expected_geometry,actions,native_records=(),upright=None,
                 scoring_matches_bank=False,snapshot_matches_bank=False,
                 unchanged_bound_pair=False,current_implementation_verified=False,
                 upright_sources_verified=False,current_evidence_verified=False):
    """A valid negative control is distinct from an invalid baseline."""
    reasons=[]
    if not current_implementation_verified:reasons.append('current_implementation_not_verified')
    if not current_evidence_verified:reasons.append('current_native_artifacts_not_bound')
    attack,defence=record.get('attack_name'),record.get('defence_name')
    if attack not in actions or defence not in actions:reasons.append('missing_or_wrong_named_action')
    elif not all(actions[n]['candidate_contract_pass']for n in (attack,defence)):
        reasons.append('source_action_contract_not_passed')
    if record.get('source_report_sha256')!=bank_sha and not unchanged_bound_pair:
        reasons.append('different_bank_snapshot')
    if not snapshot_matches_bank:reasons.append('source_snapshot_missing_or_changed')
    if not scoring_matches_bank:reasons.append('scoring_snapshot_missing_or_changed')
    actual_geometry=geometry_identity(record,*native_records)
    if not expected_geometry:reasons.append('expected_geometry_not_frozen')
    elif actual_geometry!=expected_geometry:reasons.append('missing_or_different_runtime_geometry')
    if len(native_records)!=2:reasons.append('both_native_summaries_required')
    elif expected_geometry and any(geometry_identity(r)!=expected_geometry for r in native_records):
        reasons.append('each_native_summary_must_bind_geometry')
    for native in native_records:
        if (native.get('native_execution_pass')is not True or native.get('qpos_resets_after_initialization')!=0 or
            not native.get('final_settled')or not all(v is True for v in native.get('final_settled',{}).values())):
            reasons.append('native_execution_reset_or_settling_failure')
    matching=record.get('matching_checks',{})
    for name in REQUIRED_MATCHING:
        if matching.get(name)is not True:reasons.append('matching_'+name)
    if any(value is not True for value in matching.values()):reasons.append('another_matching_check_failed')
    if record.get('baseline_valid_threat')is not True:reasons.append('baseline_not_a_valid_threat')
    if record.get('all_regions_checked')is not True:reasons.append('not_all_target_regions_checked')
    if record.get('attacker_slot')not in ('a1','a2'):reasons.append('missing_physical_slot')
    offset=record.get('defence_offset_s')
    if not isinstance(offset,(int,float))or isinstance(offset,bool)or not math.isfinite(offset):
        reasons.append('missing_finite_offset')
    baseline_hit=record.get('baseline_first_triangle_hit_s')
    contact=record.get('first_pre_hit_weapon_contact_s')
    if not isinstance(baseline_hit,(int,float))or not math.isfinite(baseline_hit):reasons.append('missing_baseline_hit_time')
    valid_baseline=not reasons
    physical=(valid_baseline and record.get('successful_physical_defence')is True and
              record.get('meaningful_pre_hit_contact')is True and
              record.get('numerical_and_full_execution_clean')is True and
              record.get('geometric_evidence_clean')is True and
              record.get('attacker_entries_entire_window')==[] and
              record.get('defender_entries_entire_window')==[] and
              isinstance(contact,(int,float))and math.isfinite(contact)and contact<baseline_hit)
    orientation=False
    require_upright=(actions.get(defence,{}).get('upright_requirement',True))
    if physical and upright:
        active=upright.get('all_active_recorded_frames',{})
        loaded=upright.get('all_nonzero_loaded_weapon_contact_frames',{})
        orientation=(upright_sources_verified and upright_contract_matches(upright,record,actions[defence])and
            upright.get('qualified_upright_example')is True and
            upright.get('actual_orientation_pass')is True and upright.get('reference_or_command_used')is False and
            upright.get('source_unchanged')is True and upright.get('comparison_unchanged')is True and
            upright.get('source_native_execution_pass')is True and
            upright.get('independent_qualified_defence_bound')is True and
            active.get('pass')is True and loaded.get('pass')is True and
            active.get('sample_count',0)>0 and loaded.get('sample_count',0)>0 and
            active.get('minimum_elevation_deg',-90)>=25. and loaded.get('minimum_elevation_deg',-90)>=25. and
            active.get('minimum_grip_height_m',0)>=.35 and loaded.get('minimum_grip_height_m',0)>=.35 and
            upright.get('min_elevation_deg',0)>=25. and upright.get('min_grip_height_m',0)>=.35)
    design_pass=physical and (orientation or not require_upright)
    status=('invalid_or_unbound_evidence'if not valid_baseline else
            'qualified_upright_pair'if orientation else
            'qualified_physical_pair_no_upright_requirement'if design_pass else
            'physical_pair_only_upright_unverified'if physical else 'valid_threat_defence_failed')
    return {'attack':attack,'defence':defence,'attacker_slot':record.get('attacker_slot'),
        'defence_offset_s':record.get('defence_offset_s'),'status':status,'reasons':reasons,
        'valid_matched_baseline':valid_baseline,'physical_defence_pass':bool(physical),
        'upright_defence_pass':bool(orientation),'upright_requirement':bool(require_upright),
        'qualified_design_pair_pass':bool(design_pass),'geometry_identity':actual_geometry,
        'runtime_entry_reachable_proven':False,'arbitrary_A2_defence_proven':False}


def chain_reached(record,*,bank_sha,expected_geometry,native_summary,actions,
                  current_implementation_verified=False,current_evidence_verified=False):
    """Only bind completed current-geometry runtime events, not graph guesses."""
    if (not current_implementation_verified or not current_evidence_verified or
        record.get('source_report_sha256')!=bank_sha or not expected_geometry or
        geometry_identity(native_summary)!=expected_geometry or record.get('chain_feasibility_pass')is not True or
        any(value is not True for value in record.get('checks',{}).values()) or
        any(record.get('checks',{}).get(k)is not True for k in CHAIN_CHECKS) or
        native_summary.get('native_execution_pass')is not True or
        native_summary.get('qpos_resets_after_initialization')!=0):return []
    names=record.get('scripted_sequence',[])
    starts=[x for x in record.get('executor_events',[])if x.get('event')=='started']
    ends=[x for x in record.get('executor_events',[])if x.get('event')=='completed']
    if not names or [x.get('action')for x in starts]!=names or [x.get('action')for x in ends]!=names:return []
    previous=record.get('initial_guard')
    for name,start,end in zip(names,starts,ends):
        if name not in actions or not actions[name]['candidate_contract_pass']:return []
        row=actions[name]
        if (row['start_guard']!=previous or start.get('node')!=row['start_guard']or
            end.get('node')!=row['end_guard']or
            end.get('time_s',-1)-start.get('time_s',0)<row['duration_s']-1e-8):return []
        previous=row['end_guard']
    if previous!=record.get('final_guard'):return []
    reached={record.get('initial_guard')}
    for event in record.get('executor_events',[]):
        if event.get('event')!='completed':continue
        name=event.get('action')
        if name not in actions or event.get('node')!=actions[name]['end_guard']:return []
        reached.add(event['node'])
    return sorted(x for x in reached if isinstance(x,str))


def candidate_reachability(actions,initial_guards):
    reached=set(initial_guards)
    changed=True
    while changed:
        changed=False
        for row in actions.values():
            if row['candidate_contract_pass']and row['start_guard']in reached and row['end_guard']not in reached:
                reached.add(row['end_guard']);changed=True
    return sorted(reached)


def summarize_cell(attack_names,defence_names,records):
    evidence=[r for r in records if r['attack']in attack_names and r['defence']in defence_names]
    qualified=[r for r in evidence if r['qualified_design_pair_pass']]
    slots=sorted({r['attacker_slot']for r in qualified})
    offsets={slot:sorted({r['defence_offset_s']for r in qualified if r['attacker_slot']==slot})for slot in slots}
    groups={}
    for r in qualified:
        key=(r['attack'],r['defence'],r['defence_offset_s'])
        groups.setdefault(key,set()).add(r['attacker_slot'])
    complete=[{'attack':a,'defence':d,'offset_s':t}for(a,d,t),s in groups.items()if s=={'a1','a2'}]
    if not attack_names:status='missing_attack_family'
    elif not defence_names:status='missing_defence_family'
    elif complete:status='both_slots_scoped_pair_pass'
    elif slots==['a1','a2']:status='different_variant_or_timing_single_slot_passes'
    elif qualified:status='one_slot_scoped_pair_pass'
    elif any(r['physical_defence_pass']for r in evidence):status='physical_only_upright_unverified'
    elif any(r['valid_matched_baseline']for r in evidence):status='tested_valid_threat_no_defence_pass'
    else:status='no_current_geometry_valid_pair_evidence'
    return {'status':status,'attack_variants':attack_names,'defence_variants':defence_names,
        'qualified_slots':slots,'tested_successful_offsets_s':offsets,
        'same_variant_and_offset_both_slots':complete,
        'continuous_timing_window_proven':False,'evidence':evidence,
        'all_future_A2_actions_covered':False}


def generate(bank,results_dirs,*,expected_geometry=None,initial_guards=()):
    bank=Path(bank).resolve();raw=bank.read_bytes();report=json.loads(raw);bank_sha=hashlib.sha256(raw).hexdigest()
    expected_geometry=expected_geometry or geometry_identity(report)
    actions={n:action_contract(n,r,report.get('guards',{}))for n,r in report.get('actions',{}).items()}
    roots=[Path(p).resolve()for p in results_dirs]
    audits={};seen_audits=set()
    for root in roots:
        for path in root.rglob('audit.json'):
            if path in seen_audits:continue
            seen_audits.add(path)
            try:row=json.loads(path.read_bytes())
            except (ValueError,OSError):continue
            if row.get('comparison_sha256'):audits.setdefault(row['comparison_sha256'],[]).append((path,row))
    records=[];seen_pairs=set()
    for root in roots:
        for path in root.rglob('comparison.json'):
            if path in seen_pairs:continue
            seen_pairs.add(path)
            try:row=json.loads(path.read_bytes())
            except (ValueError,OSError):continue
            if 'successful_physical_defence'not in row:continue
            native=[]
            for name in ('baseline','defended'):
                summary=path.parent/name/'summary.json'
                if summary.exists():native.append(json.loads(summary.read_bytes()))
            linked=audits.get(digest(path),[])
            upright=linked[0][1]if len(linked)==1 else None
            snapshot=path.parent/'source_report_snapshot.json'
            scoring=path.parent/'scoring_snapshot.json'
            snapshot_hash=digest(snapshot)if snapshot.exists()else None
            declared_source=bool(snapshot_hash and snapshot_hash==row.get('source_report_sha256'))
            saved=json.loads(snapshot.read_bytes())if declared_source else {}
            reusable=(declared_source and native_dependencies_current(native)and
                      same_pair_definition(saved,report,row.get('attack_name'),row.get('defence_name')))
            current_impl=(native_dependencies_current(native)and hashes_current(row.get('probe_source_sha256')))
            upright_bound=bool(upright and hashes_current({
                'sim/a1_upright_audit.py':upright.get('implementation_sha256')})and
                set(upright.get('source_sha256',{}))=={'scene.xml','trace.npz','summary.json','contacts.json'}and all(
                    (path.parent/'defended'/name).is_file()and
                    digest(path.parent/'defended'/name)==expected
                    for name,expected in upright['source_sha256'].items()))
            result=pair_verdict(row,bank_sha=bank_sha,expected_geometry=expected_geometry,
                actions=actions,native_records=native,upright=upright,
                snapshot_matches_bank=bool(declared_source and (snapshot_hash==bank_sha or reusable)),
                scoring_matches_bank=scoring.exists()and json.loads(scoring.read_bytes())==report.get('scoring'),
                unchanged_bound_pair=reusable,current_implementation_verified=current_impl,
                upright_sources_verified=upright_bound,
                current_evidence_verified=pair_files_bound(path.parent,row,saved))
            result['reference_binding']=('exact_bank'if snapshot_hash==bank_sha else
                'unchanged_pair_from_verified_source'if reusable else 'not_bound')
            result['policy_or_graph_conclusions_transferred']=False
            result['evidence_path']=str(path)
            result['upright_audit_path']=str(linked[0][0])if len(linked)==1 else None
            records.append(result)
    families={f:sorted(n for n,a in actions.items()if a['family']==f and a['candidate_contract_pass'])
              for f in ATTACKS+DEFENCES}
    present={f:sorted(n for n,a in actions.items()if a['family']==f)for f in ATTACKS+DEFENCES}
    matrix={a:{d:summarize_cell(families[a],families[d],records)for d in DEFENCES}for a in ATTACKS}
    for a in ATTACKS:
        for d in DEFENCES:
            if present[a]and not families[a]:matrix[a][d]['status']='attack_family_not_diagnostic_eligible'
            elif present[d]and not families[d]:matrix[a][d]['status']='defence_family_not_diagnostic_eligible'
    starts=list(initial_guards)or report.get('initial_guards',[])
    if not starts and report.get('initial_guard'):starts=[report['initial_guard']]
    reachable=candidate_reachability(actions,starts)
    runtime_reached=set();chain_evidence=[];seen_chains=set()
    for root in roots:
        for path in root.rglob('result.json'):
            if path in seen_chains:continue
            seen_chains.add(path)
            try:row=json.loads(path.read_bytes())
            except (ValueError,OSError):continue
            if 'chain_feasibility_pass'not in row:continue
            native=path.parent/'native/summary.json'
            summary=json.loads(native.read_bytes())if native.exists()else {}
            audit_path=path.parent/'triangle_audit.json'
            audit=json.loads(audit_path.read_bytes())if audit_path.exists()else {}
            chain_snapshot=path.parent/'source_report_snapshot.json'
            actor=row.get('actor_slot')
            evidence_bound=(trace_files_bound(path.parent/'native',audit)and
                chain_snapshot.is_file()and digest(chain_snapshot)==bank_sha and
                summary.get('metadata',{}).get('source_report_sha256')==bank_sha and
                summary.get('executor_events',{}).get(actor)==row.get('executor_events'))
            reached=chain_reached(row,bank_sha=bank_sha,expected_geometry=expected_geometry,
                native_summary=summary,actions=actions,
                current_implementation_verified=(native_dependencies_current([summary])and
                                                 hashes_current(row.get('implementation_sha256'))),
                current_evidence_verified=evidence_bound)
            runtime_reached.update(reached)
            chain_evidence.append({'path':str(path),'bound_runtime_reached_guards':reached})
    for row in actions.values():
        row['entry_candidate_graph_reachable']=row['start_guard']in reachable
        row['entry_native_runtime_reachable']=row['start_guard']in runtime_reached
    nodes={}
    for node in report.get('guards',{}):
        outgoing=[a for a in actions.values()if a['start_guard']==node and a['candidate_contract_pass']]
        nodes[node]={'candidate_graph_reachable':node in reachable,
            'native_runtime_entry_reachable':node in runtime_reached,
            'outgoing_attacks':[a['name']for a in outgoing if a['kind']=='attack'],
            'outgoing_defences':[a['name']for a in outgoing if a['kind']=='defence'],
            'outgoing_explicit_transfers':[a['name']for a in outgoing if a['kind']=='transition'],
            'policy_callable_actions':[a['name']for a in outgoing if a['policy_callable_declared']],
            'waiting_is_not_a_defence_certificate':True}
    return {'schema_version':1,'status':'coverage_manifest_not_training_admission',
        'audit_implementation_sha256':digest(Path(__file__)),
        'source_bank':str(bank),'source_bank_sha256':bank_sha,'expected_geometry_sha256':expected_geometry,
        'actions':actions,'families':families,'present_family_variants':present,
        'missing_families':[f for f,n in present.items()if not n],
        'unavailable_families':[f for f,n in families.items()if not n],
        'attack_defence_matrix':matrix,'entry_guards':starts,'guard_choices':nodes,
        'all_evidence':records,'historical_or_unbound_evidence_count':sum(not r['valid_matched_baseline']for r in records),
        'bound_chain_evidence':chain_evidence,
        'native_runtime_reachability_unverified':not bool(runtime_reached),'admitted_to_training':False,
        'limitations':['A missing family never counts as covered.',
            'Same-name or same motion without exact bank and runtime geometry identity is not current evidence.',
            'A valid matched threat and complete recovery are required before a defensive success counts.',
            'Discrete successful offsets do not certify the interval between them.',
            'Graph reachability is structural only; current-state native handoff and reactive policy use require separate evidence.',
            'The 6x3 table is descriptive; not every defence is expected to defeat every attack.',
            'No finite test matrix establishes defence against arbitrary A2 continuous trajectories.']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank',required=True)
    parser.add_argument('--results',nargs='*',default=[])
    parser.add_argument('--geometry-sha256')
    parser.add_argument('--entry-guard',action='append',default=[])
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=Path(args.output).resolve()
    if output.exists():raise FileExistsError('Use a fresh manifest output; preserve previous evidence')
    result=generate(args.bank,args.results,expected_geometry=args.geometry_sha256,initial_guards=args.entry_guard)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'missing_families':result['missing_families'],
        'expected_geometry_sha256':result['expected_geometry_sha256'],
        'unbound_evidence':result['historical_or_unbound_evidence_count'],
        'admitted_to_training':False},ensure_ascii=False))


if __name__=='__main__':main()
