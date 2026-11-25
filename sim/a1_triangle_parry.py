"""Matched full-window native parry experiments for the triangle proposal.

Do not turn a pose label, first weapon contact or defender's early score into
immunity. All three regions remain observable through recovery. This is an
open-loop feasibility experiment, not a policy with access to a future script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .assembly import ROOT, _finite_json
from .a1_triangle_native import audit_recorded_triangle
from .constrained_trial import ScheduledReference, hold_reference, run_constrained_trial
from .motion_spline import C2JointSpline


DEFAULT=ROOT/'config/a1_triangle_parry_v1.json'


def parry_program(ready, pose, spec, horizon):
    times=[0., spec['approach_s'], spec['release_s']-spec['start_s'],
           spec['release_s']-spec['start_s']+spec['return_s']]
    if not all(np.isfinite(times)) or not 0 < times[1] < times[2] < times[3]:
        raise ValueError('Parry needs ordered approach, hold and recovery')
    curve=C2JointSpline(times,[ready,pose,pose,ready])
    if max(curve.analytic_extrema()['peak_speed_rad_s']) > 2.+1e-9:
        raise ValueError('Candidate exceeds the common 2 rad/s reference ceiling')
    program=ScheduledReference(curve,spec['start_s'])
    if program.end_s+2. > horizon+1e-9:
        raise ValueError('Full recovery and two-second settle tail required')
    return program


def triangle_defence_gate(baseline, baseline_audit, defended, defended_audit,
                          contacts, *, matching_checks, attacker='a1'):
    """Pure evidence gate. Delayed/off-region entries remain defence failures."""
    if attacker not in ('a1', 'a2'):
        raise ValueError('Attacker must be a physical slot')
    matching_checks=dict(matching_checks)
    for label,summary,audit in [('baseline',baseline,baseline_audit),
                                ('defended',defended,defended_audit)]:
        # Historical records without the metadata field were all a1 attacks.
        matching_checks[label+'_attacker_identity']=(
            summary.get('metadata',{}).get('attacker_slot','a1')==attacker
            and audit.get('attacker_slot',attacker)==attacker)
    base_hits=[h for h in baseline_audit['all_entries'] if h['agent']==attacker]
    late_hits=[h for h in defended_audit['all_entries'] if h['agent']==attacker]
    defender_hits=[h for h in defended_audit['all_entries'] if h['agent']!=attacker]
    numerical_clean=all(s.get('native_execution_pass') and s.get('source_unchanged')
                        and not s.get('failure') for s in (baseline,defended))
    geometric_clean=all(not a.get('initial_overlap') and not a.get('sweep_failures')
                        for a in (baseline_audit,defended_audit))
    valid_threat=bool(base_hits and baseline_audit['intended_region_first'])
    hit_time=min((h['time_s'] for h in base_hits),default=None)
    first_hit=min(base_hits,key=lambda h:h['time_s']) if base_hits else {}
    cutoff=first_hit.get('step_t0_s',hit_time)
    impulse=peak=0.; first_contact=None; contact_types=set()
    straddling=[]
    for interval in contacts:
        if hit_time is None:
            break
        # Endpoint force cannot be assigned retroactively to the portion of
        # its sample interval before a hit. Exclude the entire cutoff frame.
        if interval['interval_end_s'] > cutoff:
            if interval['interval_start_s'] < cutoff:
                straddling.append([interval['interval_start_s'],interval['interval_end_s']])
            continue
        weight=max(0.,interval['interval_end_s']-interval['interval_start_s'])
        for c in interval['contacts']:
            if c['category'] not in ('cross_blade_blade','cross_blade_guard'):
                continue
            names=[c.get('geom1',''),c.get('geom2','')]
            if {n[:2] for n in names}!={'a1','a2'}:
                continue
            force=max(0.,float(c['normal_force_n']))
            impulse+=force*weight;peak=max(peak,force)
            if force>0:
                first_contact=interval['interval_end_s'] if first_contact is None else min(first_contact,interval['interval_end_s'])
                contact_types.add(c['category'])
    cfg=baseline['runtime_config']
    meaningful=peak>=cfg['minimum_blade_force_n'] and impulse>=cfg['minimum_blade_impulse_ns']
    matched=bool(matching_checks and all(matching_checks.values()))
    if not numerical_clean or not geometric_clean:
        status='invalid'
    elif not matched:
        status='unmatched'
    elif not valid_threat:
        status='no_threat'
    elif late_hits or not meaningful:
        status='failed'
    elif defender_hits:
        status='defence_with_counter_candidate'
    else:
        status='qualified'
    return {'status':status,'successful_physical_defence':status=='qualified',
        'matching_checks':matching_checks,'baseline_valid_threat':valid_threat,
        'baseline_first_triangle_hit_s':hit_time,'first_pre_hit_weapon_contact_s':first_contact,
        'pre_hit_contact_cutoff_s':cutoff,'excluded_straddling_intervals':straddling,
        'pre_hit_weapon_impulse_ns':impulse,'pre_hit_peak_weapon_force_n':peak,
        'pre_hit_contact_types':sorted(contact_types),'meaningful_pre_hit_contact':meaningful,
        'attacker_entries_entire_window':late_hits,'defender_entries_entire_window':defender_hits,
        'all_regions_checked':True,'numerical_and_full_execution_clean':bool(numerical_clean),
        'geometric_evidence_clean':bool(geometric_clean),'admitted_to_training':False,
        'scope':'This exact open-loop attack, entry, timing and full recovery only; not general defence or policy reaction'}


def compare(baseline_folder, defended_folder, scoring, target='upper',attacker='a1'):
    baseline_folder,defended_folder=Path(baseline_folder),Path(defended_folder)
    baseline=json.loads((baseline_folder/'summary.json').read_text())
    defended=json.loads((defended_folder/'summary.json').read_text())
    checks={n:baseline[n]==defended[n] for n in
            ('runtime_config','joint_constraints','source_sha256','requested_duration_s','model_source_sha256')}
    checks['attack_start_s']=baseline['program_start_s'][attacker]==defended['program_start_s'][attacker]
    with np.load(baseline_folder/'references.npz',allow_pickle=False) as b, np.load(defended_folder/'references.npz',allow_pickle=False) as d:
        checks['attack_reference']=all(np.array_equal(b[attacker+'_'+k],d[attacker+'_'+k]) for k in ('knot_time_s','knot_qpos','coefficients'))
    dt=baseline['runtime_config']['physics_timestep_s']
    for label,folder in [('baseline',baseline_folder),('defended',defended_folder)]:
        with np.load(folder/'trace.npz',allow_pickle=False) as trace:
            time=trace['time_s']; expected=baseline['requested_duration_s']
            checks[label+'_complete_native_trace']=bool(abs(time[0])<1e-9 and abs(time[-1]-expected)<1e-8 and np.allclose(np.diff(time),dt,rtol=0,atol=1e-9))
            if label=='baseline':
                initial_q=trace['qpos'][0].copy();initial_dq=trace['qvel'][0].copy()
            else:
                checks['same_initial_qpos']=np.array_equal(initial_q,trace['qpos'][0])
                checks['same_initial_qvel']=np.array_equal(initial_dq,trace['qvel'][0])
    ba=audit_recorded_triangle(baseline_folder,scoring,target,attacker)
    da=audit_recorded_triangle(defended_folder,scoring,target,attacker)
    contacts=json.loads((defended_folder/'contacts.json').read_text())
    gate=triangle_defence_gate(baseline,ba,defended,da,contacts,matching_checks=checks,attacker=attacker)
    return dict(gate,baseline_triangle_audit=ba,defended_triangle_audit=da,
                attacker_slot=attacker,
                native_contact_source_sha256=hashlib.sha256(
                    (defended_folder/'contacts.json').read_bytes()).hexdigest())


def strict_review(experiment, output, attacker='a1'):
    """Re-gate immutable saved entries/contacts without rerunning physics.

    Retain original comparisons; a new directory records the current gate and
    verifies that all summary/trace/scene hashes still match their audits.
    New comparisons also freeze contact hashes. Legacy comparisons did not:
    label that limitation rather than retroactively inventing provenance.
    """
    experiment,output=Path(experiment).resolve(),Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve earlier reviews; use a new output path')
    output.mkdir(parents=True)
    result={}
    for path in sorted(experiment.glob('*_comparison.json')):
        name=path.name.removesuffix('_comparison.json')
        previous=json.loads(path.read_text())
        summaries=[]; checks=dict(previous['matching_checks'])
        for label,folder in [('baseline',experiment/'baseline'),('defended',experiment/name)]:
            summaries.append(json.loads((folder/'summary.json').read_text()))
            audit=previous[label+'_triangle_audit']
            checks[label+'_recorded_evidence_unchanged']=all(
                hashlib.sha256((folder/n).read_bytes()).hexdigest()==h
                for n,h in audit['native_trace_source_sha256'].items())
        contact_path=experiment/name/'contacts.json'
        contact_hash=hashlib.sha256(contact_path.read_bytes()).hexdigest()
        original_contact_hash=previous.get('native_contact_source_sha256')
        if original_contact_hash is not None:
            checks['recorded_contacts_unchanged']=contact_hash==original_contact_hash
        contacts=json.loads(contact_path.read_text())
        gate=triangle_defence_gate(*[summaries[0],previous['baseline_triangle_audit'],summaries[1],previous['defended_triangle_audit'],contacts],
                                  matching_checks=checks,attacker=attacker)
        gate.update(input_comparison_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    gate_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    gate_revision='complete_pre_hit_frames_only_v2',
                    previous_status=previous['status'],
                    current_contact_snapshot_sha256=contact_hash,
                    contact_hash_provenance=('verified_against_original_comparison'
                        if original_contact_hash is not None else
                        'legacy_comparison_did_not_freeze_contact_hash'),
                    attacker_slot=attacker)
        (output/(name+'.json')).write_text(json.dumps(_finite_json(gate),ensure_ascii=False,indent=2)+'\n')
        result[name]=gate
    (output/'summary.json').write_text(json.dumps(_finite_json(result),ensure_ascii=False,indent=2)+'\n')
    return result


def run(output, cases, config=DEFAULT, attacker='a1', start_times=None):
    output,config=Path(output).resolve(),Path(config).resolve()
    cfg=json.loads(config.read_text());source=ROOT/cfg['source']
    if attacker not in ('a1','a2'):
        raise ValueError('Attacker must be a physical slot')
    defender='a2' if attacker=='a1' else 'a1'
    report=json.loads((source/'report.json').read_text())
    score=json.loads((ROOT/cfg['scoring']).read_text())
    if cfg['admitted_to_training'] or score['runtime_activation_allowed']:
        raise ValueError('Candidate-only physical tests required')
    if any(name not in cfg['cases'] for name in cases):
        raise ValueError('Unknown parry case')
    probes=[(name,dict(cfg['cases'][name])) for name in cases]
    if start_times is not None:
        if len(cases)!=1 or not 1<=len(start_times)<=5 or any(not np.isfinite(t) or t<0 for t in start_times):
            raise ValueError('One case and one to five finite start-time probes required')
        probes=[]
        for time_s in start_times:
            suffix=format(float(time_s),'.6f').rstrip('0').rstrip('.').replace('.','p')
            probes.append((cases[0]+'_start_'+suffix,dict(cfg['cases'][cases[0]],start_s=float(time_s))))
        if len({n for n,_ in probes})!=len(probes):
            raise ValueError('Start times collapse to duplicate output names')
    attack=report['actions'][cfg['attack']]
    attack_program=ScheduledReference(C2JointSpline(np.asarray(attack['knot_time_s'])*cfg['attack_time_scale'],attack['knot_qpos']),cfg['attack_start_s'])
    ready=np.asarray(report['guards']['center']['qpos_rad'])
    programs={attacker:attack_program,defender:hold_reference(ready,cfg['attack_start_s'],attack_program.curve.duration_s)}
    metadata={'parry_config_sha256':hashlib.sha256(config.read_bytes()).hexdigest(),
              'source_report_sha256':hashlib.sha256((source/'report.json').read_bytes()).hexdigest(),
              'not_policy_reaction':True,'all_times_are_offline_probe_settings':True}
    if attacker!='a1':
        metadata['attacker_slot']=attacker
    output.mkdir(parents=True,exist_ok=True)
    baseline=output/'baseline'
    if not baseline.exists():
        run_constrained_trial(programs,baseline,duration_s=cfg['duration_s'],metadata=dict(metadata,case='baseline'),progress=print)
    else:
        old=json.loads((baseline/'summary.json').read_text())
        if old.get('metadata')!=dict(metadata,case='baseline'):
            raise ValueError('Existing baseline has a different identity')
    results={}
    for name,spec in probes:
        folder=output/name
        if folder.exists():
            raise FileExistsError('Existing case preserved; select an unrun case or a new output directory')
        q=cfg['geometry_derived_poses'][spec['pose']] if 'pose' in spec else report['guards'][spec['guard']]['qpos_rad']
        program=parry_program(ready,q,spec,cfg['duration_s'])
        run_constrained_trial({attacker:attack_program,defender:program},folder,duration_s=cfg['duration_s'],metadata=dict(metadata,case=name,defence_probe=spec),progress=print)
        result=compare(baseline,folder,score,attacker=attacker)
        results[name]=result
        (output/(name+'_comparison.json')).write_text(json.dumps(_finite_json(result),ensure_ascii=False,indent=2)+'\n')
        print(name,result['status'],flush=True)
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--config',default=str(DEFAULT));p.add_argument('--cases',nargs='+')
    p.add_argument('--attacker-slot',choices=('a1','a2'),default='a1')
    p.add_argument('--start-times',type=float,nargs='+',help='Bounded timing probes for a single named case')
    p.add_argument('--review-to',help='Only re-check existing evidence into this new directory')
    a=p.parse_args()
    if a.review_to:
        strict_review(a.output,a.review_to,a.attacker_slot)
    elif not a.cases:
        p.error('--cases is required unless --review-to is given')
    else:
        run(a.output,a.cases,a.config,a.attacker_slot,a.start_times)


if __name__=='__main__':
    main()
