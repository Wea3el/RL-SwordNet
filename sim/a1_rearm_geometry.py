"""Explicit right-parry-hold to dynamic-left-parry-entry candidates.

The final hold of a completed parry is not automatically a blocking stance for
a fresh attack.  This file records that negative result and screens a genuine
return to the old dynamic parry's exact input guard. No native integration.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from .a1_dynamics_screen import screen_curve
from .a1_guard_defence_screen import contact_state
from .a1_recovery_geometry import exact_scene, screen_path, static_screen, write
from .assembly import ROOT
from .joint_constraints import load_joint_constraints, model_joint_ranges_rad
from .limited_model import build_limited_floor
from .motion_spline import C2JointSpline
from .triangle_task import TriangleGeometry


def curve_from(row):
    return C2JointSpline(row['knot_time_s'], row['knot_qpos'])


def collect(scene, geometry, attacker, defender, duration, samples, both_slots=True):
    results = {}
    for slot in (('a1','a2') if both_slots else ('a1',)):
        records, entries = {}, {}
        blade_gap = .2
        for t in np.linspace(0., duration, samples):
            state = contact_state(scene, geometry, attacker(t), defender(t), slot)
            blade_gap = min(blade_gap, state['minimum_blade_gap_m'])
            for item in state['contacts']:
                key = '|'.join(item['geoms'])
                row = records.setdefault(key, dict(item, first_local_s=float(t),
                                                   last_local_s=float(t), max_depth_m=0.))
                row['last_local_s'] = float(t)
                row['max_depth_m'] = max(row['max_depth_m'],item['depth_m'])
            for owner, gaps in state['target_gaps_m'].items():
                for name, gap in gaps.items():
                    if gap < 0:
                        entries.setdefault(owner+':'+name, float(t))
        results[slot] = {'all_reference_contacts':list(records.values()),
            'first_reference_target_entries':entries, 'minimum_blade_gap_m':blade_gap,
            'clear_pass':not records and not entries}
    return {'by_attacker_slot':results, 'samples_per_slot':samples,
            'all_clear':all(row['clear_pass']for row in results.values()),
            'scope':'Sampled two moving references, not physical contact response.'}


def hold_diagnostic(output, scene, geometry, source):
    """Record why fixed left_hold cannot inherit the dynamic-parry certificate."""
    pair_dir=ROOT/'outputs/a1_short_defence/native16_switch_right_to_left'
    old=json.loads((pair_dir/'source_report_snapshot.json').read_text())
    native=json.loads((pair_dir/'defended/summary.json').read_text())
    attack=curve_from(old['actions']['cut_down_left_to_right'])
    hold=np.asarray(old['guards']['left_parry_hold']['qpos_rad'])
    samples=collect(scene,geometry,attack.sample,lambda t:hold,attack.duration_s,1001,False)
    contacts=json.loads((pair_dir/'defended/contacts.json').read_text())
    illegal=[dict(frame_time_s=frame['time_s'],**c)for frame in contacts for c in frame['contacts']
             if c['category'].startswith('illegal_')]
    result={'status':'negative_result_fixed_hold_is_not_dynamic_parry',
        'source_report':str(pair_dir/'source_report_snapshot.json'),
        'source_sha256':hashlib.sha256((pair_dir/'source_report_snapshot.json').read_bytes()).hexdigest(),
        'native_summary_sha256':hashlib.sha256((pair_dir/'defended/summary.json').read_bytes()).hexdigest(),
        'guard':'left_parry_hold','guard_qpos_rad':hold.tolist(),
        'incoming_attack':'cut_down_left_to_right','reference_clock':'Seconds from start of negative down attack, not whole prepared sequence.',
        'fixed_hold_reference_screen':samples,
        'actual_prepared_pair_failure':{'completed_global_s':native['completed_time_s'],
            'negative_attack_started_global_s':5.6792586762920205,
            'first_illegal_contacts':illegal[:3], 'failure':native['failure'],
            'all_target_entries':native['all_target_entries']},
        'interpretation':[
            'Fixed left_hold exposes arm_proxy_3 before the incoming blade reaches the guarding blade.',
            'Original dynamic parry intercepts during entry before reaching this final hold; its causal path must be preserved.',
            'Reference contacts after an earlier sword interception do not predict native deflection and are not reused as physical verdicts.',
            'The hold can remain a post-parry graph node, but no fresh-attack blocking or arbitrary waiting safety is certified.',
            'Re-arm to exact left_guard_positive before invoking the full timed dynamic parry; that combined transition still requires new native tests.'],
        'thresholds_or_old_sources_changed':False,'training_admission':False}
    write(output/'fixed_left_hold_failure_diagnostic.json',result)


def build(output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve prior candidates and failures')
    output.mkdir(parents=True)
    source=ROOT/'outputs/a1_short_defence/switch18_right_to_left_withdraw/report.json'
    source_bytes=source.read_bytes();base=json.loads(source_bytes)
    scene_path=ROOT/'outputs/a1_recovery_geometry/withdraw01/scene.xml'
    scene=exact_scene(scene_path)
    geometry=TriangleGeometry.from_mapping({k:base['scoring'][k]for k in ('base_distance','regions')})
    model=build_limited_floor().model
    hold_diagnostic(output,scene,geometry,base)
    qa=np.asarray(base['guards']['right_parry_hold']['qpos_rad'])
    qb=np.asarray(base['guards']['left_guard_positive']['qpos_rad'])
    attack=curve_from(base['actions']['prepared_attack'])
    # Exact start of the final, negative-down component, in prepared-attack time.
    prep_end=float(base['actions']['recover_positive_high_to_low']['knot_time_s'][-1]+.1+
                   base['actions']['switch_positive_to_negative']['knot_time_s'][-1]+.1)
    transfer_start=.1
    safety_hold=.05
    available=prep_end-transfer_start-safety_hold
    bounds=model_joint_ranges_rad(load_joint_constraints())
    rows=[]
    for duration in (4.,4.25,4.5,4.75,5.):
        curve=C2JointSpline([0.,duration],[qa,qb])
        stat=static_screen(model,curve)
        dyn=screen_curve(model,curve,.025)
        sync=collect(scene,geometry,attack.sample,
                     lambda t,c=curve:c.sample(max(0.,t-transfer_start)),prep_end,401)
        valid=bool(duration<=available and stat['peak_static_cap_ratio']<1 and
            dyn['ideal_demand_within_caps_at_samples'] and sync['all_clear'] and
            np.all(curve.bezier_controls>=bounds[:,0])and np.all(curve.bezier_controls<=bounds[:,1]))
        rows.append({'duration_s':duration,'static':stat,'ideal_free_space_dynamics':dyn,
                     'sync_preparation':sync,'candidate_pass':valid})
        write(output/'attempts.json',rows)
        print(json.dumps({'duration_s':duration,'static':stat['peak_static_cap_ratio'],
              'ideal':dyn['worst_utilization'],'sync_clear':sync['all_clear'],'candidate':valid}),flush=True)
    selected=next((row for row in rows if row['duration_s']==4.5 and row['candidate_pass']),None)
    if selected is None:selected=next((row for row in rows if row['candidate_pass']),None)
    report=copy.deepcopy(base)
    meta={'source_report':str(source),'source_report_sha256':hashlib.sha256(source_bytes).hexdigest(),
          'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'scene_sha256':hashlib.sha256(scene_path.read_bytes()).hexdigest(),
          'attacker_preparation_local_start_s':0.,'defender_transfer_local_start_s':transfer_start,
          'negative_attack_local_start_s':prep_end,'global_preparation_start_s':.75,
          'global_defender_transfer_start_s':.85,'global_negative_attack_start_s':.75+prep_end,
          'maximum_transfer_duration_with_50ms_margin_s':available,
          'native_dynamics_tested':False,'training_admission':False,
          'purpose':'Return to the dynamic parry input; do not replace it with its final hold.'}
    report.update(status='right_hold_to_left_parry_entry_geometry_candidate',
                  native_dynamics_tested=False,admitted_to_training=False,reentry_repair=meta)
    if selected:
        curve=C2JointSpline([0.,selected['duration_s']],[qa,qb])
        sync=collect(scene,geometry,attack.sample,
            lambda t:curve.sample(max(0.,t-transfer_start)),prep_end,1201)
        fixed={}
        for name in ('center','right_parry_hold','left_parry_hold',
                     'left_high_negative_robust','right_low_negative_robust','left_guard_positive'):
            fixed[name]=screen_path(scene,geometry,curve,np.asarray(base['guards'][name]['qpos_rad']),
                                    samples=401,both_slots=True)
        dyn=screen_curve(model,curve,.01)
        # This artifact is explicitly a synchronized preparation experiment,
        # not an unconditional transfer callable against every held stance.
        # Keep all failed fixed-opponent controls; do not use them as proof of
        # arbitrary-time availability or hide them behind a global pass flag.
        passed=bool(sync['all_clear'] and selected['static']['peak_static_cap_ratio']<1
                    and dyn['ideal_demand_within_caps_at_samples'])
        action={'kind':'transition','family':'explicit_dynamic_parry_reentry',
            'start_guard':'right_parry_hold','end_guard':'left_guard_positive',
            'knot_time_s':curve.knot_time_s.tolist(),'knot_qpos':curve.knot_qpos.tolist(),
            'candidate_pass':passed,'sampled_reference_pass':passed,
            'native_dynamics_tested':False,'admitted_to_training':False,
            'sync_preparation_screen':sync,'fixed_guard_screens':fixed,
            'static':selected['static'],'ideal_free_space_dynamics':dyn,
            'unwrapped_joint_travel_deg':np.rad2deg(abs(qb-qa)).tolist(),
            'candidate_pass_context':'Only the declared synchronous withdrawal-first opponent preparation; not arbitrary held opponents.',
            'general_fixed_opponent_pass':all(s['all_clear']for s in fixed.values()),
            'scope':'Exact synchronized entry transfer only; full dynamic parry and new whole-window pair remain unverified.'}
        report['actions']['right_hold_to_left_entry']=action
        meta.update(selected_duration_s=curve.duration_s,
            at_entry_before_attack_margin_s=prep_end-transfer_start-curve.duration_s,
            candidate_pass=passed,
            unconditional_skill_admission=False,
            failed_fixed_opponent_controls=[name for name,s in fixed.items()if not s['all_clear']],
            transfer_entry_to_full_parry='After measured settling, invoke original parry_small_J0_00_0 with original timing; no reset or hidden hold substitution.')
        np.savez_compressed(output/'right_hold_to_left_entry.npz',
            time_s=np.linspace(0,curve.duration_s,401),
            qpos=curve.sample(np.linspace(0,curve.duration_s,401)),
            knot_time_s=curve.knot_time_s,knot_qpos=curve.knot_qpos)
    else:meta['candidate_pass']=False
    write(output/'report.json',report)
    (output/'builder_snapshot.py').write_bytes(Path(__file__).read_bytes())
    shutil.copyfile(scene_path,output/'scene.xml')
    print(json.dumps(meta,indent=2),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    build(p.parse_args().output)
