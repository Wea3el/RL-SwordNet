"""Read-only six-attack / three-defence cross-screen on a frozen bank.

Reference overlap is only a way to shortlist physical experiments. It never
proves that contact deflects a blade, that a policy can reach the entry guard,
or that a finite library can stop arbitrary continuous A2 actions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from .a1_coverage_audit import ATTACKS
from .a1_profile_bank import blade_identity
from .a1_recovery_geometry import exact_scene
from .assembly import ROOT, _finite_json
from .blade_geometry import blade_collision_geom_ids
from .duel_scene import classify_contact
from .fencing_rules import _point_cylinder_distances
from .limited_duel import build_limited_duel
from .motion_spline import C2JointSpline
from .triangle_task import TriangleGeometry


DEFENCES=('parry_left_upright_00','parry_right_upright_early','parry_high')


def signature(report,name):
    row=report['actions'][name]
    return {'times':row['knot_time_s'],'q':row['knot_qpos'],
            'start_q':report['guards'][row['start_guard']]['qpos_rad'],
            'end_q':report['guards'][row['end_guard']]['qpos_rad'],
            'kind':row['kind'],'target':row.get('target_region'),
            'direction':row.get('direction_intent_attacker_frame')}


def snapshot_baselines(bank,root):
    """Match local actual-baseline evidence, never whole-bank certificates."""
    records=[];geometry=bank['blade_geometry']['sha256']
    for p in Path(root).rglob('comparison.json'):
        try:
            c=json.loads(p.read_text());snap=json.loads((p.parent/'source_report_snapshot.json').read_text())
            a,d=c.get('attack_name'),c.get('defence_name')
            if a not in ATTACKS or d not in snap['actions']:continue
            summary=json.loads((p.parent/'baseline/summary.json').read_text())
            geometry_match=summary.get('metadata',{}).get('collision_geometry_sha256')==geometry
            source_hash_match=hashlib.sha256((p.parent/'source_report_snapshot.json').read_bytes()).hexdigest()==c.get('source_report_sha256')
            current_runtime=all((ROOT/n).is_file()and hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h
                                for n,h in summary.get('source_sha256',{}).items())
            attack_match=signature(bank,a)==signature(snap,a)
            for defence in DEFENCES:
                entry_match=(bank['guards'][bank['actions'][defence]['start_guard']]['qpos_rad']==
                             snap['guards'][snap['actions'][d]['start_guard']]['qpos_rad'])
                complete_required=max(bank['actions'][a]['knot_time_s'][-1],bank['actions'][defence]['knot_time_s'][-1])+.75+2.
                full=summary.get('completed_time_s',0)>=complete_required-1e-8 and summary.get('native_execution_pass')is True
                checks={'current_geometry':geometry_match,'snapshot_hash':source_hash_match,
                        'same_attack':attack_match,'same_entry':entry_match,
                        'same_scoring':snap.get('scoring')==bank.get('scoring'),
                        'current_runtime':current_runtime,'full_required_window':full}
                if all(checks.values()):
                    records.append({'attack':a,'defence_entry_for':defence,'comparison':str(p.resolve()),
                        'attacker_slot':c.get('attacker_slot'),'baseline_valid_threat':c.get('baseline_valid_threat')is True,
                        'baseline_first_hit_s':c.get('baseline_first_triangle_hit_s'),
                        'matching_checks':checks,
                        'defence_action_identical':defence in snap['actions'] and signature(bank,defence)==signature(snap,defence),
                        'source_defence':d,'source_successful_physical_defence':c.get('successful_physical_defence')is True,
                        'not_a_policy_or_entry_reachability_certificate':True})
        except (KeyError,ValueError,TypeError,FileNotFoundError):
            continue
    return records


def state(scene,geometry,qa,qd):
    model,data=scene.model,scene.data
    data.qpos[scene.agents['a1']['qpos_ids']]=qa
    data.qpos[scene.agents['a2']['qpos_ids']]=qd
    data.qvel[:]=0.;mujoco.mj_forward(model,data)
    contacts=[]
    for c in data.contact:
        if c.dist>=-1e-8:continue
        names=[model.geom(int(g)).name for g in (c.geom1,c.geom2)]
        normal=c.frame[:3].copy()
        if not names[0].startswith('a1_'):normal*=-1
        contacts.append({'category':classify_contact(model,c,rules_version='whole_blade_first_touch_v2'),
                         'geoms':names,'depth_m':float(-c.dist),'point_world_m':c.pos.tolist(),
                         'force_on_attacker_normal_world':(-normal).tolist()})
    gaps={}
    for owner,other in (('a1','a2'),('a2','a1')):
        ids=np.array(blade_collision_geom_ids(model,owner+'_'))
        direction=data.geom_xmat[ids].reshape(-1,3,3)[:,:,2]
        half=model.geom_size[ids,1,None];center=data.geom_xpos[ids]
        gaps[owner]={}
        for r in geometry.regions:
            distances,_=_point_cylinder_distances(np.array(geometry.world_center(r.name,other)),
                center-half*direction,center+half*direction,model.geom_size[ids,0])
            gaps[owner][r.name]=float(min(distances)-r.radius)
    return {'contacts':contacts,'gaps':gaps}


def scan(scene,geometry,attack,defence,period=.035):
    duration=max(attack.duration_s,defence.duration_s)+2.
    times=np.unique(np.r_[np.arange(0.,duration,period),attack.knot_time_s,defence.knot_time_s,duration])
    base={'first_entries':{},'first_contact':None,'illegal_contacts':{},'minimum_gaps':{'left':1e6,'right':1e6,'upper':1e6}}
    defended={'first_entries':{},'first_contact':None,'illegal_contacts':{},'maximum_overlap_m':0.,
              'pre_contact_illegal':False,'minimum_gaps':{'left':1e6,'right':1e6,'upper':1e6}}
    initial=None
    for t in times:
        for record,q in ((base,defence.knot_qpos[0]),(defended,defence.sample(t))):
            s=state(scene,geometry,attack.sample(t),q)
            if initial is None:initial=s
            for region,gap in s['gaps']['a1'].items():
                record['minimum_gaps'][region]=min(record['minimum_gaps'][region],gap)
                if gap<0:record['first_entries'].setdefault('a1_'+region,float(t))
            for region,gap in s['gaps']['a2'].items():
                if gap<0:record['first_entries'].setdefault('a2_'+region,float(t))
            for c in s['contacts']:
                if c['category'].startswith('illegal_'):
                    record['illegal_contacts'].setdefault('|'.join(c['geoms']),dict(c,time_s=float(t)))
                    if record is defended and record['first_contact']is None:record['pre_contact_illegal']=True
                if c['category']in ('cross_blade_blade','cross_blade_guard')and record['first_contact']is None:
                    record['first_contact']=dict(c,time_s=float(t))
                if record is defended:record['maximum_overlap_m']=max(record['maximum_overlap_m'],c['depth_m'])
    for r in (base,defended):r['illegal_contacts']=list(r['illegal_contacts'].values())
    return {'baseline_reference':base,'defended_reference':defended,
            'initial_pair_clear':not initial['contacts']and all(g>0 for s in initial['gaps'].values()for g in s.values()),
            'period_s':period,'sample_count':len(times),'full_reference_window_local_s':duration,
            'future_reference_used_only_for_offline_authoring':True}


def run(bank_path,scene_path,results_root,output):
    bank_path,scene_path,output=Path(bank_path).resolve(),Path(scene_path).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError('Use a fresh diagnostic output')
    bank_raw=bank_path.read_bytes();bank=json.loads(bank_raw)
    geo=TriangleGeometry.from_mapping({k:bank['scoring'][k]for k in ('regions','base_distance')})
    # Saved XML is a rounded text snapshot; the reference screen uses the
    # exact current constructor and matches its geometry identity to the bank.
    # The supplied saved scene is provenance, not an alternate physical model.
    scene=build_limited_duel(geo.base_distance,
        initial_qpos=bank['guards'][bank['actions'][DEFENCES[0]]['start_guard']]['qpos_rad'])
    if blade_identity(scene.model,'a1_')['sha256']!=bank['blade_geometry']['sha256']:
        raise ValueError('Frozen source model geometry mismatch')
    evidence=snapshot_baselines(bank,results_root)
    output.parent.mkdir(parents=True,exist_ok=True)
    report={'status':'read_only_cross_family_reference_shortlist','bank':str(bank_path),
            'bank_sha256':hashlib.sha256(bank_raw).hexdigest(),'collision_geometry_sha256':bank['blade_geometry']['sha256'],
            'scene_sha256':hashlib.sha256(scene_path.read_bytes()).hexdigest(),
            'reference_model':'Exact current constructor, strict bank geometry hash; not rounded XML reload',
            'cells':[],'matched_actual_baselines':evidence,'admitted_to_training':False,
            'limitations':['Reference crossings are not successful physical deflections.',
                'A held-entry baseline may already obstruct a different attack; that is not active-defence evidence.',
                'Source guard availability is not runtime reachability from the preceding attack.',
                'Only offset0 is screened here; no hidden timing optimization or mirrored success inference.',
                'Every proposed alternative still needs matched full-window native tests in both slots.']}
    for an in ATTACKS:
        a=bank['actions'][an];ac=C2JointSpline(a['knot_time_s'],a['knot_qpos'])
        for dn in DEFENCES:
            d=bank['actions'][dn];dc=C2JointSpline(d['knot_time_s'],d['knot_qpos'])
            row={'attack':an,'defence':dn,'entry_guard':d['start_guard'],'offset_s':0.}
            audit=scan(scene,geo,ac,dc);row.update(audit)
            baselines=[e for e in evidence if e['attack']==an and e['defence_entry_for']==dn]
            row['actual_baseline_records']=baselines
            actual_threat=any(e['baseline_valid_threat']for e in baselines)
            reference_first=audit['baseline_reference']['first_entries'].get('a1_'+a['target_region'])
            baseline_contact=audit['baseline_reference']['first_contact']
            baseline_clean=bool(not audit['baseline_reference']['illegal_contacts']and audit['initial_pair_clear']
                and (baseline_contact is None or (reference_first is not None and baseline_contact['time_s']>reference_first)))
            first=audit['defended_reference']['first_contact']
            # This is deliberately a shortlist, not a certificate; compare the
            # event against a same-entry actual hit when one is available.
            hit=min([e['baseline_first_hit_s']-.75 for e in baselines if e['baseline_valid_threat']and e['baseline_first_hit_s']is not None],default=reference_first)
            contact_precedes=bool(first and hit is not None and first['time_s']<hit)
            threatened=bool(actual_threat if baselines else (baseline_clean and reference_first is not None))
            row['baseline_threat_level']=('matched_native'if actual_threat else
                ('matched_native_no_threat_or_invalid'if baselines else
                 ('reference_only'if threatened else 'missing_or_invalid')))
            row['candidate_for_matched_native']=bool(threatened and audit['initial_pair_clear']and
                contact_precedes and not audit['defended_reference']['pre_contact_illegal'])
            exact_pairs=[e for e in baselines if e['source_defence']==dn and e['defence_action_identical']]
            row['known_exact_pair_outcomes']=[{'comparison':e['comparison'],
                'attacker_slot':e['attacker_slot'],
                'successful_physical_defence':e['source_successful_physical_defence']}for e in exact_pairs]
            row['new_unverified_alternative']=bool(row['candidate_for_matched_native'] and not exact_pairs)
            row['existing_nominal_failure_not_repaired_by_screen']=bool(exact_pairs and not any(
                e['source_successful_physical_defence']for e in exact_pairs))
            row['reference_contact_before_same_entry_hit']=contact_precedes
            row['successful_defence_certified']=False
            report['cells'].append(row)
            print(json.dumps({k:row[k]for k in ('attack','defence','baseline_threat_level','candidate_for_matched_native')}),flush=True)
            output.write_text(json.dumps(_finite_json(report),indent=2)+'\n')
    report['bank_unchanged']=bank_path.read_bytes()==bank_raw
    output.write_text(json.dumps(_finite_json(report),indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank',required=True);p.add_argument('--scene',required=True)
    p.add_argument('--results-root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.bank,a.scene,a.results_root,a.output)
