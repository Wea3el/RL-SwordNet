"""Package explicitly qualified native candidates without inheriting admission."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import mujoco

from .assembly import ROOT, _finite_json
from .a1_triangle_reactive import PUBLIC_FIELDS, corridor_observation
from .feedback_policy import GraphAction, SkillGraphExecutor
from .motion_spline import C2JointSpline


def _references(folder,slot,summary):
    """Bind historical command files to the already hash-locked native trace.

    Old geometry audits did not hash references.npz. Rebuilding its exact
    coefficients and comparing EVERY native desired-reference sample prevents
    replacing a successful command with an unrelated curve sharing endpoints.
    The recorded desired reference is not the zero-order-held control target.
    """
    folder=Path(folder)
    with np.load(folder/'references.npz',allow_pickle=False) as ref:
        curve=C2JointSpline(ref[slot+'_knot_time_s'],ref[slot+'_knot_qpos'])
        if not np.array_equal(ref[slot+'_coefficients'],curve.coefficients):
            raise ValueError('Saved reference coefficients do not exactly rebuild from knots')
    start=summary.get('program_start_s',{}).get(slot)
    if isinstance(start,bool) or start is None or not np.isfinite(start) or start<0:
        raise ValueError('Recorded ScheduledReference start is required')
    model=mujoco.MjModel.from_xml_path(str(folder/'scene.xml'))
    try:
        ids=[model.joint(slot+'_J'+str(i)).id for i in range(7)]
    except KeyError as exc:
        raise ValueError('Recorded model lacks the named slot joints') from exc
    qids=model.jnt_qposadr[ids]
    with np.load(folder/'trace.npz',allow_pickle=False) as trace:
        if 'desired_reference_qpos' not in trace:
            raise ValueError('Historical trace lacks exact desired-reference evidence')
        times=trace['time_s'];desired=trace['desired_reference_qpos']
        duration=summary.get('requested_duration_s')
        dt=summary.get('runtime_config',{}).get('physics_timestep_s')
        if (times.ndim!=1 or len(times)<2 or not np.isfinite(times).all()
                or duration is None or dt is None or not np.isfinite([duration,dt]).all()
                or duration<=0 or dt<=0 or abs(times[0])>1e-9
                or abs(times[-1]-duration)>1e-8
                or not np.allclose(np.diff(times),dt,rtol=0,atol=1e-9)
                or desired.shape!=(len(times),model.nq) or not np.isfinite(desired).all()):
            raise ValueError('A complete finite native desired-reference trace is required')
        expected=curve.sample(times-float(start))
        if not np.allclose(desired[:,qids],expected,rtol=0,atol=2e-12):
            raise ValueError('Command reference does not match the hash-locked native trace')
    return {'time_s':curve.knot_time_s.tolist(),'qpos':curve.knot_qpos.tolist()}


def _gate_evidence(gate,baseline,defended,expected_scoring):
    """Validate source identity before extracting a locally expressed skill."""
    attacker=gate.get('attacker_slot')
    if attacker not in ('a1','a2'):
        raise ValueError('Gate must identify its physical attacker slot')
    defender='a2' if attacker=='a1' else 'a1'
    checks=gate.get('matching_checks',{})
    if not checks or not all(checks.values()):
        raise ValueError('Mismatched source evidence')
    summaries={}
    for label,folder in (('baseline',Path(baseline)),('defended',Path(defended))):
        summary=json.loads((folder/'summary.json').read_text())
        if (summary.get('native_execution_pass') is not True
                or summary.get('status')!='completed' or summary.get('failure')
                or summary.get('source_unchanged') is not True):
            raise ValueError('Incomplete or invalid native source')
        # Pre-slot-metadata native records were explicitly the a1-attacker
        # version; do not infer a mirrored role from a filename or skill name.
        if summary.get('metadata',{}).get('attacker_slot','a1')!=attacker:
            raise ValueError('Native attacker identity disagrees with the gate')
        audit=gate[label+'_triangle_audit']
        if audit.get('attacker_slot')!=attacker:
            raise ValueError('Geometry audit attacker identity disagrees with the gate')
        if audit.get('scoring_snapshot')!=expected_scoring:
            raise ValueError('Baseline/defender scoring snapshots must match the packaged task')
        hashes=audit.get('native_trace_source_sha256',{})
        if not {'summary.json','trace.npz','scene.xml'}<=set(hashes):
            raise ValueError('Geometry audit does not bind all required native evidence')
        if any(hashlib.sha256((folder/name).read_bytes()).hexdigest()!=hashes[name]
               for name in ('summary.json','trace.npz','scene.xml')):
            raise ValueError('Native evidence changed since its geometry audit')
        if audit.get('initial_overlap') or audit.get('sweep_failures'):
            raise ValueError('Invalid geometric source evidence')
        summaries[label]=summary
    contact_hash=gate.get('native_contact_source_sha256')
    if not contact_hash or hashlib.sha256((Path(defended)/'contacts.json').read_bytes()).hexdigest()!=contact_hash:
        raise ValueError('Native contact evidence is missing or changed since its comparison')
    actual=gate['defended_triangle_audit']['all_entries']
    if gate.get('attacker_entries_entire_window')!=[h for h in actual if h['agent']==attacker]:
        raise ValueError('Gate attack entries disagree with the bound geometric audit')
    if gate.get('defender_entries_entire_window')!=[h for h in actual if h['agent']==defender]:
        raise ValueError('Gate counter entries disagree with the bound geometric audit')
    if gate['attacker_entries_entire_window']:
        raise ValueError('A leaking defence cannot be packaged')
    return attacker,defender,summaries


def build(specification, output, *, require_three_regions=True):
    specification,output=Path(specification).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve existing frozen library')
    specs=json.loads(specification.read_text())
    scoring=json.loads((ROOT/'config/scoring_triangle_v1_candidate.json').read_text())
    regions={s['region'] for s in specs['defences']}
    if len(regions)!=len(specs['defences']) or not regions<= {'left','right','upper'}:
        raise ValueError('One explicit candidate for each named region required')
    if require_three_regions and regions!={'left','right','upper'}:
        raise ValueError('Full release requires three separately validated target directions')
    cases={};skills=[];ready=None;provenance={}
    def digest(path):
        path=Path(path);provenance[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    for spec in specs['defences']:
        baseline=ROOT/spec['baseline'];defended=ROOT/spec['defended'];gate_path=ROOT/spec['comparison']
        gate=json.loads(gate_path.read_text());digest(gate_path)
        if gate['status']!='qualified' or gate.get('attacker_entries_entire_window'):
            raise ValueError('Only full-window qualified pure defences may be packaged')
        attacker,defender,summaries=_gate_evidence(gate,baseline,defended,scoring)
        attack=_references(baseline,attacker,summaries['baseline'])
        defence=_references(defended,defender,summaries['defended'])
        q=defence['qpos'][0]
        if ready is None:ready=q
        if not np.array_equal(ready,q) or not np.array_equal(ready,defence['qpos'][-1]):
            raise ValueError('Candidates must share one exact physically validated entry/exit node')
        for label,folder in (('baseline',baseline),('defended',defended)):
            for name in ('references.npz','summary.json','scene.xml','trace.npz'):digest(folder/name)
        digest(defended/'contacts.json')
        observed=corridor_observation(baseline,spec['calibrate_s'],observer=defender)
        fields=observed['physical']
        skill={'name':'parry_'+spec['region'],'region':spec['region'],'defence':defence,
            'corridor':{'tip':fields[PUBLIC_FIELDS[2]],'grip':fields[PUBLIC_FIELDS[3]],'velocity':fields[PUBLIC_FIELDS[4]]},
            'calibration':observed,'coverage':'only tested native input corridor, no arbitrary A2 guarantee'}
        if spec.get('counter'):
            counter=spec['counter'];cp=ROOT/counter['comparison'];cg=json.loads(cp.read_text());digest(cp)
            if not cg.get('counterattack_feasibility_pass') or cg['status']!='defence_with_counter_candidate':
                raise ValueError('Counter requires its own actual-hit and full-window defence evidence')
            native=ROOT/counter['native']
            cb=ROOT/counter['baseline'] if counter.get('baseline') else baseline
            ca,cd,cs=_gate_evidence(cg,cb,native,scoring)
            if ca!=attacker:
                raise ValueError('Counter and defence must use the same physical attacker role')
            skill['counter']=_references(native,cd,cs['defended'])
            # Also bind the counter's attack source, not only its defender.
            _references(cb,ca,cs['baseline'])
            departure=counter['departure_relative_s']
            if isinstance(departure,bool) or not np.isfinite(departure) or departure<0:
                raise ValueError('Counter departure must be a finite relative skill time')
            absolute=cs['defended']['program_start_s'][cd]+departure
            times=skill['counter']['time_s']
            if departure>=times[-1] or not np.any(np.isclose(times,departure,rtol=0,atol=1e-10)):
                raise ValueError('Counter departure must identify an explicit reference knot')
            hits=cg['defender_entries_entire_window']
            contact=cg.get('first_pre_hit_weapon_contact_s')
            if (not hits or contact is None or not np.isfinite(contact)
                    or any(not np.isfinite(h['time_s']) or h['time_s']<absolute or h['time_s']<=contact for h in hits)):
                raise ValueError('Counter needs actual bound hits after departure and interception')
            if 'counter_release_s' in cg and not np.isclose(cg['counter_release_s'],absolute,rtol=0,atol=1e-9):
                raise ValueError('Configured counter departure disagrees with its native experiment')
            skill['counter_departure_relative_s']=float(departure)
            for folder in (cb,native):
                for name in ('references.npz','summary.json','scene.xml','trace.npz'):digest(folder/name)
            digest(native/'contacts.json')
        curve=C2JointSpline(defence['time_s'],defence['qpos'])
        SkillGraphExecutor({'ready':ready},{skill['name']:GraphAction(skill['name'],'ready','ready',curve)},'ready')
        if 'counter' in skill:
            cc=C2JointSpline(skill['counter']['time_s'],skill['counter']['qpos'])
            SkillGraphExecutor({'ready':ready},{'counter':GraphAction('counter','ready','ready',cc)},'ready')
        skills.append(skill)
        cases[spec['region']]={'region':spec['region'],'ready_qpos':ready,'attack':attack}
    result={'schema_version':1,'candidate_only':True,'admitted_to_training':False,
        'scoring':'config/scoring_triangle_v1_candidate.json','duration_s':14.,'cases':cases,'skills':skills,
        'provenance_sha256':provenance,'specification_sha256':hashlib.sha256(specification.read_bytes()).hexdigest(),
        'coverage_regions':sorted(regions),'is_learned_policy':False,
        'qualification_note':'Native skill feasibility only. Revalidate the combined selector, timing perturbations, and both slots before relying on the combined bank.'}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(_finite_json(result),ensure_ascii=False,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--specification',required=True);p.add_argument('--output',required=True)
    p.add_argument('--allow-partial-development-bank',action='store_true');a=p.parse_args()
    build(a.specification,a.output,require_three_regions=not a.allow_partial_development_bank)
