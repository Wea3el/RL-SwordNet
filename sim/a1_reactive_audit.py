"""Separate per-attack and decision audit; do not rewrite physical evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .assembly import _finite_json
from .a1_triangle_reactive import PUBLIC_FIELDS


def episode_checks(comparison, baseline, defended, observations, attack_duration_s):
    attacker=baseline['metadata']['attacker_slot'];defender='a2' if attacker=='a1' else 'a1'
    expected=baseline['metadata'].get('repeat_count',1)
    starts=[e['time_s'] for e in baseline['executor_events'][attacker] if e['event']=='started']
    defence_starts=[e for e in defended['executor_events'][defender] if e['event']=='started']
    completed=[e for e in defended['executor_events'][defender] if e['event']=='completed']
    cues=[e for e in defended['policy_events'][defender] if e['event']=='current_geometry_trigger']
    baseline_hits=[e for e in comparison['baseline_triangle_audit']['all_entries'] if e['agent']==attacker]
    target=comparison['baseline_triangle_audit']['target_region']
    windows=[]
    for i,start in enumerate(starts):
        end=min(start+attack_duration_s+2.,starts[i+1] if i+1<len(starts) else baseline['completed_time_s'])
        hits=[e for e in baseline_hits if start<=e['time_s']<end]
        windows.append({'start_s':start,'end_s':end,'baseline_target_hit':any(e['region']==target for e in hits),
            'defence_start_count':sum(start<=e['time_s']<end for e in defence_starts)})
    checks={'physical_pair_gate':comparison['status'] in ('qualified','defence_with_counter_candidate'),
        'expected_attack_count':len(starts)==expected,
        'every_attack_has_a_real_target_threat':len(windows)==expected and all(w['baseline_target_hit'] for w in windows),
        'one_defence_per_attack':len(defence_starts)==expected and all(w['defence_start_count']==1 for w in windows),
        'all_defences_complete':len(completed)==expected,
        'one_public_trigger_per_defence':len(cues)==expected,
        'no_early_defence_before_first_attack':bool(starts and all(e['time_s']>=starts[0] for e in defence_starts)),
        'public_fields_identical_for_both_slots':all(set(row['physical'][s])==set(PUBLIC_FIELDS)
            for row in observations for s in ('a1','a2')),
        'requests_respect_available_own_actions':all(row['requests'][s] is None or row['requests'][s] in row['available'][s]
            for row in observations for s in ('a1','a2')),
        'no_defender_clock_or_future_fields':all(set(row['physical'][defender])==set(PUBLIC_FIELDS) for row in observations),
        'observed_rearm_before_repeat':expected==1 or sum(e['event']=='observed_rearm'
            for e in defended['policy_events'][defender])>=expected-1}
    # Verify that the actual public snapshot emitted the requested action; a
    # posthoc event label alone cannot establish causal request provenance.
    provenance=[]
    for cue in cues:
        i=cue['sample']-1
        valid=0<=i<len(observations)
        if valid:
            row=observations[i]
            valid=(row['requests'][defender]==cue['action'] and all(
                np.array_equal(row['physical'][defender][key],cue['public_physical'][key]) for key in PUBLIC_FIELDS))
        provenance.append(valid)
    checks['trigger_matches_recorded_public_snapshot']=bool(provenance and all(provenance))
    return {'status':'passed' if all(checks.values()) else 'failed','checks':checks,'attack_windows':windows,
        'defence_starts':defence_starts,'defence_completions':completed,
        'no_learned_policy_claim':True,'admitted_to_training':False,
        'scope':'These tested motions, geometry corridors, timing and dynamics only; not unseen A2 trajectory coverage'}


def audit(folder,library=None):
    folder=Path(folder).resolve();output=folder/'decision_audit.json'
    if output.exists(): raise FileExistsError('Preserve prior decision audit')
    paths=[folder/'comparison.json',folder/'baseline/summary.json',folder/'defended/summary.json',folder/'defended/observations.json']
    comp,b,d,obs=[json.loads(p.read_text()) for p in paths]
    attacker=b['metadata']['attacker_slot']
    with np.load(folder/'baseline/graph_references.npz',allow_pickle=False) as ref:
        duration=float(ref[attacker+'__attack__knot_time_s'][-1])
    result=episode_checks(comp,b,d,obs,duration)
    if d['metadata'].get('mode')=='counter':
        verified=False;phases=[]
        if library is not None:
            library=Path(library)
            verified=hashlib.sha256(library.read_bytes()).hexdigest()==d['metadata']['library_sha256']
            if verified:
                skills={s['name']:s for s in json.loads(library.read_text())['skills']}
                for event in result['defence_starts']:
                    skill=skills[event['action']]
                    if 'counter_departure_relative_s' not in skill:
                        verified=False;break
                    phases.append(event['time_s']+skill['counter_departure_relative_s'])
        hits=comp['defender_entries_entire_window']
        contact=comp['first_pre_hit_weapon_contact_s']
        counter_ok=bool(verified and phases and contact is not None and hits
            and all(any(h['time_s']>=p for h in hits) for p in phases)
            and all(h['time_s']>=min(phases) and h['time_s']>contact for h in hits))
        result['checks']['counter_library_identity_and_phase_verified']=verified
        result['checks']['real_hit_after_interception_and_counter_departure']=counter_ok
        result['counter_departure_times_s']=phases
        result['counter_entries']=hits
        result['status']='passed' if all(result['checks'].values()) else 'failed'
    result['source_sha256']={str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    output.write_text(json.dumps(_finite_json(result),indent=2)+'\n')
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder');p.add_argument('--library')
    a=p.parse_args();audit(a.folder,a.library)
