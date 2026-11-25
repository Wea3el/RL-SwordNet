"""Full-window native parry-to-counter feasibility against a fixed upper cut.

No policy, geometry, joint limits or scorer is changed. The first reference
prefix reproduces an independently qualified parry; a bounded continuation
then uses a task-authored existing attack. Actual physics is never reset at a
phase boundary. This is not an autonomous or symmetric-learning duel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .assembly import ROOT, _finite_json
from .a1_triangle_parry import compare
from .constrained_trial import ScheduledReference, hold_reference, run_constrained_trial
from .motion_spline import C2JointSpline


def combined_curve(ready, parry, action, release_s, *, start_s=2.2, scale=2.):
    """Preserve the approach/hold prefix then leave parry without a q reset."""
    if release_s <= start_s+.3:
        raise ValueError('Complete the certified parry approach before release')
    times=[0., .3, release_s-start_s]
    poses=[ready, parry, parry]
    source_times=np.asarray(action['knot_time_s'],float)
    source_poses=np.asarray(action['knot_qpos'],float)
    # Substitute the held parry for the old initial guard; all later source
    # waypoints remain unchanged, including the final rest guard.
    times.extend((release_s-start_s+scale*source_times[1:]).tolist())
    poses.extend(source_poses[1:])
    curve=C2JointSpline(times,poses)
    if max(curve.analytic_extrema()['peak_speed_rad_s'])>2.+1e-9:
        raise ValueError('The combined reference exceeds the shared 2 rad/s ceiling')
    return curve


def run(output, *, release_s=3.4, counter='thrust_right', scale=2., horizon=12., attacker='a1'):
    output=Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve existing probe evidence')
    if attacker not in ('a1','a2'):
        raise ValueError('Select a physical attacker slot')
    defender='a2' if attacker=='a1' else 'a1'
    cfg=json.loads((ROOT/'config/a1_triangle_parry_v1.json').read_text())
    refined=ROOT/cfg['source']/'report.json'
    lower=ROOT/'outputs/a1_triangle_screen_v2/report.json'
    reports=[json.loads(p.read_text()) for p in (refined,lower)]
    attack=reports[0]['actions'][cfg['attack']]
    ready=np.asarray(reports[0]['guards']['center']['qpos_rad'])
    pose=np.asarray(cfg['geometry_derived_poses']['cross_early'])
    name=counter+'__from_center__to_center' if counter!='none' else 'short_parry_return_center'
    if counter=='none':
        curve=C2JointSpline([0.,.3,release_s-2.2,release_s-2.2+1.],
                            [ready,pose,pose,ready])
    else:
        source_report=reports[0] if 'upper' in counter else reports[1]
        candidate=source_report['actions'][name]
        if not candidate['sampled_reference_pass']:
            raise ValueError('Counter source has not passed its own reference screen')
        curve=combined_curve(ready,pose,candidate,release_s,scale=scale)
    counter_program=ScheduledReference(curve,2.2)
    if counter_program.end_s+2.>horizon:
        raise ValueError('Retain at least two seconds after the full recovery')
    original=ScheduledReference(C2JointSpline(
        np.asarray(attack['knot_time_s'])*cfg['attack_time_scale'],
        attack['knot_qpos']),cfg['attack_start_s'])
    output.mkdir(parents=True)
    metadata={'case':'fixed_attack_parry_then_counter','attacker_slot':attacker,
        'counter_source':name,'counter_release_s':release_s,'counter_scale':scale,
        'parry_start_s':2.2,'parry_arrival_s':2.5,
        'counter_end_s':counter_program.end_s,
        'not_policy_reaction':True,'qpos_resets_at_phase_boundaries':0,
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (refined,lower,Path(__file__))}}
    (output/'design.json').write_text(json.dumps(_finite_json({
        'metadata':metadata,'phase_times_s':{
            'parry_start':2.2,'parry_arrival':2.5,'counter_departure':release_s,
            'counter_recovery_complete':counter_program.end_s,'observation_end':horizon},
        'counter_reference_report':curve.report(),
        'all_physical_sources_unchanged':True,
        'scope':'One fixed upper cut followed through recovery; not learned response or training admission',
        'upstream_dependency':'Same triangular scoring and independently qualified parry prefix',
        'downstream_risk':'Other attack directions and arbitrary continuous A2 trajectories are not covered'
    }),indent=2)+'\n')
    summary=run_constrained_trial({attacker:original,defender:counter_program},output/'native',
        duration_s=horizon,metadata=metadata,progress=print)
    baseline=ROOT/('outputs/a1_triangle_parry_v1/baseline' if attacker=='a1'
                   else 'outputs/a1_triangle_parry_mirror_v1/baseline')
    if not np.isclose(horizon,12.):
        baseline=output.parent/('baseline_'+attacker+'_'+format(horizon,'g')+'s')
        baseline_metadata={'case':'matched_counter_baseline','horizon_s':horizon,
                           'attacker_slot':attacker,'not_policy_reaction':True}
        if not baseline.exists():
            run_constrained_trial({attacker:original,defender:hold_reference(ready,.75,
                original.curve.duration_s)},baseline,duration_s=horizon,
                metadata=baseline_metadata,progress=print)
        else:
            prior=json.loads((baseline/'summary.json').read_text())
            if prior.get('metadata')!=baseline_metadata:
                raise ValueError('Existing baseline has a different identity')
    score=json.loads((ROOT/cfg['scoring']).read_text())
    report=compare(baseline,output/'native',score,attacker=attacker)
    hits=report['defender_entries_entire_window']
    valid_counters=[h for h in hits if h['time_s']>=release_s]
    preemptive=[h for h in hits if h['time_s']<release_s]
    report.update(counter_source=name,counter_release_s=release_s,
        reference_c2_continuity=curve.continuity_report(),
        counter_recovery_complete_s=counter_program.end_s,
        valid_counter_entries=valid_counters,unexpected_pre_counter_entries=preemptive,
        short_parry_feasibility_pass=bool(counter=='none' and report['status']=='qualified'),
        counterattack_feasibility_pass=bool(counter!='none' and
            report['status']=='defence_with_counter_candidate' and valid_counters
            and not preemptive and summary['native_execution_pass']),
        learned_response=False,admitted_to_training=False,
        scope='Exact fixed upper cut, parry prefix, departure clock and counter path; no general policy claim')
    (output/'comparison.json').write_text(json.dumps(_finite_json(report),indent=2)+'\n')
    print('COUNTER RESULT',report['status'],report['counterattack_feasibility_pass'],flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--release-s',type=float,default=3.4)
    parser.add_argument('--counter',default='thrust_right')
    parser.add_argument('--scale',type=float,default=2.)
    parser.add_argument('--horizon',type=float,default=12.)
    parser.add_argument('--attacker-slot',choices=('a1','a2'),default='a1')
    args=parser.parse_args()
    run(args.output,release_s=args.release_s,counter=args.counter,scale=args.scale,
        horizon=args.horizon,attacker=args.attacker_slot)


if __name__=='__main__':
    main()
