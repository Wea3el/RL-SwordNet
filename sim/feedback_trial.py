"""Bounded online-observation tests of the new constrained skill graph.

Same native control, collision, scoring and foul handling as constrained_trial.
No training updates. Scripted test opponents are identified as such. Policies
receive current physical observations and available own actions only.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import time
import mujoco
import numpy as np

from .a1_runtime import RuntimeProfile, apply_runtime_profile, assert_runtime_profile, load_runtime_profile
from .assembly import ROOT, _finite_json
from .blade_geometry import blade_geometry_identity
from .duel_scene import duel_contact_records
from .feedback_policy import SkillGraphExecutor, TimedRequests
from .fencing_rules import FirstTouchScorer, blade_sections_from_scene
from .limited_duel import build_limited_duel
from .manual_duel import position_control
from .training_contacts import TrainingContactMonitor, TrainingEnvironmentError, settle_training_tick
from .trial_completion import ExecutorCompletionSnapshot, TrialCompletionSnapshot, assess_trial_completion

SLOTS = ('a1', 'a2')


def physical_observation(scene, slot):
    """Same physical quantities for either slot; no phase, time or program."""
    enemy = SLOTS[1-SLOTS.index(slot)]
    own, other = scene.agents[slot], scene.agents[enemy]
    rotation = np.diag([1., 1., 1.]) if slot == 'a1' else np.diag([-1., -1., 1.])
    target = np.asarray(scene.config['virtual_targets'][slot]['center_world_m'])
    jac = np.zeros((3, scene.model.nv))
    mujoco.mj_jacSite(scene.model, scene.data, jac, None, other['sites']['shinai_tip'])
    return {
        'own_q_rad': scene.data.qpos[own['qpos_ids']].copy(),
        'own_qvel_rad_s': scene.data.qvel[own['dof_ids']].copy(),
        'opponent_tip_minus_own_target_local_m': rotation@(scene.data.site_xpos[other['sites']['shinai_tip']]-target),
        'opponent_grip_minus_own_target_local_m': rotation@(scene.data.site_xpos[other['sites']['shinai_grip']]-target),
        'opponent_tip_velocity_local_m_s': rotation@(jac@scene.data.qvel),
    }


def run_feedback_trial(executors, policies, output, *, duration_s, metadata=None, progress=None,
                       runtime_profile=None):
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError('Preserve prior evidence; use a fresh output')
    if set(executors) != set(SLOTS) or set(policies) != set(SLOTS):
        raise ValueError('Explicit independent executors and policies required for both slots')
    if executors['a1'] is executors['a2'] or policies['a1'] is policies['a2']:
        raise ValueError('Slots cannot share mutable controller/policy objects')
    if any(not isinstance(e, SkillGraphExecutor) or e.events or e.active or e._trial_consumed for e in executors.values()):
        raise ValueError('Fresh skill graph executors required')
    if any(getattr(p,'_trial_consumed',False) for p in policies.values()):
        raise ValueError('Fresh policies required; recurrent state cannot leak across trials')
    if isinstance(duration_s, bool) or not np.isfinite(duration_s) or not 0 < duration_s <= 60:
        raise ValueError('Finite bounded duration required')
    # An explicit opt-in numerical profile affects both agents and only clocks.
    # The historical default, physical parameters and scoring gates are intact.
    profile = (load_runtime_profile(ROOT/'config/a1_runtime_rk4_candidate.json') if runtime_profile is None
               else runtime_profile if isinstance(runtime_profile, RuntimeProfile)
               else load_runtime_profile(runtime_profile))
    cfg = profile.runtime_config()
    dt = cfg['physics_timestep_s']
    if not np.isclose(duration_s/dt, round(duration_s/dt), atol=1e-8, rtol=0):
        raise ValueError('Duration must be an integer number of physical steps')
    scene = build_limited_duel(cfg['base_distance_m'], initial_qpos={s:e.initial_q for s,e in executors.items()})
    apply_runtime_profile(scene, profile)
    model, data = scene.model, scene.data
    blade_identities = {slot: blade_geometry_identity(model, slot+'_') for slot in SLOTS}
    if blade_identities['a1']['sha256'] != blade_identities['a2']['sha256']:
        raise ValueError('Both agents must use identical local blade collision geometry')
    actual_geometry = blade_identities['a1']['sha256']
    metadata = copy.deepcopy(metadata or {})
    if metadata.get('collision_geometry_sha256', actual_geometry) != actual_geometry:
        raise ValueError('Requested blade identity differs from actual native geometry')
    metadata['collision_geometry_sha256'] = actual_geometry
    output.mkdir(parents=True)
    (output/'scene.xml').write_text(scene.spec.to_xml())
    files = ('sim/feedback_trial.py','sim/feedback_policy.py','sim/trial_completion.py','sim/constrained_trial.py',
        'sim/limited_duel.py','sim/limited_model.py','sim/motion_spline.py','sim/joint_constraints.py',
        'sim/training_contacts.py','sim/foul_attribution.py','sim/fencing_rules.py','sim/manual_duel.py',
        'sim/duel_scene.py','sim/assembly.py','sim/floor.py','sim/model.py','sim/shinai.py',
        'sim/blade_geometry.py','assets/shinai_profile.json',
        'sim/a1_runtime.py','config/assembly.json','config/joint_constraints_observed.json',
        'config/a1_runtime_rk4_candidate.json')
    hashes = {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}
    initial_nodes = {s:e.node for s,e in executors.items()}
    arrays, graph_description = {}, {}
    for s,e in executors.items():
        graph_description[s] = {'initial_node':e.node,
            'nodes':{name:q.tolist() for name,q in e.nodes.items()},'actions':{}}
        for name, action in e.actions.items():
            graph_description[s]['actions'][name] = {'start_node':action.start_node,'end_node':action.end_node}
            for field in ('knot_time_s','knot_qpos','coefficients'):
                arrays[f'{s}__{name}__{field}'] = getattr(action.curve,field)
    np.savez_compressed(output/'graph_references.npz',**arrays)
    result = {'status':'running', 'scope':'online physical observation diagnostic, not trained policy or self-play',
        'runtime_config':cfg,'requested_duration_s':float(duration_s), 'metadata':metadata,
        'numerical_profile_source_sha256':profile.source_sha256,
        'numerical_profile_source':json.loads(profile.source_bytes),
        'source_sha256':hashes,'model_source_sha256':scene.source_hashes,'graph':graph_description,
        'joint_constraints':scene.config['joint_constraints'],'qpos_resets_after_initialization':0,
        'admitted_to_training':False,'selfplay_or_training_run':False,
        'initial_qpos_rad':data.qpos.tolist(), 'policy_classes':{s:type(p).__name__ for s,p in policies.items()},
        'initial_policy_state':{s:copy.deepcopy(p.__dict__) for s,p in policies.items()},
        'native_events':[], 'failure':None, 'evidence_complete':False}
    def save():
        (output/'summary.json').write_text(json.dumps(_finite_json(result),indent=2)+'\n')
    save()
    for e in executors.values(): e._trial_consumed=True
    for p in policies.values(): p._trial_consumed=True
    gravity = mujoco.MjData(model)
    commands = {s:e.initial_q for s,e in executors.items()}
    position_control(scene,gravity,commands);mujoco.mj_forward(model,data)
    centers = {s:scene.config['virtual_targets'][SLOTS[1-i]]['center_world_m'] for i,s in enumerate(SLOTS)}
    radii = {s:scene.config['virtual_targets'][SLOTS[1-i]]['radius_m'] for i,s in enumerate(SLOTS)}
    scorer = FirstTouchScorer(centers,radii)
    previous = blade_sections_from_scene(scene)
    scorer.update(previous,previous,0.,0.)
    monitor = TrainingContactMonitor(scene,cfg)
    caps = np.minimum(-model.actuator_forcerange[:,0],model.actuator_forcerange[:,1])
    ranges = model.jnt_range.copy()
    total_steps = round(duration_s/dt)
    control_steps = round(cfg['control_period_s']/dt)
    decision_steps = round(cfg['decision_period_s']/dt)
    rows, observations, contact_log, failures = [],[],[],[]
    max_penetration = max_violation = max_limit_force = 0.
    max_tracking = np.zeros(model.nq)
    first_terminal = None
    final_tail = {s:[] for s in SLOTS}

    def faults(contacts):
        nonlocal max_penetration,max_violation,max_limit_force
        found=[]
        if not all(np.isfinite(a).all() for a in (data.qpos,data.qvel,data.qacc,data.ctrl,data.actuator_force)):
            found.append('nonfinite_native_state')
        if any(w.number for w in data.warning): found.append('mujoco_warning')
        if np.any(model.body_gravcomp) or np.any(data.qfrc_applied) or np.any(data.xfrc_applied):
            found.append('unsupported_external_force')
        if np.any(np.abs(data.actuator_force)>caps+1e-6): found.append('native_actuator_cap_exceeded')
        violation=max(0.,float(np.max(ranges[:,0]-data.qpos)),float(np.max(data.qpos-ranges[:,1])))
        max_violation=max(max_violation,violation)
        if violation>1e-6: found.append('actual_confirmed_joint_limit_violation')
        depth=max([0.]+[-c['distance_m'] for c in contacts]);max_penetration=max(max_penetration,depth)
        if depth>cfg['penetration_limit_m']: found.append('excessive_proxy_penetration')
        ids=np.flatnonzero(data.efc_type[:data.nefc]==int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT))
        force=float(np.max(np.abs(data.efc_force[ids]),initial=0.));max_limit_force=max(max_limit_force,force)
        if force>cfg['hypothetical_limit_force_ceiling_nm']: found.append('joint_limit_assistance')
        if scorer.failure: found.append('scorer_failure')
        return found

    def record():
        command=np.empty(model.nq)
        for s in SLOTS: command[scene.agents[s]['qpos_ids']]=commands[s]
        max_tracking[:]=np.maximum(max_tracking,np.abs(data.qpos-command))
        rows.append((float(data.time),data.qpos.copy(),data.qvel.copy(),command,data.ctrl.copy(),data.actuator_force.copy()))

    initial_contacts=duel_contact_records(scene,rules_version=cfg['rules_version'])
    failures+=faults(initial_contacts)
    if scorer.initial_overlap_events: failures.append('initial_target_overlap')
    if any(c['distance_m']<-.001 for c in initial_contacts): failures.append('initial_proxy_penetration')
    if any(c['category'].startswith('illegal_') and c['normal_force_n']>cfg['illegal_contact_force_threshold_n'] for c in initial_contacts):
        failures.append('invalid_initial_forbidden_force')
    record()
    completed_steps=0
    start_wall=last_progress=time.monotonic()
    try:
        for step in range(total_steps):
            if failures: break
            if time.monotonic()-start_wall>240.:
                failures.append('wall_clock_diagnostic_budget_exhausted');break
            assert_runtime_profile(scene,profile)
            now=float(data.time)
            for s,e in executors.items():
                m=scene.agents[s];e.observe(now,data.qpos[m['qpos_ids']],data.qvel[m['dof_ids']])
            if step%decision_steps==0:
                obs={s:physical_observation(scene,s) for s in SLOTS}
                available={s:executors[s].available() for s in SLOTS}
                # BOTH choices use the same pre-action physical snapshot.
                requests={s:policies[s].choose(copy.deepcopy(obs[s]),available[s]) for s in SLOTS}
                observations.append({'time_s':now,
                    'physical':{s:{k:v.tolist() for k,v in fields.items()} for s,fields in obs.items()},
                    'available':available,'requests':requests})
                for s,request in requests.items():
                    if request is not None: executors[s].request(request,now)
            if step%control_steps==0:
                commands={s:e.target(now) for s,e in executors.items()}
                position_control(scene,gravity,commands);mujoco.mj_forward(model,data)
            monitor.capture_before_step(scene)
            old_time=float(data.time)
            mujoco.mj_step(model,data);mujoco.mj_forward(model,data)
            completed_steps=step+1
            current=blade_sections_from_scene(scene)
            hits=scorer.update(previous,current,old_time,float(data.time))
            try:
                contact=monitor.inspect(scene,completed_steps)
            except TrainingEnvironmentError as exc:
                raw=duel_contact_records(scene,rules_version=cfg['rules_version'])
                contact_log.append({'time_s':float(data.time),'interval_start_s':old_time,
                    'interval_end_s':float(data.time),'contacts':raw,'evidence_error':copy.deepcopy(exc.details)})
                raise
            tick_faults=faults(contact['contacts'])+list(contact['numerical_faults'])
            failures.extend(tick_faults)
            if contact['contacts']:
                contact_log.append({'time_s':float(data.time),'interval_start_s':old_time,
                                    'interval_end_s':float(data.time),'contacts':contact['contacts']})
            settlement=settle_training_tick(completed_steps,contacts=contact['verdicts'],
                    hit_agents=[h['agent'] for h in hits],numerical_faults=tick_faults)
            if hits or contact['verdicts'] or tick_faults:
                result['native_events'].append({'time_s':float(data.time),'hits':hits,
                    'contact_verdicts':contact['verdicts'],'numerical_faults':tick_faults,'settlement':settlement})
            if settlement['terminal'] and first_terminal is None:
                first_terminal=copy.deepcopy(settlement);first_terminal['time_s']=float(data.time)
            if contact['verdicts']: failures.append('foul_ends_physical_diagnostic')
            if data.time>=duration_s-.1-1e-9:
                for s,e in executors.items():
                    m=scene.agents[s]
                    final_tail[s].append(bool(e.active is None and
                        np.max(np.abs(data.qpos[m['qpos_ids']]-e.nodes[e.node]))<=.05 and
                        np.max(np.abs(data.qvel[m['dof_ids']]))<=.10))
            previous=current;record()
            if time.monotonic()-last_progress>=10.:
                result['completed_time_s']=float(data.time);save()
                if progress: progress(f'{output.name}: {data.time:.3f}/{duration_s:.3f} simulated seconds',flush=True)
                last_progress=time.monotonic()
    except TrainingEnvironmentError as exc:
        failures.append('training_contact_or_numeric_evidence_error');result['failure_details']=copy.deepcopy(exc.details)
    except ValueError as exc:
        failures.append('runtime_validation_error');result['failure_details']=str(exc)
    except Exception as exc:
        failures.append('unexpected_execution_error')
        result['failure_details']={'type':type(exc).__name__,'message':str(exc)}
    if rows[-1][0]!=float(data.time): record()
    unchanged=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    if not unchanged: failures.append('source_changed_during_trial')
    if not failures: scorer.finish(float(data.time))
    for s,e in executors.items():
        mapping=scene.agents[s]
        e.observe(float(data.time),data.qpos[mapping['qpos_ids']],data.qvel[mapping['dof_ids']])
    settled={s:bool(final_tail[s] and all(final_tail[s])) for s in SLOTS}
    def completion_snapshot(extra_evidence=()):
        details=result.get('failure_details',{})
        numeric=list(details.get('numerical_faults',())) if isinstance(details,dict) else []
        evidence_detail=list(details.get('evidence_errors',())) if isinstance(details,dict) else []
        physical_codes={'nonfinite_native_state','mujoco_warning','unsupported_external_force',
            'native_actuator_cap_exceeded','actual_confirmed_joint_limit_violation',
            'excessive_proxy_penetration','joint_limit_assistance','foul_ends_physical_diagnostic'}
        evidence_codes={'source_changed_during_trial','scorer_failure','evidence_write_failure'}
        physical=tuple(numeric+[f for f in failures if f in physical_codes])
        evidence=tuple([f for f in failures if f in evidence_codes]+list(extra_evidence)
                       +(['contact_evidence_error'] if evidence_detail else []))
        execution=tuple(f for f in failures if f not in physical_codes|evidence_codes
            and f!='wall_clock_diagnostic_budget_exhausted'
            and not(f=='training_contact_or_numeric_evidence_error' and (numeric or evidence_detail)))
        stop=('wall_budget' if 'wall_clock_diagnostic_budget_exhausted' in failures else
              'horizon' if completed_steps==total_steps else 'interrupted')
        snapshots={s:ExecutorCompletionSnapshot(active_action=e.active,
            reference_end_s=None if e.active is None else e.started_s+e.actions[e.active].curve.duration_s,
            settled=settled[s],pending_script_requests=(tuple(a for _,a in policies[s].requests[policies[s].next:])
                if isinstance(policies[s],TimedRequests) else ())) for s,e in executors.items()}
        return assess_trial_completion(TrialCompletionSnapshot(completed_time_s=float(data.time),
            requested_duration_s=float(duration_s),stop_reason=stop,executors=snapshots,
            physical_failures=physical,evidence_failures=evidence,execution_failures=execution))
    completion=completion_snapshot()
    complete=completion['termination_category']=='complete'
    result.update(status='completed' if complete else 'truncated' if completion['is_truncation'] else 'failed',failure=failures or None,
        completion=completion,
        completed_time_s=float(data.time),physics_steps=completed_steps,source_unchanged=unchanged,
        first_settlement=first_terminal,official_match_result=first_terminal if first_terminal is not None else (
            {'status':'timeout_draw','match_winner':None,'rl_reward':{s:0. for s in SLOTS}}
            if completion['allows_timeout_draw_if_no_scored_result'] else None),
        all_target_entries=scorer.all_entries,initial_overlap_events=scorer.initial_overlap_events,
        minimum_target_clearance_m=scorer.minimum_clearance_m,final_settled=settled,
        maximum_actual_joint_violation_rad=max_violation,maximum_joint_limit_force_nm=max_limit_force,
        maximum_penetration_m=max_penetration,maximum_joint_tracking_error_rad=max_tracking.tolist(),
        native_execution_pass=complete and completed_steps==total_steps and all(settled.values()),
        executor_events={s:e.events for s,e in executors.items()},policy_events={s:list(p.events) for s,p in policies.items()},
        initial_nodes=initial_nodes,final_nodes={s:e.node for s,e in executors.items()},
        elapsed_wall_s=time.monotonic()-start_wall)
    save()
    try:
        np.savez_compressed(output/'trace.npz',**{name:np.asarray([r[i] for r in rows]) for i,name in
            enumerate(('time_s','qpos','qvel','target_qpos','ctrl','actuator_force_nm'))})
        (output/'contacts.json').write_text(json.dumps(_finite_json(contact_log),indent=2)+'\n')
        (output/'observations.json').write_text(json.dumps(_finite_json(observations),indent=2)+'\n')
        result['evidence_complete']=True
    except (OSError, TypeError, ValueError) as exc:
        result['native_execution_pass']=False
        result['status']='failed'
        result['failure']=(result['failure'] or [])+['evidence_write_failure']
        result['failure_details']=str(exc)
        result['completion']=completion_snapshot(extra_evidence=('evidence_write_failure',))
        save()
        raise
    save()
    return result
