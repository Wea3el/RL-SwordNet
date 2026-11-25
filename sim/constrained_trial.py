"""Bounded native motion experiments under confirmed joint limits.

Scheduled references are diagnostic controls, not a learned policy or a new
training action space. Geometry scoring and approved foul priority are reused;
first touch is recorded without hiding later entries during safe continuation.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from .a1_runtime import apply_runtime_profile, assert_runtime_profile, load_runtime_profile
from .assembly import ROOT, _finite_json
from .blade_geometry import blade_geometry_identity
from .duel_scene import duel_contact_records
from .fencing_rules import FirstTouchScorer, blade_sections_from_scene
from .joint_constraints import enforce_joint_trajectory, load_joint_constraints, model_joint_ranges_rad
from .limited_duel import build_limited_duel
from .manual_duel import position_control
from .motion_spline import C2JointSpline
from .training_contacts import TrainingContactMonitor, TrainingEnvironmentError, settle_training_tick


SLOTS = ('a1', 'a2')


@dataclass(frozen=True)
class ScheduledReference:
    curve: C2JointSpline
    start_s: float = .75

    def __post_init__(self):
        if (not isinstance(self.curve, C2JointSpline) or isinstance(self.start_s, bool)
                or not np.isfinite(self.start_s) or self.start_s < 0
                or self.curve.knot_time_s[0] != 0):
            raise ValueError('A C2 curve starting at zero and finite nonnegative scheduled start required')

    def target(self, now):
        return self.curve.sample(float(now)-self.start_s)

    @property
    def end_s(self):
        return float(self.start_s+self.curve.duration_s)


def hold_reference(q, start_s=.75, duration_s=1.):
    return ScheduledReference(C2JointSpline([0., duration_s], [q, q]), start_s)


def validate_references(programs, constraints, duration_s):
    if set(programs) != set(SLOTS):
        raise ValueError('Both slots need explicit references')
    if (isinstance(duration_s, bool) or not np.isfinite(duration_s)
            or duration_s <= 0 or duration_s > 60):
        raise ValueError('Bounded positive trial duration up to 60 seconds required')
    ranges = model_joint_ranges_rad(constraints)
    for program in programs.values():
        if not isinstance(program, ScheduledReference):
            raise ValueError('Only explicit scheduled C2 references accepted')
        enforce_joint_trajectory(program.curve.knot_qpos, constraints)
        controls = program.curve.bezier_controls
        if np.any(controls < ranges[:, 0]) or np.any(controls > ranges[:, 1]):
            raise ValueError('Whole command curve must remain inside confirmed ranges')
        if duration_s < program.end_s+.1-1e-9:
            raise ValueError('Keep at least the existing .1 second endpoint observation dwell')


def run_constrained_trial(programs, output, *, duration_s, metadata=None, progress=None):
    """Run one full-window two-arm diagnostic without post-reset state writes.

    Preserve first settlement while continuing after a hit. A foul, numerical
    failure or invalid contact evidence stops this trial and prevents motion
    certification. No changes to spacing, targets, caps, forces or game rules.
    """
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a fresh directory; keep failed trials')
    constraints = load_joint_constraints()
    validate_references(programs, constraints, duration_s)
    profile_path = ROOT/'config/a1_runtime_rk4_candidate.json'
    profile = load_runtime_profile(profile_path)
    cfg = profile.runtime_config()
    dt = cfg['physics_timestep_s']
    if not np.isclose(duration_s/dt, round(duration_s/dt), atol=1e-8, rtol=0):
        raise ValueError('Trial duration must be an integer count of physical steps')
    scene = build_limited_duel(cfg['base_distance_m'],
        initial_qpos={slot: program.curve.sample(0) for slot, program in programs.items()},
        constraints=constraints)
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
    files = ('sim/constrained_trial.py', 'sim/limited_duel.py', 'sim/limited_model.py',
             'sim/motion_spline.py', 'sim/joint_constraints.py', 'sim/training_contacts.py',
             'sim/foul_attribution.py', 'sim/fencing_rules.py', 'sim/manual_duel.py',
             'sim/duel_scene.py', 'sim/assembly.py', 'sim/floor.py', 'sim/model.py',
             'sim/shinai.py', 'sim/blade_geometry.py', 'sim/a1_runtime.py', 'config/assembly.json',
             'assets/shinai_profile.json', 'config/joint_constraints_observed.json',
             'config/a1_runtime_rk4_candidate.json')
    hashes = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}
    arrays = {}
    for slot, program in programs.items():
        arrays[slot+'_knot_time_s'] = program.curve.knot_time_s
        arrays[slot+'_knot_qpos'] = program.curve.knot_qpos
        arrays[slot+'_coefficients'] = program.curve.coefficients
    np.savez_compressed(output/'references.npz', **arrays)
    result = {'status': 'running', 'scope': 'scheduled native diagnostic, not learned A1/A2 or training admission',
        'runtime_config': cfg, 'initialization': 'one explicit legal reset; no approach from CAD zero certified',
        'qpos_resets_after_initialization': 0, 'metadata': metadata,
        'requested_duration_s': float(duration_s), 'source_sha256': hashes,
        'model_source_sha256': scene.source_hashes,
        'program_start_s': {s:p.start_s for s,p in programs.items()},
        'joint_constraints': scene.config.get('joint_constraints'),
        'native_events': [], 'failure': None, 'first_settlement': None,
        'admitted_to_training': False, 'selfplay_or_training_run': False,
    }

    def save():
        (output/'summary.json').write_text(json.dumps(_finite_json(result), indent=2)+'\n')

    save()
    gravity = mujoco.MjData(model)
    commands = {s:p.target(0.) for s,p in programs.items()}
    position_control(scene, gravity, commands); mujoco.mj_forward(model, data)
    centers = {s:scene.config['virtual_targets'][SLOTS[1-i]]['center_world_m'] for i,s in enumerate(SLOTS)}
    radii = {s:scene.config['virtual_targets'][SLOTS[1-i]]['radius_m'] for i,s in enumerate(SLOTS)}
    scorer = FirstTouchScorer(centers, radii)
    previous = blade_sections_from_scene(scene)
    scorer.update(previous, previous, 0., 0.)
    monitor = TrainingContactMonitor(scene, cfg)
    caps = np.minimum(-model.actuator_forcerange[:, 0], model.actuator_forcerange[:, 1])
    ranges = model.jnt_range.copy()
    control_steps = round(cfg['control_period_s']/dt)
    total_steps = round(duration_s/dt)
    rows = []
    worst_error = np.zeros(model.nq)
    worst_reference_error = np.zeros(model.nq)
    max_penetration = max_range_violation = max_limit_force = 0.
    failures = []
    first_terminal = None
    contact_log = []
    entry_ready = {}
    final_tail = {s:[] for s in SLOTS}

    def faults(contacts):
        nonlocal max_penetration, max_range_violation, max_limit_force
        found = []
        if not all(np.isfinite(a).all() for a in (data.qpos, data.qvel, data.qacc, data.ctrl, data.actuator_force)):
            found.append('nonfinite_native_state')
        if any(w.number for w in data.warning): found.append('mujoco_warning')
        if np.any(model.body_gravcomp) or np.any(data.qfrc_applied) or np.any(data.xfrc_applied):
            found.append('unsupported_external_force')
        if np.any(np.abs(data.actuator_force) > caps+1e-6): found.append('native_actuator_cap_exceeded')
        violation = max(0., float(np.max(ranges[:,0]-data.qpos)), float(np.max(data.qpos-ranges[:,1])))
        max_range_violation = max(max_range_violation, violation)
        if violation > 1e-6: found.append('actual_confirmed_joint_limit_violation')
        depth = max([0.] + [-c['distance_m'] for c in contacts])
        max_penetration = max(max_penetration, depth)
        if depth > cfg['penetration_limit_m']: found.append('excessive_proxy_penetration')
        lid = np.flatnonzero(data.efc_type[:data.nefc] == int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT))
        force = float(np.max(np.abs(data.efc_force[lid]), initial=0.))
        max_limit_force = max(max_limit_force, force)
        if force > cfg['hypothetical_limit_force_ceiling_nm']: found.append('joint_limit_assistance')
        if scorer.failure: found.append('scorer_failure')
        return found

    def record():
        target, reference = np.empty(model.nq), np.empty(model.nq)
        for slot, program in programs.items():
            qid = scene.agents[slot]['qpos_ids']
            target[qid] = commands[slot]
            reference[qid] = program.target(data.time)
        worst_error[:] = np.maximum(worst_error, np.abs(data.qpos-target))
        worst_reference_error[:] = np.maximum(worst_reference_error, np.abs(data.qpos-reference))
        rows.append((float(data.time), data.qpos.copy(), data.qvel.copy(), target,
                     data.ctrl.copy(), data.actuator_force.copy(), reference))

    initial_contacts = duel_contact_records(scene, rules_version=cfg['rules_version'])
    failures += faults(initial_contacts)
    if scorer.initial_overlap_events: failures.append('initial_target_overlap')
    if any(c['distance_m'] < -.001 for c in initial_contacts): failures.append('initial_proxy_penetration')
    if any(c['category'].startswith('illegal_') and
           c['normal_force_n'] > cfg['illegal_contact_force_threshold_n'] for c in initial_contacts):
        failures.append('invalid_initial_forbidden_force')
    record()
    completed_steps = 0
    wall_start = time.monotonic()
    last_progress = wall_start
    try:
        for step in range(total_steps):
            if failures: break
            if time.monotonic()-wall_start > 180.:
                failures.append('wall_clock_diagnostic_budget_exhausted')
                break
            assert_runtime_profile(scene, profile)
            now = float(data.time)
            for slot, program in programs.items():
                if slot not in entry_ready and now >= program.start_s-1e-9:
                    mapping = scene.agents[slot]
                    entry_ready[slot] = bool(np.max(np.abs(data.qpos[mapping['qpos_ids']]-program.curve.sample(0))) <= .05
                        and np.max(np.abs(data.qvel[mapping['dof_ids']])) <= .10)
                    if not entry_ready[slot]: failures.append('entry_not_settled:'+slot)
            if failures: break
            if step % control_steps == 0:
                commands = {s:p.target(now) for s,p in programs.items()}
                position_control(scene, gravity, commands)
                mujoco.mj_forward(model, data)
            monitor.capture_before_step(scene)
            previous_time = float(data.time)
            mujoco.mj_step(model, data); mujoco.mj_forward(model, data)
            completed_steps = step+1
            current = blade_sections_from_scene(scene)
            hits = scorer.update(previous, current, previous_time, float(data.time))
            try:
                contact = monitor.inspect(scene, completed_steps)
            except TrainingEnvironmentError as exc:
                raw = duel_contact_records(scene, rules_version=cfg['rules_version'])
                contact_log.append({'time_s':float(data.time), 'interval_start_s':previous_time,
                    'interval_end_s':float(data.time), 'physics_tick':completed_steps,
                    'contacts':raw, 'evidence_error':copy.deepcopy(exc.details)})
                result['native_events'].append({'time_s':float(data.time),
                    'physics_tick':completed_steps, 'hits':hits, 'settlement':None,
                    'contact_evidence_error':copy.deepcopy(exc.details)})
                raise
            tick_faults = faults(contact['contacts']) + list(contact['numerical_faults'])
            if contact['contacts']:
                contact_log.append({'time_s': float(data.time), 'interval_start_s': previous_time,
                                    'interval_end_s': float(data.time), 'physics_tick': completed_steps,
                                    'contacts': contact['contacts']})
            event = None
            if hits or contact['verdicts'] or tick_faults:
                event = {'time_s': float(data.time), 'physics_tick': completed_steps,
                         'hits': hits, 'contact_verdicts': contact['verdicts'],
                         'numerical_faults': tick_faults, 'settlement': None}
                result['native_events'].append(event)
            try:
                settlement = settle_training_tick(completed_steps, contacts=contact['verdicts'],
                    hit_agents=[h['agent'] for h in hits], numerical_faults=tick_faults)
            except TrainingEnvironmentError as exc:
                if event is not None: event['settlement_error'] = copy.deepcopy(exc.details)
                raise
            if event is not None: event['settlement'] = settlement
            if settlement['terminal'] and first_terminal is None:
                first_terminal = copy.deepcopy(settlement)
                first_terminal['time_s'] = float(data.time)
            if contact['verdicts']:
                failures.append('foul_ends_physical_diagnostic')
            for slot, program in programs.items():
                if data.time >= duration_s-.1-1e-9:
                    mapping = scene.agents[slot]
                    final_tail[slot].append(bool(
                        np.max(np.abs(data.qpos[mapping['qpos_ids']]-program.curve.sample(program.curve.duration_s))) <= .05
                        and np.max(np.abs(data.qvel[mapping['dof_ids']])) <= .10))
            previous = current
            record()
            if time.monotonic()-last_progress >= 10.:
                result['completed_time_s'] = float(data.time)
                save()
                if progress: progress(f'{output.name}: {data.time:.3f}/{duration_s:.3f} simulated seconds', flush=True)
                last_progress = time.monotonic()
    except TrainingEnvironmentError as exc:
        failures.append('training_contact_or_numeric_evidence_error')
        result['failure_details'] = copy.deepcopy(exc.details)
    except ValueError as exc:
        failures.append('runtime_validation_error')
        result['failure_details'] = str(exc)
    if not rows or rows[-1][0] != float(data.time): record()
    unchanged = all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == h for p,h in hashes.items())
    if not unchanged: failures.append('source_changed_during_trial')
    if not failures: scorer.finish(float(data.time))
    settled = {s:bool(final_tail[s] and all(final_tail[s])) for s in SLOTS}
    result.update(status='completed' if not failures and completed_steps == total_steps else 'failed',
        failure=failures or None, completed_time_s=float(data.time), physics_steps=completed_steps,
        first_settlement=first_terminal,
        official_match_result=first_terminal if first_terminal is not None else (
            {'status': 'timeout_draw', 'match_winner': None, 'rl_reward': {s:0. for s in SLOTS}}
            if not failures else None), geometry_first_touch_diagnostic=scorer.round_result,
        initial_overlap_events=scorer.initial_overlap_events, all_target_entries=scorer.all_entries,
        minimum_target_clearance_m=scorer.minimum_clearance_m, entry_ready=entry_ready,
        final_settled=settled, maximum_joint_tracking_error_rad=worst_error.tolist(),
        maximum_joint_reference_error_rad=worst_reference_error.tolist(),
        maximum_actual_joint_violation_rad=max_range_violation, maximum_penetration_m=max_penetration,
        maximum_joint_limit_force_nm=max_limit_force, contact_monitor=monitor.summary(),
        native_execution_pass=not failures and completed_steps == total_steps and all(settled.values()),
        attack_or_defence_certified=False,
        elapsed_wall_s=time.monotonic()-wall_start,
        source_unchanged=unchanged)
    np.savez_compressed(output/'trace.npz', **{name:np.asarray([r[i] for r in rows]) for i,name in
        enumerate(('time_s','qpos','qvel','target_qpos','ctrl','actuator_force_nm','desired_reference_qpos'))})
    (output/'contacts.json').write_text(json.dumps(_finite_json(contact_log), indent=2)+'\n')
    save()
    return result
