"""Read-only saved-frame target sensitivity audit, NOT a new match scorer.

Run from the repository root with ``python -m tools.audit_target_candidates``.
Only actual recorded qpos is replayed through mj_forward; there is no mj_step,
resizing, controller, contact modification or rewriting of source evidence.
Negative sampled occupancy does NOT establish continuous non-intersection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SLOTS = ('a1', 'a2')
DEFAULT_TRIALS = {
    'original_attack_hold': 'outputs/a1_common_ready_native_v4/baseline',
    'original_observed_parry': 'outputs/a1_glancing_feedback_v1/online',
    'down_attack_hold': 'outputs/a1_vertical_native_v1/down_10cm',
    'up_attack_hold': 'outputs/a1_vertical_native_v1/up_10cm',
    'up_low_parry': 'outputs/a1_vertical_parry_native_v1/up_low',
    'down_high_parry_failed': 'outputs/a1_vertical_parry_native_v1/down_high',
    'parry_counter_response': 'outputs/a1_counter_chain_v1/counter_response',
}
# These are arbitrary sensitivity probes, NOT anatomical recommendations.
CANDIDATES = {
    'original_ball': ((1.05, .06),),
    'compact_torso_probe': ((.90, .12), (1.05, .12), (1.20, .12)),
    'wide_torso_probe': ((.85, .16), (1.05, .16), (1.25, .16)),
    'compact_plus_head_probe': ((.90, .12), (1.05, .12), (1.20, .12), (1.45, .10)),
}


def candidate_geometry(original_centers):
    """Keep each defender's original X/Y, with symmetric candidate heights."""
    original = np.asarray(original_centers, dtype=float)
    if original.shape != (2, 3) or not np.isfinite(original).all():
        raise ValueError('Two finite original target centers required')
    result = {}
    for name, zones in CANDIDATES.items():
        centers = np.repeat(original[:, None, :], len(zones), axis=1)
        centers[:, :, 2] = [z for z, radius in zones]
        result[name] = (centers, np.array([radius for z, radius in zones]))
    return result


def sphere_clearances(starts, ends, blade_radii, centers, sphere_radii):
    """Minimum finite-cylinder surface clearance for each target sphere.

    starts/ends [owner, section, 3], radii [owner, section], centers
    [owner, sphere, 3], sphere_radii [sphere]. Preserves flat cylinder ends.
    """
    starts, ends, blade_radii, centers, sphere_radii = map(np.asarray,
        (starts, ends, blade_radii, centers, sphere_radii))
    if starts.ndim != 3 or starts.shape[-1] != 3 or ends.shape != starts.shape \
            or blade_radii.shape != starts.shape[:2] or centers.ndim != 3 \
            or centers.shape[0] != starts.shape[0] or centers.shape[-1] != 3 \
            or sphere_radii.shape != (centers.shape[1],) \
            or any(not np.isfinite(x).all() for x in (starts, ends, blade_radii, centers, sphere_radii)) \
            or np.any(blade_radii <= 0) or np.any(sphere_radii <= 0):
        raise ValueError('Finite matching blade/target geometry with positive radii required')
    axis = ends-starts
    lengths = np.linalg.norm(axis, axis=-1)
    if np.any(lengths <= 1e-12):
        raise ValueError('Nondegenerate finite cylinders required')
    direction = axis/lengths[:, :, None]
    offset = centers[:, :, None, :]-starts[:, None, :, :]
    axial = np.sum(offset*direction[:, None, :, :], axis=-1)
    radial = np.linalg.norm(offset-axial[:, :, :, None]*direction[:, None, :, :], axis=-1)
    radial_gap = np.maximum(0., radial-blade_radii[:, None, :])
    end_gap = np.maximum(np.maximum(0., -axial), axial-lengths[:, None, :])
    return np.min(np.hypot(radial_gap, end_gap), axis=-1)-sphere_radii[None, :]


def summarize_samples(times, clearance, original_clearance, *, first_recorded_entry_for_attacker_s=None,
                      source_official_terminal_time_s=None, source_fault_or_foul_s=None):
    """Sampled union entry episodes, not continuous-time collision events."""
    times, clearance, original = map(np.asarray, (times, clearance, original_clearance))
    if times.ndim != 1 or len(times) < 2 or clearance.shape != times.shape \
            or original.shape != times.shape or not np.all(np.diff(times) > 0) \
            or not all(np.isfinite(x).all() for x in (times, clearance, original)):
        raise ValueError('Strictly increasing finite samples and scalar clearances required')
    inside, old_inside = clearance < -1e-10, original < -1e-10
    entry_indices = np.flatnonzero(inside[1:] & ~inside[:-1])+1
    episodes = [{'first_inside_saved_time_s': float(times[i]),
                 'crossing_time_bracket_saved_s': [float(times[i-1]), float(times[i])],
                 'clearance_m': float(clearance[i])} for i in entry_indices]
    def first(mask):
        indices = np.flatnonzero(mask)
        return float(times[indices[0]]) if len(indices) else None
    new = inside & ~old_inside
    result = {
        'initial_clearance_m': float(clearance[0]),
        'initial_overlap_saved_frame': bool(inside[0]),
        'minimum_saved_clearance_m': float(np.min(clearance)),
        'first_inside_saved_time_s': first(inside),
        'inside_saved_frame_count': int(np.count_nonzero(inside)),
        'sampled_entry_episodes': episodes,
        'first_new_inside_vs_original_saved_time_s': first(new),
        'new_inside_vs_original_saved_frame_count': int(np.count_nonzero(new)),
        'adds_hit_to_original_no_sampled_hit': bool(inside.any() and not old_inside.any()),
        'first_recorded_entry_for_attacker_s': first_recorded_entry_for_attacker_s,
        'sampled_entry_episodes_after_first_recorded_attacker_entry': [],
        'first_occupied_saved_time_after_first_recorded_attacker_entry_s': None,
        'source_official_terminal_time_s': source_official_terminal_time_s,
        'sampled_entry_episodes_after_source_official_terminal': [],
        'first_occupied_saved_time_after_source_official_terminal_s': None,
        'source_first_recorded_attacker_entry_after_official_terminal': None,
        'first_occupied_saved_time_at_or_after_source_fault_or_foul_s': None,
        'geometric_overlap_is_legal_match_success': False,
        'negative_result_is_continuous_miss_certificate': False,
    }
    if first_recorded_entry_for_attacker_s is not None:
        result['sampled_entry_episodes_after_first_recorded_attacker_entry'] = [
            event for event in episodes if event['first_inside_saved_time_s'] > first_recorded_entry_for_attacker_s]
        result['first_occupied_saved_time_after_first_recorded_attacker_entry_s'] = first(
            inside & (times > first_recorded_entry_for_attacker_s))
    if source_official_terminal_time_s is not None:
        result['sampled_entry_episodes_after_source_official_terminal'] = [
            event for event in episodes if event['first_inside_saved_time_s'] > source_official_terminal_time_s]
        result['first_occupied_saved_time_after_source_official_terminal_s'] = first(
            inside & (times > source_official_terminal_time_s))
        if first_recorded_entry_for_attacker_s is not None:
            result['source_first_recorded_attacker_entry_after_official_terminal'] = bool(
                first_recorded_entry_for_attacker_s > source_official_terminal_time_s)
    if source_fault_or_foul_s is not None:
        result['first_occupied_saved_time_at_or_after_source_fault_or_foul_s'] = first(
            inside & (times >= source_fault_or_foul_s))
    return result


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_trial(trial, output):
    trial, output = Path(trial).resolve(), Path(output)
    inputs = {name: _hash(trial/name) for name in ('scene.xml', 'trace.npz', 'summary.json')}
    summary = json.loads((trial/'summary.json').read_text())
    with np.load(trial/'trace.npz', allow_pickle=False) as trace:
        times, qpos = trace['time_s'].copy(), trace['qpos'].copy()
    model = mujoco.MjModel.from_xml_path(str(trial/'scene.xml'))
    data = mujoco.MjData(model)
    if times.ndim != 1 or len(times) < 2 or qpos.shape != (len(times), model.nq) \
            or not np.isfinite(times).all() or not np.isfinite(qpos).all() \
            or not np.all(np.diff(times) > 0):
        raise ValueError('Finite actual qpos at increasing saved times required')
    # Centers below are mapped by attacking owner to the opposing target.
    original_centers = []
    for defender in reversed(SLOTS):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, defender+'_target')
        if sid < 0 or model.site_type[sid] != int(mujoco.mjtGeom.mjGEOM_SPHERE):
            raise ValueError('Original virtual sphere targets must be present')
        if abs(model.site_size[sid, 0]-.06) > 1e-9 or abs(model.site_pos[sid, 2]-1.05) > 1e-9:
            raise ValueError('This explicitly declared probe requires original r=.06,z=1.05')
        original_centers.append(model.site_pos[sid].copy())
    geometry = candidate_geometry(original_centers)
    gids = np.array([[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
        f'{slot}_shinai_blade_collision_{i}') for i in range(3)] for slot in SLOTS])
    if np.any(gids < 0) or np.any(model.geom_type[gids] != int(mujoco.mjtGeom.mjGEOM_CYLINDER)):
        raise ValueError('Three native finite blade cylinders required per owner')
    blade_radii, half_lengths = model.geom_size[gids, 0], model.geom_size[gids, 1]
    clearances = {name: np.empty((len(times), 2)) for name in geometry}
    zone_minima = {name: np.full((2, len(radii)), np.inf) for name, (_, radii) in geometry.items()}
    for index, (now, q) in enumerate(zip(times, qpos)):
        data.qpos[:] = q
        data.time = float(now)
        mujoco.mj_forward(model, data)  # Read-only geometry reconstruction, NO mj_step.
        direction = data.geom_xmat[gids].reshape(2, 3, 3, 3)[:, :, :, 2]
        centers = data.geom_xpos[gids]
        starts = centers-half_lengths[:, :, None]*direction
        ends = centers+half_lengths[:, :, None]*direction
        for name, (sphere_centers, sphere_radii) in geometry.items():
            zones = sphere_clearances(starts, ends, blade_radii, sphere_centers, sphere_radii)
            clearances[name][index] = np.min(zones, axis=1)
            zone_minima[name] = np.minimum(zone_minima[name], zones)
    source_unchanged = all(_hash(trial/name) == digest for name, digest in inputs.items())
    if not source_unchanged:
        raise RuntimeError('Source evidence changed during audit')
    requested = summary.get('requested_duration_s')
    last_matches = requested is not None and abs(float(requested)-float(times[-1])) <= 1e-8
    source_complete = bool(summary.get('status') == 'completed' and summary.get('native_execution_pass')
                           and summary.get('evidence_complete') and last_matches)
    faults = [event for event in summary.get('native_events', [])
              if event.get('numerical_faults') or event.get('contact_verdicts')]
    fault_times = [float(event['time_s']) for event in faults if 'time_s' in event]
    first_fault = min(fault_times) if fault_times else None
    terminal = summary.get('first_settlement')
    terminal_time = float(terminal['time_s']) if isinstance(terminal, dict) and terminal.get('time_s') is not None else None
    result = {'source_trial': str(trial), 'input_sha256': inputs,
        'input_unchanged': source_unchanged,
        'source_summary': {key: summary.get(key) for key in ('status', 'native_execution_pass',
            'evidence_complete', 'failure', 'requested_duration_s', 'completed_time_s', 'completion')},
        'source_complete_native_record': source_complete,
        'source_first_settlement_unchanged_not_rescored': summary.get('first_settlement'),
        'source_official_terminal_time_s': terminal_time,
        'source_native_fault_or_foul_events_unchanged': faults,
        'first_source_fault_or_foul_s': first_fault,
        'sampling': {'frame_count': len(times), 'start_s': float(times[0]), 'end_s': float(times[-1]),
            'minimum_interval_s': float(np.min(np.diff(times))),
            'maximum_interval_s': float(np.max(np.diff(times))), 'every_saved_frame_used': True},
        'original_target_centers_by_attacker_m': {s: list(map(float, c)) for s,c in zip(SLOTS, original_centers)},
        'native_blade_radii_by_attacker_m': {s: blade_radii[i].tolist() for i,s in enumerate(SLOTS)},
        'candidates': {}, 'admitted_to_training': False,
        'continuous_miss_or_successful_defence_certificate': False}
    official_entries = summary.get('all_target_entries', [])
    for name, (sphere_centers, sphere_radii) in geometry.items():
        result['candidates'][name] = {
            'geometry': {slot: [{'center_m': center.tolist(), 'radius_m': float(radius)}
                for center,radius in zip(sphere_centers[i], sphere_radii)] for i,slot in enumerate(SLOTS)},
            'interpretation': 'original control' if name == 'original_ball' else
                'arbitrary sensitivity probe; not a proposed human body dimension',
            'attackers': {}}
        for owner_index, slot in enumerate(SLOTS):
            hits = [float(e['time_s']) for e in official_entries if e.get('agent') == slot]
            details = summarize_samples(times, clearances[name][:, owner_index],
                clearances['original_ball'][:, owner_index],
                first_recorded_entry_for_attacker_s=min(hits) if hits else None,
                source_official_terminal_time_s=terminal_time,
                source_fault_or_foul_s=first_fault)
            details['source_recorded_entries_for_attacker'] = [
                {key: entry.get(key) for key in ('time_s', 'decides_round', 'step_invalid')}
                for entry in official_entries if entry.get('agent') == slot]
            details['minimum_saved_clearance_each_sphere_m'] = zone_minima[name][owner_index].tolist()
            result['candidates'][name]['attackers'][slot] = details
    np.savez_compressed(output, time_s=times, **{name+'_clearance_m': values for name,values in clearances.items()})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='outputs/rules_review_v1/target_geometry')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Preserve prior audit evidence; use a new output path')
    output.mkdir(parents=True)
    began = time.monotonic()
    report = {'schema_version': 2, 'status': 'running',
        'method': 'saved actual qpos -> mj_forward -> finite cylinder / sphere union clearance',
        'scope': 'offline arbitrary target sensitivity; no production rule changes or defence admission',
        'limitations': ['No mj_step or new physical interaction is computed.',
            'Every saved frame is tested; unsampled within-frame motion is not reconstructed.',
            'A positive overlap is a saved-state witness, not necessarily a legal scored hit.',
            'A negative sampled result is NOT a continuous miss or successful defence certificate.',
            'Candidate dimensions are arbitrary sensitivity probes, not anatomical recommendations.',
            'Late occupied frames and new entry episodes are distinguished; occupancy is not repeated scoring.',
            'No new match winner is assigned. Source settlements, faults and fouls are retained unchanged.',
            'A first recorded entry per attacker can be a shadow entry after the other side ended the match; it is not called an official hit.',
            'The replay does not rerun physical_observation or policy decisions for new targets; new closed-loop trials remain required.'],
        'tool_sha256': _hash(__file__), 'scorer_geometry_sha256': _hash(ROOT/'sim/fencing_rules.py'),
        'trials': {}}
    report_path = output/'report.json'
    def save(): report_path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    save()
    try:
        for name, path in DEFAULT_TRIALS.items():
            report['trials'][name] = audit_trial(ROOT/path, output/(name+'_clearances.npz'))
            save()
            print(f'{name}: complete saved-state geometry audit', flush=True)
        report['status'] = 'completed'
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['elapsed_wall_s'] = time.monotonic()-began
        save()


if __name__ == '__main__': main()
