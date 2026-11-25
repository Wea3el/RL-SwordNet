"""Read actual grip-to-tip orientation, independently of camera and commands.

Upright is an A1 style requirement, not a scoring rule for either agent.  The
declared active interval and every loaded cross-weapon contact are checked.
This read-only audit does not manufacture a defence or training certificate.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import mujoco
import numpy as np


def upright_metrics(grip, tip, min_elevation_deg=25., min_grip_height_m=.35):
    grip, tip = np.asarray(grip, float), np.asarray(tip, float)
    if (grip.ndim != 2 or grip.shape[1:] != (3,) or grip.shape != tip.shape
            or not len(grip) or not np.isfinite([grip, tip]).all()):
        raise ValueError('Nonempty finite matching N by 3 grip and tip positions required')
    if not (np.isfinite(min_elevation_deg) and 0 < min_elevation_deg < 90
            and np.isfinite(min_grip_height_m) and min_grip_height_m >= 0):
        raise ValueError('Explicit positive upward elevation and valid grip height required')
    delta = tip-grip
    lengths = np.linalg.norm(delta, axis=1)
    if np.any(lengths < 1e-8):
        raise ValueError('A zero-length sword has no signed axis')
    z = delta[:, 2]/lengths
    elevation = np.rad2deg(np.arcsin(np.clip(z, -1., 1.)))
    up = z >= np.sin(np.deg2rad(min_elevation_deg))-1e-9
    height = grip[:, 2] >= min_grip_height_m-1e-9
    return {
        'minimum_elevation_deg': float(elevation.min()),
        'maximum_elevation_deg': float(elevation.max()),
        'minimum_grip_height_m': float(grip[:, 2].min()),
        'signed_upward_pass': bool(up.all()), 'grip_height_pass': bool(height.all()),
        'pass': bool(up.all() and height.all()),
        'failed_local_indices': np.flatnonzero(~(up & height)).tolist(),
        'sample_count': len(grip),
    }


def loaded_contact_indices(times, contacts):
    """Retain every nonzero-load weapon contact, including outside phase labels."""
    indices = set()
    for row in contacts:
        loaded = any(c.get('category') in ('cross_blade_blade', 'cross_blade_guard')
                     and float(c.get('normal_force_n', 0.)) > 0. for c in row['contacts'])
        if loaded:
            t = float(row['time_s'])
            i = int(np.argmin(abs(times-t)))
            if abs(times[i]-t) > 1e-7:
                raise ValueError('Loaded contact is missing its exact recorded physical state')
            indices.add(i)
    return np.array(sorted(indices), dtype=int)


def audit(trial, output, *, defender, active_start_s, active_end_s,
          min_elevation_deg=25., min_grip_height_m=.35, comparison=None):
    trial, output = Path(trial).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve earlier orientation evidence')
    if defender not in ('a1', 'a2'):
        raise ValueError('Explicit physical defender slot required')
    paths = {name: trial/name for name in ('scene.xml', 'trace.npz', 'summary.json', 'contacts.json')}
    raw = {name: path.read_bytes() for name, path in paths.items()}
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in raw.items()}
    with np.load(io.BytesIO(raw['trace.npz']), allow_pickle=False) as trace:
        times, qpos = trace['time_s'].copy(), trace['qpos'].copy()
    model = mujoco.MjModel.from_xml_path(str(paths['scene.xml']))
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0) or qpos.shape != (len(times), model.nq)
            or not np.isfinite(qpos).all()):
        raise ValueError('Finite, strictly ordered actual physical state trace required')
    if not (np.isfinite([active_start_s, active_end_s]).all()
            and times[0] <= active_start_s < active_end_s <= times[-1]+1e-8):
        raise ValueError('Declare a nonempty active interval inside the complete recorded trace')
    active = np.flatnonzero((times >= active_start_s-1e-9) & (times <= active_end_s+1e-9))
    contact = loaded_contact_indices(times, json.loads(raw['contacts.json']))
    indices = np.union1d(active, contact)
    data = mujoco.MjData(model)
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, defender+'_shinai_'+name)
           for name in ('grip', 'tip')]
    if min(ids) < 0:
        raise ValueError('Named physical grip and tip sites required')
    grips, tips = {}, {}
    for i in indices:
        data.qpos[:] = qpos[i]
        mujoco.mj_kinematics(model, data)
        grips[int(i)], tips[int(i)] = data.site_xpos[ids[0]].copy(), data.site_xpos[ids[1]].copy()
    def metrics(selected):
        if not len(selected):
            return {'pass': False, 'sample_count': 0, 'reason': 'No loaded weapon contact recorded'}
        result = upright_metrics([grips[int(i)] for i in selected], [tips[int(i)] for i in selected],
                                 min_elevation_deg, min_grip_height_m)
        result['first_failed_time_s'] = (float(times[selected[result['failed_local_indices'][0]]])
                                         if result['failed_local_indices'] else None)
        return result
    active_result, contact_result = metrics(active), metrics(contact)
    summary = json.loads(raw['summary.json'])
    pair_bound = False
    pair_hash = None
    comparison_unchanged = comparison is None
    if comparison is not None:
        pair_bytes = Path(comparison).read_bytes()
        pair_hash = hashlib.sha256(pair_bytes).hexdigest()
        pair = json.loads(pair_bytes)
        saved = pair.get('defended_triangle_audit', {}).get('native_trace_source_sha256', {})
        pair_bound = bool(pair.get('status') == 'qualified'
            and pair.get('successful_physical_defence') is True
            and pair.get('attacker_slot') == ('a1' if defender == 'a2' else 'a2')
            and pair.get('matching_checks') and all(pair['matching_checks'].values())
            and all(saved.get(name) == hashes[name] for name in ('scene.xml', 'trace.npz', 'summary.json')))
        comparison_unchanged = Path(comparison).read_bytes() == pair_bytes
    unchanged = all(path.read_bytes() == raw[name] for name, path in paths.items())
    orientation_pass = bool(active_result['pass'] and contact_result['pass'] and unchanged)
    native_clean = summary.get('native_execution_pass') is True
    result = {
        'schema_version': 1, 'trial': str(trial), 'defender_slot': defender,
        'signed_axis': 'world normalized shinai_tip minus shinai_grip; camera independent',
        'min_elevation_deg': min_elevation_deg, 'min_grip_height_m': min_grip_height_m,
        'active_interval_s': [active_start_s, active_end_s],
        'all_active_recorded_frames': active_result,
        'all_nonzero_loaded_weapon_contact_frames': contact_result,
        'actual_orientation_pass': orientation_pass,
        'independent_qualified_defence_bound': pair_bound,
        'qualified_upright_example': bool(orientation_pass and pair_bound and native_clean and comparison_unchanged),
        'source_native_execution_pass': native_clean,
        'source_sha256': hashes, 'comparison_sha256': pair_hash, 'source_unchanged': unchanged,
        'comparison_unchanged': comparison_unchanged,
        'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'reference_or_command_used': False, 'new_physics_steps': 0,
        'scoring_changed': False, 'a2_legal_orientation_restricted': False,
        'admitted_to_training': False,
    }
    output.mkdir(parents=True)
    (output/'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trial', required=True); p.add_argument('--output', required=True)
    p.add_argument('--defender', required=True, choices=('a1', 'a2'))
    p.add_argument('--active-start', type=float, required=True)
    p.add_argument('--active-end', type=float, required=True)
    p.add_argument('--min-elevation', type=float, default=25.)
    p.add_argument('--min-grip-height', type=float, default=.35)
    p.add_argument('--comparison')
    a = p.parse_args()
    result = audit(a.trial, a.output, defender=a.defender,
        active_start_s=a.active_start, active_end_s=a.active_end,
        min_elevation_deg=a.min_elevation, min_grip_height_m=a.min_grip_height,
        comparison=a.comparison)
    print(json.dumps({key: result[key] for key in (
        'actual_orientation_pass', 'independent_qualified_defence_bound',
        'qualified_upright_example', 'source_native_execution_pass',
        'source_unchanged', 'admitted_to_training')}, indent=2))
