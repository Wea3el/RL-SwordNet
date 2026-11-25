"""Re-screen exact motion references on the current blade geometry.

This is a diagnostic bank, not a training bank. Old guards are namespaced,
not silently aliased onto a new joint branch. Source success certificates do
not transfer. A sampled self-collision check only allows a native experiment;
it does not establish attacks, parries, connectivity or policy reactions.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from .assembly import ROOT, _finite_json
from .blade_geometry import blade_geometry_identity
from .joint_constraints import load_joint_constraints, model_joint_ranges_rad
from .limited_model import build_limited_floor
from .motion_spline import C2JointSpline


SOURCES = (
    ('outputs/a1_chain_geometry/chain12/report.json', '', (
        'cut_down_left_to_right', 'cut_up_right_to_left', 'cut_up_left_to_right',
        'recover_positive_high_to_low', 'switch_negative_to_positive', 'switch_positive_to_negative')),
    ('outputs/multiguard_v1_geometry/bundle13/report.json', 'legacy_', (
        'cut_down_right_to_left', 'cut_down_center', 'thrust_center', 'parry_high')),
    ('outputs/a1_upright_left/search03/report.json', '', ('parry_left_upright_00',)),
    ('outputs/a1_upright_right/early05/report.json', '', ('parry_right_upright_early',)),
    ('outputs/a1_guard_defence_screen/low_transfer06/report.json', '', (
        'switch_positive_low_to_negative_low', 'switch_negative_low_to_positive_low')),
)


def blade_identity(model, prefix=''):
    """Geometry-only identity: excludes robot placement, dynamics and qpos."""
    return blade_geometry_identity(model, prefix)


def build(output, samples=81):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve earlier geometry and failed experiments')
    if type(samples) is not int or samples < 3:
        raise ValueError('At least three whole-reference samples required')
    output.mkdir(parents=True)
    assembly = build_limited_floor()
    model, data = assembly.model, mujoco.MjData(assembly.model)
    bounds = model_joint_ranges_rad(load_joint_constraints())
    identity = blade_identity(model)
    report = {'schema_version': 1, 'status': 'current_geometry_diagnostic_candidates_only',
              'guards': {}, 'actions': {}, 'sources': {}, 'admitted_to_training': False,
              'blade_geometry': identity,
              'blade_profile_contract': assembly.config.get('collision', {}).get('blade_profile_contract'),
              'source_certificates_inherited': False,
              'limitations': [
                  'Old branch guards are distinct; missing connecting edges are not invented.',
                  'Only sampled self/ground checks are performed here, not swept pair or native validation.',
                  'High parry is an unverified candidate, not a newly certified upright defence.',
                  'A diagnostic-eligible action is not admitted to training.',
                  'No arbitrary A2 attack coverage or reactive timing is certified.']}
    for relative, namespace, names in SOURCES:
        path = ROOT / relative
        raw = path.read_bytes()
        source = json.loads(raw)
        if 'scoring' not in report:
            report['scoring'] = copy.deepcopy(source['scoring'])
        elif source['scoring'] != report['scoring']:
            raise ValueError('Source scoring differs; explicit reconciliation required')
        report['sources'][relative] = hashlib.sha256(raw).hexdigest()
        for name in names:
            row = copy.deepcopy(source['actions'][name])
            if name in report['actions']:
                raise ValueError('Duplicate action name')
            for field in ('start_guard', 'end_guard'):
                old = row[field]
                new = namespace + old
                node = copy.deepcopy(source['guards'][old])
                if new in report['guards'] and report['guards'][new]['qpos_rad'] != node['qpos_rad']:
                    raise ValueError('Conflicting exact guard ' + new)
                report['guards'][new] = node
                row[field] = new
            curve = C2JointSpline(row['knot_time_s'], row['knot_qpos'])
            bounded = bool(np.all(curve.bezier_controls >= bounds[:, 0]) and
                           np.all(curve.bezier_controls <= bounds[:, 1]))
            endpoints = all(np.array_equal(curve.knot_qpos[i], report['guards'][row[k]]['qpos_rad'])
                            for k, i in (('start_guard', 0), ('end_guard', -1)))
            collisions = {}
            if bounded:
                for t in np.unique(np.r_[np.linspace(0., curve.duration_s, samples), curve.knot_time_s]):
                    data.qpos[:] = curve.sample(t)
                    data.qvel[:] = 0.
                    mujoco.mj_forward(model, data)
                    for contact in data.contact:
                        if contact.dist < -1e-8:
                            pair = tuple(model.geom(int(g)).name for g in (contact.geom1, contact.geom2))
                            collisions.setdefault('|'.join(pair), {'geoms': pair, 'first_time_s': float(t),
                                                                   'depth_m': float(-contact.dist)})
            row['source_motion'] = {'report': relative, 'sha256': report['sources'][relative],
                                    'name': name, 'guard_namespace': namespace,
                                    'exact_knots_preserved': True}
            row['prior_certificate_inherited'] = False
            row['native_dynamics_tested'] = False
            row['admitted_to_training'] = False
            row['current_geometry_screen'] = {'geometry_sha256': identity['sha256'],
                'analytic_joint_bounds_pass': bounded, 'exact_endpoints_pass': endpoints,
                'sampled_self_contacts': list(collisions.values()), 'sample_count_minimum': samples,
                'scope': 'Diagnostic entry gate only; not attack/parry effectiveness or connectivity.'}
            row['candidate_pass'] = bool(bounded and endpoints and not collisions)
            row['sampled_reference_pass'] = row['candidate_pass']
            report['actions'][name] = row
            print(json.dumps({'action': name, 'diagnostic_eligible': row['candidate_pass'],
                              'self_contact_pairs': len(collisions)}), flush=True)
    report['outgoing_actions'] = {}
    for name, row in report['actions'].items():
        if row['candidate_pass']:
            report['outgoing_actions'].setdefault(row['start_guard'], []).append(name)
    report['disconnected_guard_warning'] = True
    report['source_unchanged'] = all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == digest
                                   for p, digest in report['sources'].items())
    (output / 'report.json').write_text(json.dumps(_finite_json(report), ensure_ascii=False, indent=2)+'\n')
    (output / 'builder_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (output / 'scene.xml').write_text(assembly.spec.to_xml())
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--samples', type=int, default=81)
    args = parser.parse_args()
    build(args.output, args.samples)
