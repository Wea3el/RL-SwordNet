"""Direction-aware parry proposals after observed wrong-side contact forces."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from .assembly import _finite_json
from .joint_constraints import load_joint_constraints, model_joint_ranges_rad
from .limited_model import build_limited_floor
from .motion_spline import C2JointSpline
from plan1_skills.matching import tool_fk


def build(source, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Keep previous direction proposals')
    original = json.loads(Path(source).read_text())
    row = original['selected'][0]
    ready = np.asarray(original['ready_qpos_rad'])
    model = build_limited_floor().model
    data = mujoco.MjData(model)
    bounds = model_joint_ranges_rad(load_joint_constraints()) + np.array([.02, -.02])
    bounds[0] = [ready[0] - .10, ready[0] + .10]
    weight = np.ones(7)
    weight[0] = 6.
    grip, axis, _ = tool_fk(model, data, row['qpos_rad'])
    old_point = grip + row['distance_m'] * axis
    results = []
    for offset in ([-.035, .02, -.035], [-.055, .03, -.055]):
        target = old_point + offset
        def objective(z):
            q, along = z[:7], z[7]
            g, a, _ = tool_fk(model, data, q)
            return np.r_[10 * (g + along * a - target), .045 * weight * (q - ready),
                         .2 * (a - axis), .005 * (along - row['distance_m'])]
        fit = least_squares(objective, np.r_[ready, row['distance_m']],
            bounds=(np.r_[bounds[:, 0], .06], np.r_[bounds[:, 1], .77]), max_nfev=250)
        q, along = fit.x[:7], fit.x[7]
        g, a, _ = tool_fk(model, data, q)
        error = float(np.linalg.norm(g + along * a - target))
        if error > .01:
            continue
        lifted_target = g + along * a + np.array([.04, 0., .1])
        def lift_objective(x):
            gg, aa, _ = tool_fk(model, data, x)
            return np.r_[10 * (gg + along * aa - lifted_target), .3 * (aa - a),
                         .03 * weight * (x - q)]
        lift = least_squares(lift_objective, q, bounds=(bounds[:, 0], bounds[:, 1]), max_nfev=250)
        approach = max(.3, float(np.max(np.abs(q - ready))) * 1.875 / 2.)
        sweep = max(.3, float(np.max(np.abs(lift.x - q))) * 1.875 / 2.)
        release = max(approach + sweep + .7, 2.35 - .77)
        recovery = max(1., float(np.max(np.abs(lift.x - ready))) * 1.875 / 2.)
        t = [0., approach, approach + sweep, release, release + recovery]
        qs = [ready, q, lift.x, lift.x, ready]
        curve = C2JointSpline(t, qs)
        results.append(dict(row, id=len(results), start_s=.77, approach_s=approach,
            qpos_rad=q.tolist(), knot_time_s=t, knot_qpos=np.asarray(qs).tolist(),
            direction_revision='opposite-side approach then upward/backward sweep',
            target_offset_local_m=offset, sweep_material_point_offset_local_m=[.04,0.,.1],
            position_error_m=error, distance_m=float(along),
            new_geometric_arrival_s=.77+approach,
            max_reference_speed_rad_s=max(curve.analytic_extrema()['peak_speed_rad_s'])))
    original.update(selected=results, direction_revision=True, parent_candidates=str(source),
        note='Derived from wrong-side observed contact normals. Not a defence certificate.')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_finite_json(original), indent=2) + '\n')
    print(json.dumps([{k:r[k] for k in ('id','new_geometric_arrival_s','max_reference_speed_rad_s')} for r in results], indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    build(a.source, a.output)
