"""Actuator-aware candidate revision; no controller, score or physics changes.

Penalizes motion at weak J0 and asks other joints to perform an interception.
The generated curves still need native paired validation through new_defence.
"""
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


def build(source, output, *, start=.77):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Keep previous weighted candidates')
    data = json.loads(Path(source).read_text())
    ready = np.asarray(data['ready_qpos_rad'])
    model = build_limited_floor().model
    state = mujoco.MjData(model)
    bounds = model_joint_ranges_rad(load_joint_constraints()) + np.array([.02, -.02])
    bounds[0] = [ready[0] - .08, ready[0] + .08]
    weight = np.ones(7)
    weight[0] = 8.
    candidates = []
    for row in data['selected']:
        grip, axis, _ = tool_fk(model, state, row['qpos_rad'])
        target = grip + row['distance_m'] * axis
        def objective(z):
            q, along = z[:7], z[7]
            g, a, _ = tool_fk(model, state, q)
            return np.r_[10 * (g + along * a - target), .05 * weight * (q - ready),
                         .003 * (along - row['distance_m'])]
        result = least_squares(objective, np.r_[ready, row['distance_m']],
            bounds=(np.r_[bounds[:, 0], .06], np.r_[bounds[:, 1], .77]), max_nfev=250)
        q, along = result.x[:7], result.x[7]
        g, a, _ = tool_fk(model, state, q)
        error = float(np.linalg.norm(g + along * a - target))
        if error > .005:
            continue
        shift = np.asarray(row['sweep_grip_shift_m'])
        desired_grip = g + shift
        def sweep_objective(x):
            gg, aa, _ = tool_fk(model, state, x)
            return np.r_[10 * (gg - desired_grip), 2 * (aa - a), .03 * weight * (x - q)]
        swipe = least_squares(sweep_objective, q, bounds=(bounds[:, 0], bounds[:, 1]), max_nfev=250)
        approach = max(.3, float(np.max(np.abs(q - ready))) * 1.875 / 2.)
        sweep = max(.5, float(np.max(np.abs(swipe.x - q))) * 1.875 / 2.)
        hold = approach + .3
        release = max(hold + sweep + .25, 2.35 - start)
        recovery = max(1., float(np.max(np.abs(swipe.x - ready))) * 1.875 / 2.)
        times = [0., approach, hold, hold + sweep, release, release + recovery]
        poses = [ready, q, q, swipe.x, swipe.x, ready]
        curve = C2JointSpline(times, poses)
        candidates.append(dict(row, id=len(candidates), original_id=row['id'], start_s=start,
            approach_s=approach, knot_time_s=times, knot_qpos=np.asarray(poses).tolist(),
            qpos_rad=q.tolist(), distance_m=float(along), actuator_aware=True,
            j0_radius_rad=.08, j0_objective_weight=8., position_error_m=error,
            new_geometric_arrival_s=start + approach,
            arrival_after_original_hit=(start + approach > data['baseline_hit_s']),
            max_reference_speed_rad_s=max(curve.analytic_extrema()['peak_speed_rad_s'])))
    data.update(selected=candidates, actuator_aware=True, parent_candidates=str(source),
                note='Proposals only. Actual native contact before hit and full recovery still required.')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_finite_json(data), indent=2) + '\n')
    print(json.dumps([{k:c[k] for k in ('id','original_id','new_geometric_arrival_s','arrival_after_original_hit','max_reference_speed_rad_s')} for c in candidates], indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--start', type=float, default=.77)
    args = p.parse_args()
    build(args.source, args.output, start=args.start)
