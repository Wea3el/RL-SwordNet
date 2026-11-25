"""Finite independent forward-neutral feasibility search; never a certificate."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import mujoco
import numpy as np

from .assembly import ROOT, _finite_json
from .balanced_orientation_geometry import static_row
from .joint_constraints import load_joint_constraints, model_joint_ranges_rad
from .limited_duel import build_limited_duel
from .limited_model import build_limited_floor
from .multiguard_geometry import GripAxisSolver
from .triangle_task import TriangleGeometry


def run(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Preserve previous independent search evidence")
    source = ROOT / "outputs/orientation_v2_geometry/search01/report.json"
    baseline = ROOT / "outputs/orientation_v2_geometry/corridor05/report.json"
    paths = [source, baseline, Path(__file__).resolve(),
             ROOT / "sim/balanced_orientation_geometry.py", ROOT / "sim/multiguard_geometry.py",
             ROOT / "config/joint_constraints_observed.json"]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    old = json.loads(source.read_text())
    previous = json.loads(baseline.read_text())
    constraints = load_joint_constraints()
    physical = model_joint_ranges_rad(constraints)
    bounds = physical + [.03, -.03]
    model = build_limited_floor(constraints).model
    data = mujoco.MjData(model)
    geometry = TriangleGeometry.from_mapping({k: old["scoring"][k] for k in ("base_distance", "regions")})
    previous_q = np.asarray(previous["guards"]["center"]["qpos_rad"])
    scene = build_limited_duel(geometry.base_distance, initial_qpos=previous_q, constraints=constraints)
    solver = GripAxisSolver(model, bounds)
    cap = np.max(abs(model.actuator_forcerange), axis=1) * abs(model.actuator_gear[:, 0])
    contacts = [row for rows in old["contact_candidates"].values() for row in rows
                if row["qpos_rad"][2] < np.pi]
    contact = min(contacts, key=lambda row: row["max_static_gravity_ratio"])
    rng = np.random.default_rng(20261125)
    random = rng.uniform(bounds[:, 0], bounds[:, 1])
    random[2] = rng.uniform(bounds[2, 0], np.pi)
    seeds = [("negative_contact_low_J2", np.asarray(contact["qpos_rad"])),
             ("previous_B", previous_q), ("deterministic_random_low_J2", random)]
    goals = list(itertools.product((.40, .55, .70), (-.15, 0., .15), (.60, .80, 1.0)))
    goals.sort(key=lambda p: (abs(p[1]), np.linalg.norm(np.asarray(p) - [.55, 0., .8])))
    report = {
        "schema_version": 1, "scope": "Finite independent centered forward-neutral IK search",
        "scoring": old["scoring"], "source_sha256": hashes,
        "goal_count_budget": 27, "seed_count": 3, "max_nfev_per_fit": 100,
        "joint_search_margin_rad": .03, "preferred_min_joint_margin_rad": .08,
        "preferred_max_static_torque_ratio": .9, "required_low_J2_less_than_rad": float(np.pi),
        "grip_position_tolerance_m": .01, "shaft_angle_tolerance_deg": 2.,
        "status": "running", "trials": [], "candidates": [],
        "native_dynamics_tested": False, "admitted_to_training": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    def save():
        (output / "report.json").write_text(json.dumps(_finite_json(report), indent=2) + "\n")
    save()
    stopped = False
    for goal in goals:
        for label, seed in seeds:
            q, fit = solver.solve(goal, [1., 0., 0.], seed, regularization=.0015, iterations=100)
            # The pair is checked with this SAME neutral pose in both actors.
            metrics = static_row(model, data, scene, geometry, q, q, "guard", cap)
            minimum_margin = float(np.minimum(q - physical[:, 0], physical[:, 1] - q).min())
            low_j2 = bool(q[2] < np.pi)
            precise = bool(fit["position_error_m"] <= .01 and fit["axis_error_deg"] <= 2.)
            candidate = bool(precise and low_j2 and minimum_margin >= .03 - 1e-9
                             and metrics["static_hold_under_cap"]
                             and metrics["static_pair_screen"]["zero_overlap_start_pass"])
            preferred = bool(candidate and minimum_margin >= .08
                             and metrics["max_static_gravity_ratio"] <= .9)
            row = {"goal_grip_m": list(goal), "seed_label": label, "qpos_rad": q.tolist(),
                   "qpos_deg": np.rad2deg(q).tolist(), "minimum_joint_margin_rad": minimum_margin,
                   "low_J2": low_j2, "candidate_pass": candidate, "preferred_candidate": preferred,
                   **fit, **metrics}
            report["trials"].append(row)
            if candidate:
                report["candidates"].append(row)
            print("neutral", len(report["trials"]), goal, label, "candidate", candidate,
                  "preferred", preferred, "fit", round(fit["position_error_m"], 4),
                  round(fit["axis_error_deg"], 2), "margin", round(minimum_margin, 4),
                  "gravity", round(metrics["max_static_gravity_ratio"], 3), flush=True)
            save()
            if preferred:
                stopped = True
                break
        if stopped:
            break
    report["source_unchanged"] = all(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == hashes[str(path)]
                                     for path in paths)
    report["status"] = "preferred_candidate_found_early" if stopped else "finite_budget_exhausted"
    if report["candidates"]:
        chosen = min(report["candidates"], key=lambda row: (not row["preferred_candidate"],
                     abs(row["actual_grip_m"][1]), row["max_static_gravity_ratio"],
                     -row["minimum_joint_margin_rad"]))
        report["selected_candidate"] = chosen
    report["limitations"] = [
        "Static IK, holding torque and zero-overlap screen only; no native hold or attack/defence validation.",
        "Low-J2 and three-seed finite search do not prove global reachability or optimality.",
        "Changing the shared neutral invalidates previous opponent-baseline and transition certificates.",
    ]
    save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({"status": result["status"], "trials": len(result["trials"]),
                      "candidates": len(result["candidates"]), "source_unchanged": result["source_unchanged"]}))


if __name__ == "__main__":
    main()
