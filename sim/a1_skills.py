"""Closed A1-family finite skill candidates; not validated fencing or RL skills.

A1 denotes the discrete policy family, not a fixed robot slot. Numeric pilot
samples remain available with their original clocks. Runtime interpolation is an
explicit, numpy-only C2 Hermite adaptation; joint-loop closure is a separately
authored quintic segment. Neither operation is original ASE demonstration data.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from plan1_skills.ase_numeric import UPSTREAM_COMMIT
from plan1_skills.library import ROOT
from plan1_skills.matching import tool_fk
from .assembly import _finite_json
from .floor import build_floor_assembly
from .shared_skills import SkillPath, _screen


SKILL_IDS = ("guard", "stab", "slash_left", "parry_left", "parry_right")
DEFAULT_PILOT = ROOT/"outputs/a1_ase_pilot_labeled"
DEFAULT_CONFIG = {
    "schema_version": 1, "policy_family": "A1", "action_space_kind": "finite_skill_selection",
    "robot_slot_independent": True, "a2_continuous_action_space": False,
    "source_time_scale": 1., "control_period_s": .01,
    "minimum_return_s": 2., "return_reference_speed_rad_s": 1.,
    "return_reference_acceleration_rad_s2": 2.,
    "ik_joint_margin_rad": .05,
    "interpolation": "C2 quintic Hermite, shared central-difference knot derivatives; whole-source endpoint velocity/acceleration zero",
    "profile": "executable_candidate_only_not_physical_or_hardware_certification",
}
_MAX_BYTES = 32*1024*1024


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json_file(path):
    raw = Path(path).read_bytes()
    if len(raw) > _MAX_BYTES:
        raise ValueError("Candidate metadata exceeds size bound")
    def invalid(value):
        raise ValueError(f"Nonfinite JSON value: {value}")
    return json.loads(raw, parse_constant=invalid), raw


def _numeric_file(path):
    raw = Path(path).read_bytes()
    if len(raw) > _MAX_BYTES:
        raise ValueError("Numeric candidate exceeds size bound")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        if sum(item.file_size for item in archive.infolist()) > _MAX_BYTES:
            raise ValueError("Expanded numeric candidate exceeds size bound")
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    if any(value.dtype.kind not in "fbiu" or not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Finite numeric-only arrays required; no object/pickle arrays")
    return arrays, raw


def _role(metadata):
    if (metadata.get("policy_family") != "A1" or metadata.get("action_space_kind") != "finite_skill_selection"
            or metadata.get("robot_slot_independent") is not True or metadata.get("a2_continuous_action_space") is not False):
        raise ValueError("A1 finite policy-family metadata required; this is not an A2 action space")
    if metadata.get("admitted_to_training") is not False or metadata.get("may_start_RL") is not False:
        raise ValueError("Unvalidated candidates cannot claim admission or RL readiness")


def _config(config):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if config is not None:
        unknown = set(config)-set(cfg)
        if unknown:
            raise ValueError(f"Unknown A1 library config keys: {sorted(unknown)}")
        cfg.update(copy.deepcopy(config))
    _role(dict(cfg, admitted_to_training=False, may_start_RL=False))
    if cfg["schema_version"] != 1 or cfg["interpolation"] != DEFAULT_CONFIG["interpolation"]:
        raise ValueError("Unsupported A1 interpolation/schema")
    for name in ("source_time_scale", "control_period_s", "minimum_return_s", "return_reference_speed_rad_s", "return_reference_acceleration_rad_s2"):
        value = cfg[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"Positive finite {name} required")
    if cfg["source_time_scale"] < 1. or cfg["ik_joint_margin_rad"] != .05:
        raise ValueError("No implicit source acceleration or joint-margin relaxation allowed")
    return cfg


def _sample_arrays(arrays):
    if not {"time_s", "qpos"}.issubset(arrays):
        raise ValueError("Candidate must contain time_s/qpos")
    t, q = arrays["time_s"], arrays["qpos"]
    if (t.ndim != 1 or len(t) < 2 or q.shape != (len(t), 7)
            or t[0] != 0. or not np.all(np.diff(t) > 0)):
        raise ValueError("Finite ordered zero-based candidate clock and Nx7 qpos required")
    return t, q


def _hermite_coefficients(times, qpos):
    """Shared finite-difference v/a at internal knots, zero only at full ends."""
    h = np.diff(times)
    velocity, acceleration = np.zeros_like(qpos), np.zeros_like(qpos)
    if len(times) > 2:
        velocity[1:-1] = (qpos[2:]-qpos[:-2])/(times[2:]-times[:-2])[:, None]
        left = (qpos[1:-1]-qpos[:-2])/h[:-1, None]
        right = (qpos[2:]-qpos[1:-1])/h[1:, None]
        acceleration[1:-1] = 2*(right-left)/(h[:-1]+h[1:])[:, None]
    c0 = qpos[:-1]
    c1 = h[:, None]*velocity[:-1]
    c2 = .5*h[:, None]**2*acceleration[:-1]
    position = qpos[1:]-c0-c1-c2
    slope = h[:, None]*velocity[1:]-c1-2*c2
    curvature = h[:, None]**2*acceleration[1:]-2*c2
    return np.stack([c0, c1, c2, 10*position-4*slope+.5*curvature,
                     -15*position+7*slope-curvature, 6*position-3*slope+.5*curvature], axis=1)


def _curve_sample(times, knots, coefficients, elapsed, derivative=0):
    if not np.isfinite(elapsed):
        raise ValueError("Own elapsed time must be finite")
    if elapsed <= times[0] or elapsed >= times[-1]:
        if derivative:
            return np.zeros(7)
        return knots[0 if elapsed <= times[0] else -1].copy()
    index = min(int(np.searchsorted(times, elapsed, side="right"))-1, len(coefficients)-1)
    interval = times[index+1]-times[index]
    u = (elapsed-times[index])/interval
    coeff = coefficients[index]
    for _ in range(derivative):
        coeff = np.arange(1, len(coeff))[:, None]*coeff[1:]
    return np.polynomial.polynomial.polyval(u, coeff)/interval**derivative


def _peak_derivatives(times, coefficients):
    peaks = {1: np.zeros(7), 2: np.zeros(7)}
    for index, coeff in enumerate(coefficients):
        interval = times[index+1]-times[index]
        for joint in range(7):
            for order in (1, 2):
                derivative = np.polynomial.polynomial.polyder(coeff[:, joint], order)
                extremum = np.polynomial.polynomial.polyder(derivative)
                roots = np.polynomial.polynomial.polyroots(extremum) if np.any(extremum) else []
                candidates = [0., 1.]+[float(z.real) for z in roots if abs(z.imag) < 1e-8 and 0. < z.real < 1.]
                peak = float(np.max(np.abs(np.polynomial.polynomial.polyval(candidates, derivative))))/interval**order
                peaks[order][joint] = max(peaks[order][joint], peak)
    return {"analytic_peak_reference_speed_rad_s": peaks[1].tolist(),
            "analytic_peak_reference_acceleration_rad_s2": peaks[2].tolist(),
            "note": "reference derivatives, not actual tracking or certified motor limits"}


def _check_curve(times, knots, coefficients):
    """Reject discontinuous/tampered polynomial banks, not only dense samples."""
    h = np.diff(times)[:, None]
    orders = np.arange(6, dtype=float)[None, :, None]
    left_position = coefficients[:, 0]
    right_position = coefficients.sum(axis=1)
    left_velocity = coefficients[:, 1]/h
    right_velocity = (orders*coefficients).sum(axis=1)/h
    left_acceleration = 2*coefficients[:, 2]/h**2
    right_acceleration = (orders*(orders-1)*coefficients).sum(axis=1)/h**2
    vjump = float(np.max(np.abs(right_velocity[:-1]-left_velocity[1:]))) if len(h) > 1 else 0.
    ajump = float(np.max(np.abs(right_acceleration[:-1]-left_acceleration[1:]))) if len(h) > 1 else 0.
    endpoint_error = max(float(np.max(np.abs(left_position-knots[:-1]))), float(np.max(np.abs(right_position-knots[1:]))))
    rest_error = max(float(np.max(np.abs(left_velocity[0]))), float(np.max(np.abs(right_velocity[-1]))),
                     float(np.max(np.abs(left_acceleration[0]))), float(np.max(np.abs(right_acceleration[-1]))))
    if endpoint_error > 1e-9 or max(vjump, ajump, rest_error) > 1e-7:
        raise ValueError("A1 C2 curve does not preserve knots, derivatives or rest endpoints")
    return {"polynomial_max_endpoint_error_rad": endpoint_error,
            "polynomial_max_velocity_jump_rad_s": vjump,
            "polynomial_max_acceleration_jump_rad_s2": ajump,
            "scope": "analytic command continuity within numeric tolerance; no actual-state reset or physical tracking guarantee"}


def _source_segments(segments, source_time):
    if not isinstance(segments, list) or not segments:
        raise ValueError("Source segment provenance missing")
    previous = 0
    for item in segments:
        start, end = item.get("start_row"), item.get("end_row")
        if (not isinstance(item.get("name"), str) or not item["name"]
                or type(start) is not int or type(end) is not int
                or start != previous or not 0 <= start < end < len(source_time)
                or not np.isclose(item.get("start_s", np.nan), source_time[start], rtol=0., atol=1e-10)
                or not np.isclose(item.get("end_s", np.nan), source_time[end], rtol=0., atol=1e-10)):
            raise ValueError("Source segment rows/clocks must cover all rows with continuous seams")
        previous = end
    if previous != len(source_time)-1:
        raise ValueError("Source segments must preserve all rows")


@dataclass
class A1SkillPath(SkillPath):
    knot_time_s: np.ndarray
    knot_qpos: np.ndarray
    coefficients: np.ndarray

    def sample(self, elapsed):
        return _curve_sample(self.knot_time_s, self.knot_qpos, self.coefficients, elapsed)

    def derivatives(self, elapsed):
        return tuple(_curve_sample(self.knot_time_s, self.knot_qpos, self.coefficients, elapsed, n) for n in (1, 2))


@dataclass
class A1Library:
    config: dict
    single_assembly: object
    guard_q: np.ndarray
    skills: dict
    reports: dict
    source_arrays: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)


def _load_pilot(folder, assembly):
    folder = Path(folder).resolve()
    metadata, raw = _json_file(folder/"pilot.json")
    _role(metadata)
    if (metadata.get("schema_version") != 2 or metadata.get("status") != "candidate_only"
            or metadata.get("source_commit") != UPSTREAM_COMMIT
            or tuple(metadata.get("skills", {})) != SKILL_IDS
            or metadata.get("robot_grip_to_tip_m") != .79
            or metadata.get("unchanged_model_source_hashes") != assembly.source_hashes):
        raise ValueError("Pilot schema/source/model/skill-set mismatch")
    if metadata.get("config", {}).get("policy_family") != "A1":
        raise ValueError("Pilot config must identify the A1 policy family")
    arrays, hashes = {}, {"pilot.json": _sha(raw)}
    for name in SKILL_IDS:
        report = metadata["skills"][name]
        if (report.get("policy_family") != "A1" or report.get("geometry_candidate_only") is not True
                or report.get("all_authored_rows_retained") is not True
                or report.get("ik", {}).get("all_frames_passed") is not True
                or report.get("strict_static_and_interpolated_screen", {}).get("candidate_passes_geometry_screen") is not True
                or report.get("strict_static_and_interpolated_screen", {}).get("all_raw_contacts") != []):
            raise ValueError(f"{name}: strict A1 source geometry gate not passed")
        arrays[name], numeric_raw = _numeric_file(folder/f"{name}.npz")
        hashes[f"{name}.npz"] = _sha(numeric_raw)
        times, qs = _sample_arrays(arrays[name])
        required = {"grip_local_m", "axis_local", "pose_success", "collision_free"}
        if not required.issubset(arrays[name]):
            raise ValueError(f"{name}: incomplete verified geometry arrays")
        for key in ("pose_success", "collision_free"):
            if arrays[name][key].shape != times.shape or arrays[name][key].dtype.kind != "b" or not arrays[name][key].all():
                raise ValueError(f"{name}: failed/nonboolean source geometry rows")
        if (arrays[name]["grip_local_m"].shape != (len(qs), 3) or arrays[name]["axis_local"].shape != (len(qs), 3)
                or not np.allclose(np.linalg.norm(arrays[name]["axis_local"], axis=1), 1., atol=1e-7)):
            raise ValueError(f"{name}: invalid source tool targets")
        for q, target, axis in zip(qs, arrays[name]["grip_local_m"], arrays[name]["axis_local"]):
            actual, direction, tip = tool_fk(assembly.model, qpos=q)
            if (np.linalg.norm(actual-target) > 1e-4+1e-9
                    or np.arccos(np.clip(direction@axis, -1., 1.)) > np.deg2rad(.05)+1e-9
                    or np.linalg.norm(tip-(target+.79*axis)) > .0005+1e-9):
                raise ValueError(f"{name}: fresh actual-tool FK does not reproduce source geometry")
        screen = _screen(assembly, qs, .05)
        if not screen["candidate_passes_geometry_screen"]:
            raise ValueError(f"{name}: fresh strict source geometry failed")
    common_guard = arrays["guard"]["qpos"][0].copy()
    if not np.array_equal(arrays["guard"]["qpos"], np.tile(common_guard, (len(arrays["guard"]["qpos"]), 1))):
        raise ValueError("Guard source must be an exact joint hold")
    for name in SKILL_IDS:
        if not np.array_equal(arrays[name]["qpos"][0], common_guard):
            raise ValueError(f"{name}: source must start at the exact common Guard; no reset or hidden entry bridge")
    return metadata, arrays, hashes, common_guard


def _build_path(name, action_id, arrays, source_segments, common_guard, cfg):
    original_time, original_q = _sample_arrays(arrays)
    source_time = original_time*cfg["source_time_scale"]
    coefficients = _hermite_coefficients(source_time, original_q)
    knots = original_q.copy()
    times = source_time.copy()
    segments = copy.deepcopy(source_segments)
    for item in segments:
        item["source_start_s"], item["source_end_s"] = item["start_s"], item["end_s"]
        item["start_s"] *= cfg["source_time_scale"]
        item["end_s"] *= cfg["source_time_scale"]
        item["source_start_row"], item["source_end_row"] = item["start_row"], item["end_row"]
        item["runtime_interpolation"] = cfg["interpolation"]
    if name != "guard":
        delta = common_guard-original_q[-1]
        duration = max(cfg["minimum_return_s"], 1.875*float(np.abs(delta).max())/cfg["return_reference_speed_rad_s"],
                       np.sqrt((10/np.sqrt(3))*float(np.abs(delta).max())/cfg["return_reference_acceleration_rad_s2"]))
        duration = float(np.ceil(duration/cfg["control_period_s"])*cfg["control_period_s"])
        returning = np.stack([original_q[-1], common_guard])
        return_times = np.array([0., duration])
        coefficients = np.concatenate([coefficients, _hermite_coefficients(return_times, returning)])
        knots = np.vstack([knots, common_guard])
        times = np.r_[times, times[-1]+duration]
        segments.append({"name": "authored_joint_return", "start_s": float(source_time[-1]), "end_s": float(times[-1]),
                         "source_frame_interval": None, "source_start_row": None, "source_end_row": None,
                         "runtime_interpolation": "authored quintic joint return, zero endpoint velocity/acceleration; NOT ASE data"})
    dense_time = []
    for i in range(len(times)-1):
        count = max(1, int(np.ceil((times[i+1]-times[i])/cfg["control_period_s"])))
        values = np.linspace(times[i], times[i+1], count+1)
        dense_time.extend(values if not dense_time else values[1:])
    dense_time = np.asarray(dense_time)
    dense_q = np.asarray([_curve_sample(times, knots, coefficients, t) for t in dense_time])
    for item in segments:
        item["start_row"] = int(np.argmin(np.abs(dense_time-item["start_s"])))
        item["end_row"] = int(np.argmin(np.abs(dense_time-item["end_s"])))
    labels = []
    for t in .5*(dense_time[:-1]+dense_time[1:]):
        labels.append(next(s["name"] for s in segments if t < s["end_s"]+1e-10))
    return A1SkillPath(name, action_id, dense_q, dense_time, labels, segments, times, knots, coefficients)


def build_a1_library(pilot_dir=DEFAULT_PILOT, config=None):
    """Build closed C2 candidates with fresh strict geometry; never auto-admit."""
    cfg = _config(config)
    assembly = build_floor_assembly(joint_ranges=np.tile([-np.pi, np.pi], (7, 1)))
    metadata, arrays, hashes, guard = _load_pilot(pilot_dir, assembly)
    skills, reports = {}, {}
    for action_id, name in enumerate(SKILL_IDS):
        segments = metadata.get("segments", {}).get(name)
        _source_segments(segments, arrays[name]["time_s"])
        path = _build_path(name, action_id, arrays[name], segments, guard, cfg)
        continuity = _check_curve(path.knot_time_s, path.knot_qpos, path.coefficients)
        screen = _screen(assembly, path.qpos, cfg["ik_joint_margin_rad"])
        closed = np.array_equal(path.qpos[0], guard) and np.array_equal(path.qpos[-1], guard)
        reports[name] = dict(screen, action_id=action_id, source_rows_preserved=True,
                            exact_guard_start=bool(np.array_equal(path.qpos[0], guard)),
                            exact_guard_end=bool(np.array_equal(path.qpos[-1], guard)),
                            candidate_passes_skill_geometry=bool(screen["candidate_passes_geometry_screen"] and closed),
                            source_time_scale=cfg["source_time_scale"], duration_s=path.duration_s,
                            interpolation_changed_from_pilot_piecewise_linear=True,
                            command_continuity=continuity,
                            physical_attack_or_defence_validation="NOT RUN", admitted_to_training=False,
                            derivative_diagnostics=_peak_derivatives(path.knot_time_s, path.coefficients))
        skills[name] = path
    reports["library"] = {
        "status": "executable_candidate", "policy_family": "A1", "action_space_kind": "finite_skill_selection",
        "robot_slot_independent": True, "a2_continuous_action_space": False,
        "all_skills_pass_geometry": all(reports[n]["candidate_passes_skill_geometry"] for n in SKILL_IDS),
        "exact_shared_joint_guard_seams": all(reports[n]["exact_guard_start"] and reports[n]["exact_guard_end"] for n in SKILL_IDS),
        "native_physical_joint_seams_certified": False, "admitted_to_training": False, "may_start_RL": False,
        "runtime_opponent_inputs": False,
        "scope": "C2 command curves with sampled strict single-arm proxy geometry only; no native dynamics, effective attack or defence certificate",
        "required_next_gate": "native A1 controller and v2 whole-window paired attack/defence/switching validation",
    }
    provenance = {"pilot_path": str(Path(pilot_dir).resolve()), "input_sha256": hashes,
                  "source_pilot_metadata": metadata, "unchanged_model_source_hashes": assembly.source_hashes,
                  "generator_sha256": _sha(Path(__file__).read_bytes())}
    return A1Library(cfg, assembly, guard, skills, reports, arrays, provenance)


def export_a1_library(library, output_dir):
    folder = Path(output_dir)
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError("Preserve evidence: select a new or empty A1 bank directory")
    folder.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, path in library.skills.items():
        arrays = dict(qpos=path.qpos, time_s=path.time_s, knot_time_s=path.knot_time_s,
                      knot_qpos=path.knot_qpos, coefficients=path.coefficients)
        arrays.update({"source_"+key: value for key, value in library.source_arrays[name].items()})
        target = folder/f"{name}.npz"
        np.savez_compressed(target, **arrays)
        files[name] = {"filename": target.name, "sha256": _sha(target.read_bytes()), "action_id": path.action_id,
                       "segments": path.segments, "segment_labels": path.segment_labels}
    manifest = dict(schema_version=1, format="a1_closed_numeric_bank_v1", status="executable_candidate",
                    policy_family="A1", action_space_kind="finite_skill_selection", robot_slot_independent=True,
                    a2_continuous_action_space=False, admitted_to_training=False, may_start_RL=False,
                    config=library.config, guard_q_rad=library.guard_q.tolist(), skills=files,
                    reports=library.reports, provenance=library.provenance)
    target = folder/"bank.json"
    target.write_text(json.dumps(_finite_json(manifest), indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return target


def load_a1_library(bank_dir):
    """Load fresh-exported numeric bank without pickle, rechecking model/geometry."""
    folder = Path(bank_dir)
    manifest, _ = _json_file(folder/"bank.json")
    _role(manifest)
    if (manifest.get("schema_version") != 1 or manifest.get("format") != "a1_closed_numeric_bank_v1"
            or manifest.get("status") != "executable_candidate" or tuple(manifest.get("skills", {})) != SKILL_IDS):
        raise ValueError("Unsupported A1 numeric bank schema/skills")
    cfg = _config(manifest["config"])
    assembly = build_floor_assembly(joint_ranges=np.tile([-np.pi, np.pi], (7, 1)))
    if manifest.get("provenance", {}).get("unchanged_model_source_hashes") != assembly.source_hashes:
        raise ValueError("A1 bank model hashes changed")
    guard = np.asarray(manifest["guard_q_rad"], dtype=float)
    if guard.shape != (7,) or not np.isfinite(guard).all():
        raise ValueError("Finite seven-joint Guard required")
    skills, source_arrays, reports = {}, {}, copy.deepcopy(manifest["reports"])
    for action_id, name in enumerate(SKILL_IDS):
        item = manifest["skills"][name]
        if item.get("filename") != name+".npz" or item.get("action_id") != action_id:
            raise ValueError("A1 filename/action-ID mismatch")
        arrays, raw = _numeric_file(folder/item["filename"])
        if _sha(raw) != item["sha256"]:
            raise ValueError("A1 bank numeric SHA256 mismatch")
        time, q = _sample_arrays(arrays)
        kt, kq, coeff = arrays["knot_time_s"], arrays["knot_qpos"], arrays["coefficients"]
        if (kt.ndim != 1 or len(kt) < 2 or kq.shape != (len(kt), 7) or coeff.shape != (len(kt)-1, 6, 7)
                or kt[0] != 0. or np.any(np.diff(kt) <= 0) or kt[-1] != time[-1]
                or not np.array_equal(q[0], guard) or not np.array_equal(q[-1], guard)):
            raise ValueError("Invalid closed A1 command knots")
        if not np.array_equal(kq[0], guard) or not np.array_equal(kq[-1], guard):
            raise ValueError("A1 knot endpoints must equal common Guard exactly")
        _check_curve(kt, kq, coeff)
        reconstructed = np.asarray([_curve_sample(kt, kq, coeff, t) for t in time])
        if not np.allclose(reconstructed, q, rtol=0., atol=1e-12):
            raise ValueError("A1 dense samples do not match their C2 curve")
        screen = _screen(assembly, q, cfg["ik_joint_margin_rad"])
        if not screen["candidate_passes_geometry_screen"] or not reports[name]["candidate_passes_skill_geometry"]:
            raise ValueError("Fresh strict A1 geometry gate failed; failed candidate cannot be loaded for execution")
        skills[name] = A1SkillPath(name, action_id, q, time, item["segment_labels"], item["segments"], kt, kq, coeff)
        source_arrays[name] = {key[len("source_"):]: value for key, value in arrays.items() if key.startswith("source_")}
    if reports["library"].get("all_skills_pass_geometry") is not True:
        raise ValueError("A1 bank preflight not passed")
    return A1Library(cfg, assembly, guard, skills, reports, source_arrays, manifest["provenance"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--source-time-scale", type=float, default=1.)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    library = build_a1_library(args.pilot_dir, {"source_time_scale": args.source_time_scale})
    print(export_a1_library(library, args.output))
    print("A1 executable candidate only; geometry:", library.reports["library"]["all_skills_pass_geometry"], "; physical/RL admission: False")


if __name__ == "__main__":
    main()
