"""Rebuild the pinned five-skill A1 v5 candidate from locally held ASE files.

This offline preparation performs IK and geometry screens, not native dynamics
or training. It never downloads or redistributes motion data. Each intermediate
candidate (including a failed one) is retained in a fresh output directory.
The published numerical v5 fingerprint is a fail-closed reproduction check;
passing it does not itself certify fencing ability or hardware safety.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import mujoco
import numpy as np

from .ase_pilot import build_pilot, export_pilot, load_config
from .a1_adaptation import build_adapted_a1_library
from .a1_defence import build_defence_candidate
from .a1_skills import SKILL_IDS, export_a1_library
from .assembly import _finite_json


ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = ROOT / "config/ase_pilot_v2_calibrated_candidate.json"
RIGHT_RECIPE = ROOT / "config/a1_defence_counter_stab_candidate.json"
LEFT_RECIPE = ROOT / "config/a1_defence_slash_vertical_candidate.json"
REFERENCE = ROOT / "config/a1_reference_bank.json"
RUNTIME_ARRAYS = ("qpos", "time_s", "knot_qpos", "knot_time_s", "coefficients")


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sources(source_dir, config):
    """Size/hash preflight only: never unpickle, download, or alter input data."""
    folder = Path(source_dir).resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"ASE --source-dir does not exist: {folder}")
    if len(config["sources"]) != 5:
        raise ValueError("The pinned v5 recipe requires exactly five ASE source files")
    verified = {}
    for name, entry in config["sources"].items():
        filename = entry["filename"]
        if Path(filename).name != filename:
            raise ValueError(f"Unsafe source filename: {filename}")
        path = folder / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing required ASE file: {path}; obtain it under the upstream data license")
        size = path.stat().st_size
        if size != entry["size_bytes"]:
            raise ValueError(f"ASE size mismatch for {filename}: expected {entry['size_bytes']}, got {size}")
        digest = _sha(path)
        if digest != entry["sha256"]:
            raise ValueError(f"ASE SHA256 mismatch for {filename}; no fallback or unverified loading is allowed")
        verified[name] = {"path": str(path), "filename": filename,
                          "size_bytes": size, "sha256": digest}
    return verified


def _freeze_inputs():
    """Record current generators and all declared unchanged CAD/model inputs."""
    code = sorted(set(ROOT.glob("sim/*.py")) | set(ROOT.glob("plan1_skills/*.py")))
    assembly_path = ROOT / "config/assembly.json"
    assembly = json.loads(assembly_path.read_text(encoding="utf-8"))
    fixed = {ROOT / "config/ase_pilot_v2.json", CALIBRATION, RIGHT_RECIPE,
             LEFT_RECIPE, REFERENCE, assembly_path, ROOT / "assets/shinai_profile.json"}
    fixed.update(p for p in (ROOT / "meshes").rglob("*") if p.is_file())
    for name in (*assembly["expected_source_sha256"], assembly["source_scene"]):
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("Declared model input must stay inside the repository")
        fixed.add(path)
    return {"implementation_sha256": {str(p.relative_to(ROOT)): _sha(p) for p in code},
            "fixed_input_sha256": {str(p.relative_to(ROOT)): _sha(p) for p in sorted(fixed)}}


def _save(output, report):
    (output / "prepare_report.json").write_text(
        json.dumps(_finite_json(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _pilot_geometry(pilot):
    skills = pilot.metadata["skills"]
    return {"all_skills_pass_geometry": all(skills[n].get("geometry_candidate_only") is True for n in SKILL_IDS),
            "skills": {n: {k: skills[n].get(k) for k in
                ("geometry_candidate_only", "ik", "strict_static_and_interpolated_screen",
                 "all_authored_rows_retained")} for n in SKILL_IDS}}


def _bank_geometry(bank):
    return {"all_skills_pass_geometry": bank.reports["library"]["all_skills_pass_geometry"],
            "reports": bank.reports}


def numeric_identity(bank_dir):
    """Fingerprint arrays, independent of ZIP timestamps or metadata paths."""
    result = {}
    for name in SKILL_IDS:
        with np.load(Path(bank_dir) / f"{name}.npz", allow_pickle=False) as archive:
            arrays = {}
            for key in RUNTIME_ARRAYS:
                value = np.ascontiguousarray(archive[key])
                if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
                    raise ValueError(f"Invalid numeric runtime array: {name}/{key}")
                arrays[key] = {"dtype": value.dtype.str, "shape": list(value.shape),
                               "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
            result[name] = {"duration_s": float(archive["time_s"][-1]), "arrays": arrays}
    return result


def prepare(source_dir, output):
    """Produce OUTPUT/library or raise, retaining failed stage evidence.

    Missing/corrupt sources fail before OUTPUT is created. Later failures save a
    report and stop immediately; they never trigger relaxed IK or other seeds.
    """
    source_dir, output = Path(source_dir).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Preserve evidence: --output must be a fresh path: {output}")
    if output.is_relative_to(source_dir):
        raise ValueError("Output must not be inside the ASE source directory")
    cfg = load_config(json.loads(CALIBRATION.read_text(encoding="utf-8")))
    verified = verify_sources(source_dir, cfg)
    right = json.loads(RIGHT_RECIPE.read_text(encoding="utf-8"))
    left = json.loads(LEFT_RECIPE.read_text(encoding="utf-8"))
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    if reference.get("schema_version") != 1 or tuple(reference.get("skills", {})) != SKILL_IDS:
        raise ValueError("Pinned v5 numeric reference schema/skill order changed")
    frozen = _freeze_inputs()
    report = {"schema_version": 1, "status": "running", "recipe": "A1_v5_Right25_Left30_enhancedSlash",
              "scope": "offline candidate reconstruction; no native physics, RL, or automatic admission",
              "admitted_to_training": False, "source_data_redistributed": False,
              "sources": verified, **frozen, "stages": [],
              "runtime_versions": {"python": platform.python_version(), "numpy": np.__version__, "mujoco": mujoco.__version__},
              "parameters": {"source_time_scale": 4., "slash_amplitude": .8,
                             "stage_order": ["pilot", "adapted", "right25", "library"]}}
    output.mkdir(parents=True, exist_ok=False)
    _save(output, report)

    def stage(name, builder, exporter, geometry):
        entry = {"name": name, "status": "running", "output": name}
        report["stages"].append(entry)
        _save(output, report)
        result = builder()
        exported = exporter(result, output / name)
        entry.update(status="exported", metadata_file=str(Path(exported).relative_to(output)),
                     geometry=geometry(result))
        _save(output, report)  # Keep failed geometry and every failed row first.
        if entry["geometry"]["all_skills_pass_geometry"] is not True:
            raise RuntimeError(f"{name}: geometry screen failed; candidate retained, later stages not run")
        entry["status"] = "geometry_passed"
        _save(output, report)
        return result

    try:
        stage("pilot", lambda: build_pilot(cfg, source_dirs=[source_dir], with_ik=True), export_pilot, _pilot_geometry)
        bank = stage("adapted", lambda: build_adapted_a1_library(output / "pilot",
                     {"source_time_scale": 4.}, {"amplitude": .8}), export_a1_library, _bank_geometry)
        bank = stage("right25", lambda: build_defence_candidate(bank, right),
                     export_a1_library, _bank_geometry)
        bank = stage("library", lambda: build_defence_candidate(bank, left),
                     export_a1_library, _bank_geometry)
        actual = numeric_identity(output / "library")
        report["bank_identity"] = {"bank_json_sha256": _sha(output / "library/bank.json"),
            "runtime_numeric_identity": actual, "expected_v5_numeric_identity": reference["skills"],
            "numeric_matches_validated_v5": actual == reference["skills"],
            "metadata_note": "Metadata paths and ZIP packaging are not the runtime numerical identity; metadata hash is recorded for this build"}
        if actual != reference["skills"]:
            raise RuntimeError("Rebuilt numerical bank differs from pinned v5; outputs retained, do not treat as validated v5")
        if _freeze_inputs() != frozen or verify_sources(source_dir, cfg) != verified:
            raise RuntimeError("Preparation inputs changed during reconstruction")
        report.update(status="complete", inputs_unchanged=True,
                      bank_directory=str(output / "library"),
                      next_step="Explicit development-mode environment only; geometry reproduction is not new physical validation")
        _save(output, report)
        return report
    except BaseException as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        if report["stages"] and report["stages"][-1]["status"] != "geometry_passed":
            report["stages"][-1]["status"] = "failed"
        _save(output, report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True, help="Directory containing all five pinned local ASE .npy files")
    parser.add_argument("--output", type=Path, required=True, help="Fresh output path; never overwritten")
    args = parser.parse_args()
    try:
        report = prepare(args.source_dir, args.output)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"A1 preparation failed: {error}\n")
    print(report["bank_directory"])
    print("Reproduced numerical v5 candidate. No training or native dynamics performed.")


if __name__ == "__main__":
    main()
