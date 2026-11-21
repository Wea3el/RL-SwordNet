"""Fetch verified source data and build portable, numerical motion templates."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "motion_skills" / "plan1" / "manifest.json"
DEFAULT_CACHE = ROOT / "outputs" / "plan1_skills"
CONVERTER_VERSION = "1.0.0"


def load_manifest() -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("plan_id") != "plan1":
        raise ValueError("Unsupported plan1 manifest")
    if [s["id"] for s in manifest["skills"]] != ["guard", "stab", "parry_up"]:
        raise ValueError("Unexpected skill set")
    return manifest


def verify_file(path: Path, source: dict) -> None:
    if path.stat().st_size != source["size_bytes"]:
        raise ValueError(f"{path.name}: source size mismatch")
    if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
        raise ValueError(f"{path.name}: source SHA256 mismatch")


def fetch_sources(directory: Path, *, acknowledge_source_notice: bool = False) -> list[Path]:
    """Read the notice before opting into downloading; no rights are granted here."""
    if not acknowledge_source_notice:
        raise ValueError("Review the source notice, then pass --acknowledge-source-notice")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    result = []
    for source in load_manifest()["sources"].values():
        target = directory / source["filename"]
        if target.exists():
            verify_file(target, source)
            result.append(target)
            continue
        request = Request(source["url"], headers={"User-Agent": "RL-SwordNet-plan1/1"})
        temporary = None
        try:
            with urlopen(request, timeout=30) as response, tempfile.NamedTemporaryFile(
                prefix=".plan1-download-", dir=directory, delete=False
            ) as stream:
                temporary = Path(stream.name)
                payload = response.read(source["size_bytes"] + 1)
                stream.write(payload)
            verify_file(temporary, source)
            # Never overwrite an independently created or modified cache file.
            if target.exists():
                raise FileExistsError(target)
            os.link(temporary, target)
            result.append(target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return result


def body_frame_motion(motion: dict) -> tuple[np.ndarray, np.ndarray]:
    grip = motion["grip_m"] - (motion["root_m"] - motion["root_m"][0])
    origin = .5 * (motion["shoulder_l_m"][0] + motion["shoulder_r_m"][0])
    x = motion["shoulder_l_m"][0] - motion["shoulder_r_m"][0]
    x = np.asarray(x, dtype=float).copy()
    x[2] = 0
    if np.linalg.norm(x) < .1:
        raise ValueError("Degenerate first-pose shoulder frame")
    x /= np.linalg.norm(x)
    up = np.array([0., 0., 1.])
    basis = np.column_stack([x, np.cross(up, x), up])
    return (grip - origin) @ basis, motion["direction"] @ basis


def build_library(source_directory: Path, output_directory: Path) -> dict:
    from .ase_numeric import read_verified_motion

    manifest = load_manifest()
    output_directory = Path(output_directory)
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError("Choose a new or empty output directory; existing artifacts are preserved")
    motions = {}
    for key, source in manifest["sources"].items():
        motion = read_verified_motion(Path(source_directory) / source["filename"], source["sha256"], source["size_bytes"])
        if motion["frames"] != source["frames"] or not np.isclose(motion["fps"], source["fps"], rtol=1e-9):
            raise ValueError(f"{key}: manifest frame/fps mismatch")
        motions[key] = motion
    poses = {key: body_frame_motion(value) for key, value in motions.items()}
    neutral_grip, neutral_axis = poses["stab"][0][0], poses["stab"][1][0]
    scale = manifest["processing"]["grip_displacement_scale"]
    arrays = {}
    for skill in manifest["skills"]:
        if skill["id"] == "guard":
            phase = np.array([0., 1.])
            delta = np.zeros((2, 3))
            axes = np.tile(neutral_axis, (2, 1))
            source_time = np.zeros(2)
        else:
            motion = motions[skill["source"]]
            grip, axes = poses[skill["source"]]
            phase = np.linspace(0., 1., len(grip))
            delta = scale * (grip - neutral_grip)
            source_time = motion["time"]
        # Validate all artifacts before writing any output.
        clip = SkillClip(phase, delta, axes)
        arrays[skill["id"]] = (clip, source_time)
    metadata = {
        "schema_version": 1, "plan_id": "plan1",
        "status": "task_space_templates_only_not_robot_ready",
        "source_commit": manifest["upstream"]["commit"],
        "source_notice_url": manifest["upstream"]["notice_url"],
        "converter": {"name": "plan1_skills", "version": CONVERTER_VERSION,
                      "numpy_version": np.__version__},
        "sources": manifest["sources"], "processing": manifest["processing"],
        "neutral_axis_in_skill_frame": neutral_axis.tolist(),
        "warning": "No robot mount, IK, joint limits, playback speed, transitions, collisions, scoring or PPO is implemented by this library.",
        "skills": [{**s, "file": s["id"] + ".npz", "frames": len(arrays[s["id"]][0].phase)} for s in manifest["skills"]],
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    for key, (clip, source_time) in arrays.items():
        np.savez_compressed(output_directory / f"{key}.npz", phase=clip.phase,
                            grip_delta_m=clip.grip_delta_m, blade_direction=clip.blade_direction,
                            source_time_s=source_time)
    (output_directory / "library.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


@dataclass(frozen=True)
class SkillClip:
    phase: np.ndarray
    grip_delta_m: np.ndarray
    blade_direction: np.ndarray

    def __post_init__(self):
        n = len(self.phase)
        if (n < 2 or self.phase.shape != (n,) or self.grip_delta_m.shape != (n, 3)
                or self.blade_direction.shape != (n, 3)):
            raise ValueError("Invalid clip array shapes")
        if not all(np.isfinite(a).all() for a in [self.phase, self.grip_delta_m, self.blade_direction]):
            raise ValueError("Nonfinite clip")
        if (not np.all(np.diff(self.phase) > 0) or not np.isclose(self.phase[0], 0, rtol=0, atol=1e-12)
                or not np.isclose(self.phase[-1], 1, rtol=0, atol=1e-12)):
            raise ValueError("Phase must increase strictly from zero to one")
        if not np.allclose(np.linalg.norm(self.blade_direction, axis=1), 1, rtol=0, atol=1e-6):
            raise ValueError("Blade directions must be unit vectors")
        if np.any(np.sum(self.blade_direction[1:] * self.blade_direction[:-1], axis=1) < -.999):
            raise ValueError("Ambiguous antipodal direction interval")

    def sample(self, phase: float) -> tuple[np.ndarray, np.ndarray]:
        if not np.isfinite(phase) or not 0 <= phase <= 1:
            raise ValueError("Phase must be finite and between zero and one")
        upper = min(max(1, int(np.searchsorted(self.phase, phase, side="right"))), len(self.phase) - 1)
        lower = upper - 1
        alpha = (phase - self.phase[lower]) / (self.phase[upper] - self.phase[lower])
        position = (1-alpha) * self.grip_delta_m[lower] + alpha * self.grip_delta_m[upper]
        a, b = self.blade_direction[lower], self.blade_direction[upper]
        angle = float(np.arccos(np.clip(a @ b, -1, 1)))
        if angle < 1e-7:
            axis = (1-alpha)*a + alpha*b
        else:
            axis = (np.sin((1-alpha)*angle)*a + np.sin(alpha*angle)*b) / np.sin(angle)
        return position, axis / np.linalg.norm(axis)


class SkillLibrary:
    def __init__(self, directory: Path):
        directory = Path(directory)
        self.metadata = json.loads((directory / "library.json").read_text(encoding="utf-8"))
        if self.metadata.get("schema_version") != 1 or self.metadata.get("plan_id") != "plan1":
            raise ValueError("Unsupported library")
        self.clips = {}
        for skill in self.metadata["skills"]:
            if (skill["id"] not in {"guard", "stab", "parry_up"}
                    or skill["file"] != skill["id"] + ".npz"):
                raise ValueError("Unexpected skill ID or filename")
            if skill["id"] in self.clips:
                raise ValueError("Duplicate skill ID")
            with np.load(directory / skill["file"], allow_pickle=False) as data:
                clip = SkillClip(data["phase"], data["grip_delta_m"], data["blade_direction"])
                if type(skill["frames"]) is not int or skill["frames"] != len(clip.phase):
                    raise ValueError("Skill frame count does not match metadata")
                self.clips[skill["id"]] = clip
        if set(self.clips) != {"guard", "stab", "parry_up"}:
            raise ValueError("Incomplete skill library")

    def sample(self, skill_id: str, phase: float) -> tuple[np.ndarray, np.ndarray]:
        return self.clips[skill_id].sample(phase)
