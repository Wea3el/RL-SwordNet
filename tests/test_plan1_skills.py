"""Offline tests for plan1 templates; no upstream motion data or network needed."""
from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from plan1_skills.ase_numeric import read_verified_motion
from plan1_skills.library import (
    SkillClip, SkillLibrary, build_library, fetch_sources, load_manifest,
)


def simple_clip():
    return SkillClip(
        np.array([0., 1.]),
        np.array([[0., 0., 0.], [2., 4., 6.]]),
        np.array([[1., 0., 0.], [0., 1., 0.]]),
    )


def write_library(directory, skills=None):
    """Generate numerical fixtures, not copies of upstream motion data."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    clip = simple_clip()
    for name in ("guard", "stab", "parry_up"):
        np.savez_compressed(
            directory / f"{name}.npz", phase=clip.phase,
            grip_delta_m=clip.grip_delta_m, blade_direction=clip.blade_direction,
        )
    entries = skills if skills is not None else [
        {"id": name, "file": f"{name}.npz"}
        for name in ("guard", "stab", "parry_up")
    ]
    metadata = {
        "schema_version": 1, "plan_id": "plan1",
        "skills": [{"frames": 2, **entry} for entry in entries],
    }
    (directory / "library.json").write_text(json.dumps(metadata), encoding="utf-8")
    return metadata


def fake_source(payload=b"offline-test-payload"):
    return {
        "filename": "example.npy", "url": "https://example.invalid/example.npy",
        "size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
    }


def synthetic_motion(root_moves=True, grip_offset=None):
    """Known shoulder basis and separable root/hand translation, all synthetic."""
    root = np.tile(np.array([10., 0., 1.]), (3, 1))
    if root_moves:
        root[:, 0] += np.array([0., 1., 2.])
    relative_grip = np.array([[0., 1., 1.], [.2, 1.4, 1.], [.4, 1.8, 1.2]])
    grip = root + relative_grip
    if grip_offset is not None:
        grip += np.asarray(grip_offset)
    direction = np.tile(np.array([0., 0., 1.]), (3, 1))
    return {
        "time": np.arange(3) / 30., "grip_m": grip, "direction": direction,
        "shoulder_l_m": root + [.2, 0., 1.],
        "shoulder_r_m": root + [-.2, 0., 1.], "root_m": root,
        "source_tip_m": grip + .74 * direction, "frames": 3, "fps": 30.,
    }


class ManifestTests(unittest.TestCase):
    def test_three_skills_and_immutable_verified_sources(self):
        manifest = load_manifest()
        self.assertEqual(manifest["plan_id"], "plan1")
        self.assertEqual([s["id"] for s in manifest["skills"]], ["guard", "stab", "parry_up"])
        self.assertEqual([s["action_id"] for s in manifest["skills"]], [0, 1, 2])
        self.assertEqual(set(manifest["sources"]), {"stab", "parry_up"})
        commit = "6f9b4f1f289603eee6a4f45d082bc6e6d83fecec"
        self.assertEqual(manifest["upstream"]["commit"], commit)
        expected = {
            "stab": ("RL_Avatar_Atk_Stab_Motion.npy", 178957, 128,
                     "84447bd38d1efa356dc44d68f5e5d516760c03cdb236aa1a13c181f3bbc6d3c0"),
            "parry_up": ("RL_Avatar_Sword_ParryUp_Motion.npy", 82077, 58,
                         "d883d04247caa594b5a1f142ea18aac930ef7a97603abe22758be386af1d4b99"),
        }
        prefix = f"https://raw.githubusercontent.com/nv-tlabs/ASE/{commit}/ase/data/motions/reallusion_sword_shield/"
        for name, (filename, size, frames, digest) in expected.items():
            source = manifest["sources"][name]
            with self.subTest(source=name):
                self.assertEqual(source["filename"], filename)
                self.assertEqual(source["url"], prefix + filename)
                self.assertEqual(source["size_bytes"], size)
                self.assertEqual(source["frames"], frames)
                self.assertEqual(source["sha256"], digest)
                self.assertTrue(29 < source["fps"] < 31)
        self.assertEqual(manifest["processing"]["grip_displacement_scale"], .5)
        self.assertTrue(manifest["processing"]["remove_root_translation"])


class SkillClipTests(unittest.TestCase):
    def test_linear_position_and_spherical_direction_interpolation(self):
        clip = simple_clip()
        point, axis = clip.sample(.5)
        np.testing.assert_allclose(point, [1., 2., 3.])
        np.testing.assert_allclose(axis, [np.sqrt(.5), np.sqrt(.5), 0.], atol=1e-12)
        for phase, index in ((0., 0), (1., -1)):
            np.testing.assert_allclose(clip.sample(phase)[0], clip.grip_delta_m[index])
            np.testing.assert_allclose(clip.sample(phase)[1], clip.blade_direction[index])

    def test_constant_direction_interpolation_stays_unit(self):
        clip = SkillClip(np.array([0., 1.]), np.zeros((2, 3)), np.tile([0., 0., 1.], (2, 1)))
        for phase in (0., .3, 1.):
            np.testing.assert_allclose(clip.sample(phase)[1], [0., 0., 1.])

    def test_invalid_sampling_phases_are_rejected(self):
        for value in (-.001, 1.001, float("nan"), float("inf"), -float("inf")):
            with self.subTest(phase=value), self.assertRaises(ValueError):
                simple_clip().sample(value)

    def test_invalid_phase_grids_and_shapes_are_rejected(self):
        for grid in ([0.], [.1, 1.], [0., .9], [0., .5, .5, 1.], [0., .8, .2, 1.], [0., float("nan"), 1.]):
            phase = np.array(grid)
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                SkillClip(phase, np.zeros((len(phase), 3)), np.tile([1., 0., 0.], (len(phase), 1)))
        with self.assertRaises(ValueError):
            SkillClip(np.array([0., 1.]), np.zeros((2, 2)), np.tile([1., 0., 0.], (2, 1)))

    def test_nonfinite_nonunit_and_antipodal_vectors_are_rejected(self):
        for axes in (
            [[0., 0., 0.], [1., 0., 0.]],
            [[2., 0., 0.], [1., 0., 0.]],
            [[1.000005, 0., 0.], [1., 0., 0.]],
            [[1., 0., 0.], [-1., 0., 0.]],
            [[float("nan"), 0., 0.], [1., 0., 0.]],
        ):
            with self.subTest(axes=axes), self.assertRaises(ValueError):
                SkillClip(np.array([0., 1.]), np.zeros((2, 3)), np.array(axes))


class SkillLibraryTests(unittest.TestCase):
    def test_loads_complete_library_and_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            write_library(directory)
            library = SkillLibrary(directory)
            self.assertEqual(set(library.clips), {"guard", "stab", "parry_up"})
            np.testing.assert_allclose(library.sample("stab", .5)[0], [1., 2., 3.])

    def test_rejects_incomplete_library(self):
        skills = [{"id": name, "file": name + ".npz"} for name in ("guard", "stab")]
        with tempfile.TemporaryDirectory() as directory:
            write_library(directory, skills)
            with self.assertRaises(ValueError):
                SkillLibrary(directory)

    def test_rejects_duplicate_skill_id_instead_of_overwriting(self):
        skills = [{"id": name, "file": name + ".npz"} for name in ("guard", "stab", "parry_up", "stab")]
        with tempfile.TemporaryDirectory() as directory:
            write_library(directory, skills)
            with self.assertRaises(ValueError):
                SkillLibrary(directory)

    def test_rejects_path_traversal_and_absolute_paths(self):
        for filename in ("../guard.npz", "/tmp/guard.npz", "nested/guard.npz"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as directory:
                skills = [{"id": "guard", "file": filename}]
                write_library(directory, skills)
                with self.assertRaises(ValueError):
                    SkillLibrary(directory)

    def test_rejects_noncanonical_skill_filename(self):
        skills = [{"id": "guard", "file": "stab.npz"}]
        with tempfile.TemporaryDirectory() as directory:
            write_library(directory, skills)
            with self.assertRaises(ValueError):
                SkillLibrary(directory)

    def test_rejects_metadata_frame_mismatch(self):
        for frame_count in (3, 2.0, True):
            with self.subTest(frames=frame_count), tempfile.TemporaryDirectory() as directory:
                skills = [{"id": "guard", "file": "guard.npz", "frames": frame_count}]
                write_library(directory, skills)
                with self.assertRaises(ValueError):
                    SkillLibrary(directory)


class FetchTests(unittest.TestCase):
    def test_requires_notice_acknowledgement_before_creating_cache_or_network(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "not-created"
            with patch("plan1_skills.library.urlopen") as network:
                with self.assertRaises(ValueError):
                    fetch_sources(target)
                network.assert_not_called()
            self.assertFalse(target.exists())

    def test_fetches_small_payload_and_verifies_hash(self):
        payload = b"offline-test-payload"
        source = fake_source(payload)
        with tempfile.TemporaryDirectory() as directory:
            with patch("plan1_skills.library.load_manifest", return_value={"sources": {"test": source}}), \
                    patch("plan1_skills.library.urlopen", return_value=io.BytesIO(payload)) as network:
                result = fetch_sources(directory, acknowledge_source_notice=True)
            self.assertEqual(result, [Path(directory) / source["filename"]])
            self.assertEqual(result[0].read_bytes(), payload)
            self.assertEqual(network.call_args.args[0].full_url, source["url"])
            self.assertEqual(set(p.name for p in Path(directory).iterdir()), {source["filename"]})

    def test_reuses_valid_cache_without_network(self):
        payload = b"offline-test-payload"
        source = fake_source(payload)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / source["filename"]
            target.write_bytes(payload)
            with patch("plan1_skills.library.load_manifest", return_value={"sources": {"test": source}}), \
                    patch("plan1_skills.library.urlopen") as network:
                self.assertEqual(fetch_sources(directory, acknowledge_source_notice=True), [target])
                network.assert_not_called()

    def test_does_not_replace_existing_corrupt_cache(self):
        source = fake_source()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / source["filename"]
            target.write_bytes(b"user-owned-corrupt-cache")
            with patch("plan1_skills.library.load_manifest", return_value={"sources": {"test": source}}), \
                    patch("plan1_skills.library.urlopen") as network:
                with self.assertRaises(ValueError):
                    fetch_sources(directory, acknowledge_source_notice=True)
                network.assert_not_called()
            self.assertEqual(target.read_bytes(), b"user-owned-corrupt-cache")

    def test_rejects_wrong_remote_hash_or_size_and_cleans_temporary_file(self):
        source = fake_source()
        for payload in (b"x" * source["size_bytes"], b"short", b"x" * (source["size_bytes"] + 10)):
            with self.subTest(payload_length=len(payload)), tempfile.TemporaryDirectory() as directory:
                with patch("plan1_skills.library.load_manifest", return_value={"sources": {"test": source}}), \
                        patch("plan1_skills.library.urlopen", return_value=io.BytesIO(payload)):
                    with self.assertRaises(ValueError):
                        fetch_sources(directory, acknowledge_source_notice=True)
                self.assertEqual(list(Path(directory).iterdir()), [])


class BuildTests(unittest.TestCase):
    def synthetic_manifest(self):
        manifest = copy.deepcopy(load_manifest())
        for source in manifest["sources"].values():
            source["frames"], source["fps"] = 3, 30.
        return manifest

    def read_fake(self, path, _sha, _size):
        return synthetic_motion(grip_offset=[.4, 0., 0.] if "Parry" in path.name else None)

    def test_build_removes_root_translation_scales_displacement_and_adds_guard(self):
        manifest = self.synthetic_manifest()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "templates"
            with patch("plan1_skills.library.load_manifest", return_value=manifest), \
                    patch("plan1_skills.ase_numeric.read_verified_motion", side_effect=self.read_fake) as reader:
                metadata = build_library(Path(directory) / "sources", output)
            self.assertEqual(reader.call_count, 2)
            self.assertIn("not_robot_ready", metadata["status"])
            library = SkillLibrary(output)
            np.testing.assert_allclose(library.clips["stab"].grip_delta_m,
                                       [[0., 0., 0.], [.1, .2, 0.], [.2, .4, .1]], atol=1e-12)
            np.testing.assert_allclose(library.clips["parry_up"].grip_delta_m[0], [.2, 0., 0.], atol=1e-12)
            guard = library.clips["guard"]
            np.testing.assert_array_equal(guard.phase, [0., 1.])
            np.testing.assert_array_equal(guard.grip_delta_m, np.zeros((2, 3)))
            np.testing.assert_allclose(guard.blade_direction, [[0., 0., 1.], [0., 0., 1.]])
            for skill in ("stab", "parry_up"):
                with np.load(output / f"{skill}.npz", allow_pickle=False) as data:
                    np.testing.assert_allclose(data["source_time_s"], np.arange(3) / 30.)

    def test_refuses_nonempty_output_without_touching_existing_content(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "templates"
            output.mkdir()
            existing = output / "library.json"
            existing.write_text("user-owned artifact", encoding="utf-8")
            with patch("plan1_skills.ase_numeric.read_verified_motion") as reader:
                with self.assertRaises(FileExistsError):
                    build_library(Path(directory) / "sources", output)
                reader.assert_not_called()
            self.assertEqual(existing.read_text(), "user-owned artifact")
            self.assertEqual(list(output.iterdir()), [existing])

    def test_rejects_manifest_frame_mismatch_before_writing(self):
        manifest = self.synthetic_manifest()
        manifest["sources"]["stab"]["frames"] = 4
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "templates"
            with patch("plan1_skills.library.load_manifest", return_value=manifest), \
                    patch("plan1_skills.ase_numeric.read_verified_motion", side_effect=self.read_fake):
                with self.assertRaises(ValueError):
                    build_library(Path(directory) / "sources", output)
            self.assertFalse(output.exists())


class NumericVerificationTests(unittest.TestCase):
    def test_size_and_hash_rejection_happen_before_decode(self):
        payload = b"not-an-upstream-motion"
        digest = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fake.npy"
            path.write_bytes(payload)
            for sha, size in ((digest, len(payload) + 1), ("0" * 64, len(payload))):
                with self.subTest(sha=sha, size=size), \
                        patch("plan1_skills.ase_numeric._decode_verified_bytes") as decode:
                    with self.assertRaises(ValueError):
                        read_verified_motion(path, sha, size)
                    decode.assert_not_called()

    def test_forbidden_global_is_rejected_even_with_matching_manifest_hash(self):
        stream = io.BytesIO()
        np.lib.format.write_array_header_1_0(stream, {"descr": "|O", "fortran_order": False, "shape": ()})
        # This contains a GLOBAL symbol, not a request to execute a command.
        payload = stream.getvalue() + b"\x80\x03cos\nsystem\n."
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "forbidden.npy"
            path.write_bytes(payload)
            with patch("os.system") as system:
                with self.assertRaisesRegex(ValueError, "GLOBAL"):
                    read_verified_motion(path, hashlib.sha256(payload).hexdigest(), len(payload))
                system.assert_not_called()


if __name__ == "__main__":
    unittest.main()
