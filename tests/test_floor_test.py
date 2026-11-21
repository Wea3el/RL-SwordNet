"""Fast runner checks: numerical fixtures plus optional verified local ASE data.

No network, rendering, or full CAD IK run is performed by these tests.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from plan1_skills.floor_test import dense_reference, main, metrics, rotation_z
from plan1_skills.library import ROOT, DEFAULT_CACHE, SkillClip, SkillLibrary, build_library, load_manifest
from plan1_skills.matching import MatchResult


def result_fixture():
    qpos = np.array([[.1, .2], [.11, .22], [.12, .24]])
    return MatchResult(qpos=qpos, position_error_m=np.full(3, .001),
                       angle_error_rad=np.full(3, .001), tip_error_m=np.full(3, .002),
                       joint_delta_rad=np.array([[9., -8.], [.01, .02], [.01, .02]]),
                       pose_success=np.ones(3, dtype=bool),
                       collision_free=np.array([True, False, True]),
                       success=np.array([True, False, True]),
                       contacts=[[], [{"geom1": "sword", "geom2": "arm",
                                       "distance_m": -.003, "path_fraction": .5}], []],
                       iterations=np.ones(3, dtype=int), collision_sample_count=np.array([2, 3, 2]))


class DenseReferenceTests(unittest.TestCase):
    def setUp(self):
        self.clip = SkillClip(np.array([0., .3, 1.]),
                              np.array([[0., 0, 0], [1., 2, 3], [2., 1, 0]]),
                              np.array([[1., 0, 0], [0., 1, 0], [0., 0, 1]]))

    def test_every_source_phase_and_endpoint_retained(self):
        for substeps in (1, 2, 4, 20):
            phase, position, axis = dense_reference(self.clip, substeps)
            self.assertEqual(len(phase), (len(self.clip.phase)-1)*substeps+1)
            np.testing.assert_array_equal(phase[::substeps], self.clip.phase)
            np.testing.assert_allclose(position[::substeps], self.clip.grip_delta_m)
            np.testing.assert_allclose(axis[::substeps], self.clip.blade_direction, atol=1e-14)
            np.testing.assert_allclose(np.linalg.norm(axis, axis=1), 1., atol=1e-14)
            self.assertTrue(np.all(np.diff(phase) > 0))

    def test_substeps_validation(self):
        for value in (True, False, 0, 21, 1.5, "4"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                dense_reference(self.clip, value)

    def test_rotation_preserves_unit_axes_and_distances(self):
        for degrees in (-30., 0., 30.):
            rotation = rotation_z(degrees)
            np.testing.assert_allclose(rotation.T@rotation, np.eye(3), atol=1e-14)
            self.assertAlmostEqual(np.linalg.det(rotation), 1.)
            p, a = self.clip.sample(.3)
            neutral = np.array([.05, -.45, 1.05])
            self.assertAlmostEqual(np.linalg.norm(neutral+p@rotation.T-neutral), np.linalg.norm(p))
            self.assertAlmostEqual(np.linalg.norm(a@rotation.T), 1.)


class RunnerEvidenceTests(unittest.TestCase):
    def test_pose_acceptance_is_not_collision_or_motion_acceptance(self):
        result = result_fixture()
        report = metrics(result, {"grip_error_m": .02, "axis_error_deg": 5.,
                                  "tip_error_m": .05, "maximum_adjacent_joint_change_rad": .2})
        self.assertTrue(report["all_pose_samples_accepted"])
        self.assertEqual(report["pose_accepted"], 3)
        self.assertEqual(report["frames_with_interpolated_path_contacts"], 1)
        self.assertEqual(report["interpolated_path_contacts"][0]["frame"], 1)
        self.assertEqual(report["interpolated_joint_path_contact_samples"], 7)
        self.assertFalse(report["initialization_path_checked"])
        self.assertAlmostEqual(report["maximum_adjacent_joint_change_rad"], .02)
        self.assertTrue(report["joint_continuity_accepted"])
        self.assertFalse(result.success.all())

    def test_summary_separates_initialization_from_source_steps(self):
        summary = result_fixture().summary()
        self.assertEqual(summary["initialization_delta_rad"], [9., -8.])
        self.assertFalse(summary["initialization_path_checked"])
        self.assertAlmostEqual(summary["max_source_adjacent_step_rad"], .02)
        self.assertAlmostEqual(summary["max_joint_step_rad"], .02)

    def test_nonempty_output_is_rejected_before_any_data_or_model_work(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"preserve"
            output.mkdir()
            sentinel = output/"summary.json"
            sentinel.write_text('{"preserve": true}\n', encoding="utf-8")
            before = sentinel.read_bytes()
            with patch("plan1_skills.floor_test.build_library") as build, patch("sim.floor.build_floor_assembly") as model:
                with self.assertRaisesRegex(FileExistsError, "preserved"):
                    main(["--output", str(output), "--source-dir", str(Path(directory)/"missing"),
                          "--config", str(Path(directory)/"also-missing.json")])
                build.assert_not_called()
                model.assert_not_called()
            self.assertEqual(sentinel.read_bytes(), before)
            self.assertEqual(sorted(p.name for p in output.iterdir()), ["summary.json"])


class VerifiedLocalMappingTests(unittest.TestCase):
    """Optional integration test; reads immutable verified sources, never pickle."""

    def test_real_clips_use_one_mapping_and_nearly_return_to_guard(self):
        manifest = load_manifest()
        candidates = [DEFAULT_CACHE/"sources", ROOT/"outputs/plan1_skills_validation/downloaded_sources",
                      ROOT/"outputs/dataset_research/ase_audit"]
        source = next((p for p in candidates if all((p/s["filename"]).is_file() for s in manifest["sources"].values())), None)
        if source is None:
            self.skipTest("Verified local ASE sources absent; test never downloads data")
        with tempfile.TemporaryDirectory() as directory:
            metadata = build_library(source, Path(directory)/"templates")
            self.assertEqual(metadata["source_commit"], manifest["upstream"]["commit"])
            library = SkillLibrary(Path(directory)/"templates")
            config = json.loads((ROOT/"config/floor_match.json").read_text(encoding="utf-8"))
            # Exactly one rotation and translation applies to every source; do
            # not recenter each clip separately to conceal disconnected starts.
            rotation = rotation_z(config["world_from_skill_yaw_deg"])
            neutral = np.asarray(config["neutral_grip_world_m"])
            guard_p, guard_a = library.sample("guard", 0.)
            guard_p, guard_a = neutral+guard_p@rotation.T, guard_a@rotation.T
            self.assertEqual(len(library.clips["stab"].phase), 128)
            self.assertEqual(len(library.clips["parry_up"].phase), 58)
            for name in ("stab", "parry_up"):
                clip = library.clips[name]
                phase, p, a = dense_reference(clip, 4)
                np.testing.assert_array_equal(phase[::4], clip.phase)
                targets, axes = neutral+p@rotation.T, a@rotation.T
                for index in (0, -1):
                    self.assertLess(np.linalg.norm(targets[index]-guard_p), .005, name)
                    angle = np.rad2deg(np.arccos(np.clip(axes[index]@guard_a, -1, 1)))
                    self.assertLess(angle, .5, name)


if __name__ == "__main__":
    unittest.main()
