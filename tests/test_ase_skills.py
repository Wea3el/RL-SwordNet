"""Full-clip retargeting checks, without coupled physics or network access."""
from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from plan1_skills.library import ROOT, DEFAULT_CACHE, SkillClip, load_manifest
from sim.ase_skills import build_ase_library, guard_transition, inset_joint_ranges, load_ase_references, map_reference, screen_configurations


class ReferenceMappingTests(unittest.TestCase):
    def setUp(self):
        self.clip = SkillClip(np.array([0., .4, 1.]),
                              np.array([[0., 0., 0.], [2., 4., 6.], [.1, .2, .3]]),
                              np.tile([1., 0., 0.], (3, 1)))

    def test_one_rigid_mapping_and_exactly_one_displacement_scale(self):
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        phase, grip, axis = map_reference(self.clip, [1., 2., 3.], rotation, .5, 4)
        self.assertEqual(len(phase), 9)
        np.testing.assert_array_equal(phase[::4], self.clip.phase)
        np.testing.assert_allclose(grip[4], [-1., 3., 6.])
        np.testing.assert_allclose(grip[-1], [.9, 2.05, 3.15])
        np.testing.assert_allclose(axis, np.tile([0., 1., 0.], (9, 1)))

    def test_improper_rotation_and_invalid_scale_are_rejected(self):
        for rotation in (np.diag([-1., 1., 1.]), np.ones((3, 3)), np.eye(2)):
            with self.subTest(rotation=rotation), self.assertRaises(ValueError):
                map_reference(self.clip, [0., 0., 0.], rotation, .5)
        for scale in (0., -1., np.nan):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                map_reference(self.clip, [0., 0., 0.], np.eye(3), scale)

    def test_authored_guard_transition_is_separate_and_endpoint_exact(self):
        start, end = np.arange(7.), np.arange(7.)+.1
        transition = guard_transition(start, end, 11)
        np.testing.assert_array_equal(transition["qpos"][0], start)
        np.testing.assert_array_equal(transition["qpos"][-1], end)
        np.testing.assert_allclose(transition["qpos"][5], .5*(start+end))
        self.assertEqual(transition["qpos"].shape, (11, 7))

    def test_forward_only_convention_rejected_before_reading_sources(self):
        with patch("sim.ase_skills.load_ase_references") as sources:
            with self.assertRaisesRegex(ValueError, "full native clips"):
                build_ase_library({"path_convention": "forward-only-then-reverse"})
            sources.assert_not_called()

    def test_missing_explicit_cache_does_not_download_or_fallback(self):
        with tempfile.TemporaryDirectory() as directory, patch("plan1_skills.library.urlopen", side_effect=AssertionError("No network allowed")) as network:
            with self.assertRaises(FileNotFoundError):
                load_ase_references(directory)
            network.assert_not_called()


class JointMarginTests(unittest.TestCase):
    def test_inset_only_changes_solver_copy_not_actual_scene_bounds(self):
        actual = np.tile([-np.pi, np.pi], (7, 1))
        original = actual.copy()
        inset = inset_joint_ranges(actual, .05)
        np.testing.assert_array_equal(actual, original)
        np.testing.assert_allclose(inset[:, 0], -np.pi+.05)
        np.testing.assert_allclose(inset[:, 1], np.pi-.05)
        np.testing.assert_array_equal(inset_joint_ranges(actual, 0.), actual)

    def test_invalid_margin_rejected_before_source_reads(self):
        for margin in (-.01, True, False, "0.05", None, np.nan, np.inf, np.pi, 4.):
            with self.subTest(margin=margin), patch("sim.ase_skills.load_ase_references") as sources:
                with self.assertRaisesRegex(ValueError, "ik_joint_margin_rad"):
                    build_ase_library({"ik_joint_margin_rad": margin})
                sources.assert_not_called()
        with self.assertRaisesRegex(ValueError, "empty joint interval"):
            inset_joint_ranges([[-.04, .04]], .05)

    def test_screen_reports_real_limit_margin_and_does_not_hide_failed_rows(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <body><joint name="j" limited="true" range="-1 1"/>
            <geom size=".01" mass="1"/>
          </body></worldbody><actuator>
            <position joint="j" forcelimited="true" forcerange="-100 100"/>
          </actuator><compiler angle="radian"/></mujoco>''')
        assembly = type("Fixture", (), {"model": model})()
        before = model.jnt_range.copy()
        report = screen_configurations(assembly, [[.95], [.96]], .05)
        self.assertEqual(report["samples"], 2)
        self.assertAlmostEqual(report["ik_joint_margin_rad"], .05)
        self.assertAlmostEqual(report["minimum_true_joint_margin_rad"], .04)
        self.assertEqual(report["joint_margin_failed_indices"], [1])
        np.testing.assert_array_equal(model.jnt_range, before)


class VerifiedFullClipTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        manifest = load_manifest()
        candidates = [DEFAULT_CACHE/"sources", ROOT/"outputs/plan1_skills_validation/downloaded_sources"]
        cls.source = next((p for p in candidates if all((p/s["filename"]).is_file() for s in manifest["sources"].values())), None)
        if cls.source is None:
            raise unittest.SkipTest("Verified local ASE sources absent; never downloads test data")
        cls.library = build_ase_library(source_dir=cls.source)

    def test_full_native_clips_and_source_end_times_are_preserved(self):
        library = self.library
        for name, q, phase, source_time, frames in (
                ("stab", library.attack_q, library.attack_phase, library.attack_source_time_s, 128),
                ("parry_up", library.parry_q, library.parry_phase, library.parry_source_time_s, 58)):
            self.assertEqual(q.shape, ((frames-1)*4+1, 7))
            report = library.reports[name]
            self.assertTrue(report["all_original_frames_retained"])
            self.assertEqual(report["original_source_frames"], frames)
            np.testing.assert_allclose(phase[report["source_phase_indices"]], np.linspace(0., 1., frames), atol=0., rtol=0.)
            self.assertTrue(np.all(np.diff(source_time) > 0))
            self.assertGreater(source_time[-1], 4. if name == "stab" else 1.8)
            self.assertTrue(report["candidate_passes_single_arm_screen"], report)

    def test_guard_transitions_connect_full_clip_without_reversing_it(self):
        library = self.library
        for prefix, guard, q in (("attack", library.guard_attack_q, library.attack_q),
                                 ("parry", library.guard_defend_q, library.parry_q)):
            incoming, outgoing = library.transitions[prefix+"_in"], library.transitions[prefix+"_out"]
            np.testing.assert_array_equal(incoming["qpos"][0], guard)
            np.testing.assert_array_equal(incoming["qpos"][-1], q[0])
            np.testing.assert_array_equal(outgoing["qpos"][0], q[-1])
            np.testing.assert_array_equal(outgoing["qpos"][-1], guard)
            self.assertEqual(len(outgoing["qpos"]), 61)
            self.assertNotEqual(len(outgoing["qpos"]), len(q))
        for grip in (library.attack_grip_local_m, library.parry_grip_local_m):
            excursion = np.linalg.norm(grip-grip[0], axis=1)
            self.assertGreater(excursion.max(), 10*excursion[-1])

    def test_source_provenance_is_verified_and_sources_remain_unchanged(self):
        provenance = self.library.provenance
        self.assertFalse(provenance["source_files_modified"])
        self.assertEqual(provenance["physical_execution"], "NOT RUN by this builder")
        for name, source in provenance["sources"].items():
            path = Path(provenance["source_paths"][name])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), source["sha256"])
            self.assertEqual(path.stat().st_size, source["size_bytes"])

    def test_default_margin_applies_to_guard_native_and_authored_connections(self):
        library = self.library
        self.assertEqual(library.config["ik_joint_margin_rad"], .05)
        np.testing.assert_allclose(library.single_assembly.model.jnt_range,
                                   np.tile([-np.pi, np.pi], (7, 1)), atol=0., rtol=0.)
        screens = [library.reports[name] for name in ("stab", "parry_up", "attack_in", "attack_out", "parry_in", "parry_out")]
        screens += [library.reports[name]["guard_static_screen"] for name in ("stab", "parry_up")]
        for report in screens:
            self.assertEqual(report["ik_joint_margin_rad"], .05)
            self.assertGreaterEqual(report["minimum_true_joint_margin_rad"], .05-1e-8)
            self.assertEqual(report["joint_margin_failed_indices"], [])


if __name__ == "__main__":
    unittest.main()
