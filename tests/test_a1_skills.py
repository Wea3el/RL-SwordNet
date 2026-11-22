"""A1-family closed command bank, strict provenance and C2 seam tests."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from plan1_skills.ase_numeric import UPSTREAM_COMMIT
from plan1_skills.matching import tool_fk
from sim.a1_skills import (SKILL_IDS, _check_curve, _curve_sample, _hermite_coefficients, build_a1_library,
                           export_a1_library, load_a1_library)
from sim.floor import build_floor_assembly


class CurveTests(unittest.TestCase):
    def test_hermite_preserves_all_knots_and_internal_derivatives_not_all_zero(self):
        times = np.array([0., .5, 1.5, 2.])
        q = np.tile(np.array([0., .1, .25, .3])[:, None], (1, 7))
        coefficients = _hermite_coefficients(times, q)
        for t, target in zip(times, q):
            np.testing.assert_array_equal(_curve_sample(times, q, coefficients, t), target)
        self.assertGreater(np.max(np.abs(_curve_sample(times, q, coefficients, .5, 1))), 0.)
        for t in times[1:-1]:
            for derivative in (0, 1, 2):
                np.testing.assert_allclose(_curve_sample(times, q, coefficients, t-1e-9, derivative),
                                           _curve_sample(times, q, coefficients, t+1e-9, derivative), atol=1e-7)
        for t in (0., 2.):
            np.testing.assert_array_equal(_curve_sample(times, q, coefficients, t, 1), np.zeros(7))
            np.testing.assert_array_equal(_curve_sample(times, q, coefficients, t, 2), np.zeros(7))

    def test_curve_integrity_rejects_hidden_derivative_or_endpoint_discontinuity(self):
        times = np.array([0., 1., 2.])
        q = np.tile(np.array([0., .1, 0.])[:, None], (1, 7))
        coefficients = _hermite_coefficients(times, q)
        self.assertLess(_check_curve(times, q, coefficients)["polynomial_max_velocity_jump_rad_s"], 1e-8)
        coefficients[1, 1, 0] += .001
        with self.assertRaisesRegex(ValueError, "C2"):
            _check_curve(times, q, coefficients)


class A1BankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assembly = build_floor_assembly(joint_ranges=np.tile([-np.pi, np.pi], (7, 1)))
        cls.guard = np.array([.9101707278193188, -1.1975924110585945, -2.056041917989702,
                              -2.464070537878126, .3705722336381225, .5394505284391571, -.12242644021395901])

    def write_pilot(self, directory):
        folder = Path(directory)
        metadata = {"schema_version": 2, "status": "candidate_only", "policy_family": "A1",
                    "action_space_kind": "finite_skill_selection", "robot_slot_independent": True,
                    "a2_continuous_action_space": False, "admitted_to_training": False, "may_start_RL": False,
                    "source_commit": UPSTREAM_COMMIT, "robot_grip_to_tip_m": .79,
                    "unchanged_model_source_hashes": self.assembly.source_hashes,
                    "config": {"policy_family": "A1"}, "skills": {}, "segments": {}}
        for name in SKILL_IDS:
            q = np.tile(self.guard, (4, 1))
            if name != "guard":
                q[:, 0] += [0., 1e-5, 2e-5, 1e-5]
            t = np.array([0., .1, .2, .4])
            poses = [tool_fk(self.assembly.model, qpos=pose) for pose in q]
            np.savez_compressed(folder/f"{name}.npz", time_s=t, qpos=q,
                                grip_local_m=np.array([pose[0] for pose in poses]),
                                axis_local=np.array([pose[1] for pose in poses]),
                                pose_success=np.ones(4, dtype=bool), collision_free=np.ones(4, dtype=bool),
                                source_frame_coordinate=np.arange(4, dtype=float))
            metadata["skills"][name] = {"policy_family": "A1", "geometry_candidate_only": True,
                "all_authored_rows_retained": True, "ik": {"all_frames_passed": True},
                "strict_static_and_interpolated_screen": {"candidate_passes_geometry_screen": True, "all_raw_contacts": []}}
            metadata["segments"][name] = [{"name": "synthetic_source", "start_s": 0., "end_s": .4,
                                            "start_row": 0, "end_row": 3, "source_frame_interval": [0, 3]}]
        (folder/"pilot.json").write_text(json.dumps(metadata))
        return metadata

    def test_closed_family_bank_exact_seams_source_preservation_and_curve_derivatives(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write_pilot(directory)
            library = build_a1_library(directory)
        self.assertEqual(tuple(library.skills), SKILL_IDS)
        self.assertTrue(library.reports["library"]["all_skills_pass_geometry"])
        self.assertFalse(library.reports["library"]["admitted_to_training"])
        self.assertFalse(library.reports["library"]["may_start_RL"])
        for index, (name, path) in enumerate(library.skills.items()):
            self.assertEqual(path.action_id, index)
            np.testing.assert_array_equal(path.qpos[0], library.guard_q)
            np.testing.assert_array_equal(path.qpos[-1], library.guard_q)
            np.testing.assert_array_equal(path.sample(-1.), library.guard_q)
            np.testing.assert_array_equal(path.sample(path.duration_s), library.guard_q)
            np.testing.assert_array_equal(path.sample(path.duration_s+10), library.guard_q)
            for t, q in zip(library.source_arrays[name]["time_s"], library.source_arrays[name]["qpos"]):
                np.testing.assert_array_equal(path.sample(t), q)
            self.assertEqual(len(path.segment_labels), len(path.qpos)-1)
            if name != "guard":
                self.assertEqual(path.segments[-1]["name"], "authored_joint_return")
                self.assertGreater(path.duration_s, .4)
                for t in (.4-1e-9, .4+1e-9):
                    velocity, acceleration = path.derivatives(t)
                    np.testing.assert_allclose(velocity, np.zeros(7), atol=1e-7)
                    np.testing.assert_allclose(acceleration, np.zeros(7), atol=1e-7)

    def test_uniform_retiming_is_explicit_and_keeps_original_clock(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write_pilot(directory)
            library = build_a1_library(directory, {"source_time_scale": 4.})
        path = library.skills["stab"]
        self.assertEqual(path.segments[0]["source_end_s"], .4)
        self.assertEqual(path.segments[0]["end_s"], 1.6)
        np.testing.assert_array_equal(library.source_arrays["stab"]["time_s"], [0., .1, .2, .4])
        for t, q in zip(library.source_arrays["stab"]["time_s"], library.source_arrays["stab"]["qpos"]):
            np.testing.assert_array_equal(path.sample(4*t), q)

    def test_role_geometry_model_and_common_start_corruption_rejected(self):
        for mode in ("role", "geometry", "model", "start", "fk"):
            with tempfile.TemporaryDirectory() as directory:
                metadata = self.write_pilot(directory)
                if mode == "role":
                    metadata["policy_family"] = "A2"
                elif mode == "geometry":
                    metadata["skills"]["stab"]["geometry_candidate_only"] = False
                elif mode == "model":
                    metadata["unchanged_model_source_hashes"] = {}
                elif mode in {"start", "fk"}:
                    path = Path(directory)/"stab.npz"
                    with np.load(path, allow_pickle=False) as archive:
                        arrays = {key: archive[key].copy() for key in archive.files}
                    if mode == "start":
                        arrays["qpos"][0, 0] += .01
                    else:
                        arrays["grip_local_m"][1, 0] += .01
                    np.savez_compressed(path, **arrays)
                (Path(directory)/"pilot.json").write_text(json.dumps(metadata))
                with self.assertRaises(ValueError):
                    build_a1_library(directory)

    def test_smoothing_geometry_failure_is_retained_not_admitted(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write_pilot(directory)
            from sim.a1_skills import _screen as real_screen
            calls = 0
            def screening(*args, **kwargs):
                nonlocal calls
                calls += 1
                report = real_screen(*args, **kwargs)
                if calls > 5:
                    report["candidate_passes_geometry_screen"] = False
                return report
            with patch("sim.a1_skills._screen", side_effect=screening):
                library = build_a1_library(directory)
        self.assertFalse(library.reports["library"]["all_skills_pass_geometry"])
        self.assertEqual(tuple(library.skills), SKILL_IDS)
        self.assertFalse(library.reports["library"]["admitted_to_training"])

    def test_roundtrip_numeric_bank_and_tamper_or_overwrite_rejected(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as output:
            self.write_pilot(directory)
            library = build_a1_library(directory)
            export_a1_library(library, output)
            loaded = load_a1_library(output)
            for name in SKILL_IDS:
                np.testing.assert_array_equal(loaded.skills[name].qpos, library.skills[name].qpos)
                for time in np.linspace(0., library.skills[name].duration_s, 11):
                    np.testing.assert_array_equal(loaded.skills[name].sample(time), library.skills[name].sample(time))
            with self.assertRaises(FileExistsError):
                export_a1_library(library, output)
            path = Path(output)/"stab.npz"
            path.write_bytes(path.read_bytes()+b"tamper")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                load_a1_library(output)

    def test_unrequested_scope_changes_and_nonfinite_clock_rejected(self):
        for cfg in ({"policy_family": "A2"}, {"source_time_scale": .5}, {"source_time_scale": float("nan")},
                    {"ik_joint_margin_rad": 0.}, {"hidden_opponent_oracle": True}):
            with self.assertRaises(ValueError):
                build_a1_library("unused", cfg)


if __name__ == "__main__":
    unittest.main()
