"""ASE pilot provenance and failure retention; no private motion files needed."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from plan1_skills.matching import MatchResult
from sim.ase_pilot import (SKILL_IDS, _cached_motion, build_pilot, export_pilot,
                           fit_pilot, load_config)


def synthetic_motion(source):
    count = source["frames"]
    phase = np.linspace(0., 1., count)
    angle = .2*np.sin(2*np.pi*phase)
    return {"frames": count, "fps": source["fps"], "time": np.arange(count)/source["fps"],
            "grip_m": np.column_stack([.4+.1*np.sin(np.pi*phase), np.zeros(count), .8+.02*phase]),
            "direction": np.column_stack([np.cos(angle), np.sin(angle), np.zeros(count)]),
            "root_m": np.zeros((count, 3)),
            "shoulder_l_m": np.tile([.2, 0., 1.], (count, 1)),
            "shoulder_r_m": np.tile([-.2, 0., 1.], (count, 1))}


def fake_source(source, directories):
    return synthetic_motion(source), Path("/verified/cache")/source["filename"]


class PilotTests(unittest.TestCase):
    def make_library(self):
        with patch("sim.ase_pilot._cached_motion", side_effect=fake_source):
            return build_pilot()

    def test_five_skills_explicit_derived_status_and_authored_common_seams(self):
        library = self.make_library()
        self.assertEqual(tuple(library.trajectories), SKILL_IDS)
        self.assertFalse(library.metadata["may_start_RL"])
        self.assertFalse(library.metadata["admitted_to_training"])
        self.assertFalse(library.metadata["runtime_opponent_inputs"])
        self.assertFalse(library.metadata["physics_model_parameters_modified"])
        self.assertEqual(library.metadata["policy_family"], "A1")
        self.assertEqual(library.metadata["action_space_kind"], "finite_skill_selection")
        self.assertTrue(library.metadata["robot_slot_independent"])
        self.assertFalse(library.metadata["a2_continuous_action_space"])
        self.assertIn("torso rotation NOT removed", library.metadata["fixed_base_approximation"])
        for name, trajectory in library.trajectories.items():
            self.assertTrue(np.all(np.diff(trajectory.time_s) > 0))
            np.testing.assert_allclose(np.linalg.norm(trajectory.axis_local, axis=1), 1.)
            np.testing.assert_array_equal(trajectory.grip_local_m[0], library.config["guard_grip_local_m"])
            np.testing.assert_array_equal(trajectory.grip_local_m[-1], library.config["guard_grip_local_m"])
            np.testing.assert_allclose(trajectory.axis_local[[0, -1]], [library.metadata["guard_axis_local"]]*2)
            metadata = library.metadata["skills"][name]
            self.assertEqual(metadata["policy_family"], "A1")
            self.assertFalse(metadata["admitted_to_training"])
            self.assertEqual(metadata["physical_attack_or_defence_validation"], "NOT RUN")
            if name != "guard":
                self.assertEqual(trajectory.segments[0]["name"], "authored_guard_entry")
                self.assertEqual(trajectory.segments[-1]["name"], "authored_guard_exit")
                self.assertIsNone(trajectory.segments[0]["source_frame_interval"])
                source_pieces = [s for s in trajectory.segments if s["source_frame_interval"] is not None]
                self.assertEqual(len(source_pieces), len(metadata["zero_based_keyframes"])-1)

    def test_keyposes_scaling_once_and_timing_are_explicit_not_native_playback(self):
        library = self.make_library()
        trajectory = library.trajectories["stab"]
        meta = library.metadata["skills"]["stab"]
        first = trajectory.segments[1]
        source = np.asarray(meta["selected_source_grip_in_fixed_body_frame_m"])
        expected = np.asarray(library.config["guard_grip_local_m"])+.5*(source[1]-source[0])
        np.testing.assert_allclose(trajectory.grip_local_m[first["end_row"]], expected)
        self.assertAlmostEqual(first["end_s"]-first["start_s"], .8)
        self.assertNotAlmostEqual(meta["selected_source_time_s"][1], .8)
        self.assertFalse(meta["all_native_frames_retained"])

    def test_missing_cache_and_corrupt_first_cache_do_not_download_or_fallback(self):
        source = load_config()["sources"]["ready"]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                _cached_motion(source, [directory])
            path = Path(directory)/source["filename"]
            path.touch()
            with self.assertRaisesRegex(ValueError, "size"):
                _cached_motion(source, [directory])

    def test_bad_rotation_keyframes_and_tool_resize_rejected(self):
        for override in ({"rotation_matrix": [[-1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]},
                         {"robot_grip_to_tip_m": .74}, {"keypose_interval_s": True}):
            with self.assertRaises(ValueError):
                load_config(override)
        cfg = load_config()
        cfg["skills"][1]["keyframes"] = [0, 128]
        with self.assertRaises(ValueError):
            load_config(cfg)

    def test_a1_library_cannot_be_relabelled_as_a2_continuous_actions(self):
        for override in ({"policy_family": "A2"}, {"action_space_kind": "continuous_joint_increment"},
                         {"robot_slot_independent": False}, {"a2_continuous_action_space": True}):
            with self.assertRaisesRegex(ValueError, "A1 policy family"):
                load_config(override)

    def test_optional_guard_seed_requires_finite_seven_values_inside_inset_bounds(self):
        self.assertNotIn("guard_seed_rad", load_config())
        accepted = [.1, -.2, .3, -.4, .5, -.6, .7]
        self.assertEqual(load_config({"guard_seed_rad": accepted})["guard_seed_rad"], accepted)
        for seed in ([0.]*6, [0.]*8, [float("nan")]*7, [float("inf")]*7,
                     [np.pi]*7, [-np.pi]*7, [True]*7, ["0"]*7):
            with self.assertRaisesRegex(ValueError, "guard_seed_rad"):
                load_config({"guard_seed_rad": seed})

    def test_explicit_seed_is_passed_without_multistart_and_default_retains_twelve(self):
        model = SimpleNamespace(jnt_range=np.tile([-np.pi, np.pi], (7, 1)))
        assembly = SimpleNamespace(model=model, source_hashes={"fixture": "unchanged"})
        seed = [.1, -.2, .3, -.4, .5, -.6, .7]
        for supplied, starts in ((seed, 1), (None, 12)):
            with patch("sim.ase_pilot._cached_motion", side_effect=fake_source):
                library = build_pilot({"guard_seed_rad": supplied})
            with patch("sim.ase_pilot.build_floor_assembly", return_value=assembly), \
                 patch("sim.ase_pilot.tool_fk", return_value=(np.zeros(3), np.array([1., 0., 0.]), np.array([.79, 0., 0.]))), \
                 patch("sim.ase_pilot.match_sequence", side_effect=RuntimeError("stop after guard initialization")) as solver:
                with self.assertRaisesRegex(RuntimeError, "stop after guard"):
                    fit_pilot(library)
            self.assertEqual(solver.call_args.kwargs["initial_q"], supplied)
            self.assertEqual(solver.call_args.kwargs["first_frame_starts"], starts)

    def test_optional_ik_retains_failed_rows_and_never_admits_library(self):
        library = self.make_library()
        model = SimpleNamespace(jnt_range=np.tile([-np.pi, np.pi], (7, 1)))
        assembly = SimpleNamespace(model=model, source_hashes={"fixture": "unchanged"})

        def failure(model, grip, axes, bounds, **kwargs):
            count = len(grip)
            qs = np.zeros((count, 7))
            return MatchResult(qs, np.ones(count), np.ones(count), np.ones(count), qs.copy(),
                               np.zeros(count, dtype=bool), np.ones(count, dtype=bool),
                               np.zeros(count, dtype=bool), [[] for _ in range(count)],
                               np.ones(count, dtype=int), np.ones(count, dtype=int))

        with patch("sim.ase_pilot.build_floor_assembly", return_value=assembly), \
             patch("sim.ase_pilot.tool_fk", return_value=(np.zeros(3), np.array([1., 0., 0.]), np.array([.79, 0., 0.]))), \
             patch("sim.ase_pilot._screen", return_value={"candidate_passes_geometry_screen": True}), \
             patch("sim.ase_pilot.match_sequence", side_effect=failure):
            fit_pilot(library)
        for name, trajectory in library.trajectories.items():
            self.assertEqual(len(library.ik[name].qpos), len(trajectory.time_s))
            self.assertFalse(library.metadata["skills"][name]["geometry_candidate_only"])
            self.assertEqual(library.metadata["skills"][name]["ik"]["failed_frame_indices"], list(range(len(trajectory.time_s))))
        self.assertFalse(library.metadata["may_start_RL"])
        self.assertFalse(library.metadata["admitted_to_training"])
        self.assertFalse(library.metadata["joint_seams_certified"])

    def test_successful_ik_cannot_override_failed_strict_geometry_screen(self):
        library = self.make_library()
        model = SimpleNamespace(jnt_range=np.tile([-np.pi, np.pi], (7, 1)))
        assembly = SimpleNamespace(model=model, source_hashes={"fixture": "unchanged"})

        def success(model, grip, axes, bounds, **kwargs):
            count = len(grip)
            qs = np.zeros((count, 7))
            return MatchResult(qs, np.zeros(count), np.zeros(count), np.zeros(count), qs.copy(),
                               np.ones(count, dtype=bool), np.ones(count, dtype=bool),
                               np.ones(count, dtype=bool), [[] for _ in range(count)],
                               np.ones(count, dtype=int), np.ones(count, dtype=int))

        with patch("sim.ase_pilot.build_floor_assembly", return_value=assembly), \
             patch("sim.ase_pilot.tool_fk", return_value=(np.zeros(3), np.array([1., 0., 0.]), np.array([.79, 0., 0.]))), \
             patch("sim.ase_pilot._screen", return_value={"candidate_passes_geometry_screen": False}), \
             patch("sim.ase_pilot.match_sequence", side_effect=success) as solver:
            fit_pilot(library)
        self.assertEqual(solver.call_args.kwargs["collision_threshold_m"], 1e-8)
        for name in library.trajectories:
            self.assertFalse(library.metadata["skills"][name]["geometry_candidate_only"])

    def test_export_numeric_arrays_and_refuse_overwrite(self):
        library = self.make_library()
        with tempfile.TemporaryDirectory() as directory:
            report = export_pilot(library, directory)
            metadata = json.loads(report.read_text())
            self.assertFalse(metadata["may_start_RL"])
            self.assertEqual(metadata["policy_family"], "A1")
            self.assertEqual(metadata["config"]["schema_version"], 2)
            with np.load(Path(directory)/"stab.npz", allow_pickle=False) as arrays:
                np.testing.assert_array_equal(arrays["time_s"], library.trajectories["stab"].time_s)
            with self.assertRaises(FileExistsError):
                export_pilot(library, directory)


if __name__ == "__main__":
    unittest.main()
