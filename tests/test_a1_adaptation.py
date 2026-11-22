"""Authored adaptation provenance and bounded continuation, no motion cache."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from plan1_skills.matching import MatchResult
from sim.a1_adaptation import (DEFAULT_RECIPE, adapt_a1_library, authored_grip_offsets,
                               build_adapted_a1_library, source_guided_match)
from sim.a1_skills import (A1Library, SKILL_IDS, _build_path, _config, export_a1_library,
                           load_a1_library)


def result_from(q, success=True):
    q = np.asarray(q, dtype=float)
    count = len(q)
    mask = np.full(count, success, dtype=bool)
    return MatchResult(q.copy(), np.zeros(count), np.zeros(count), np.zeros(count), np.zeros_like(q),
                       mask, np.ones(count, dtype=bool), mask.copy(), [[] for _ in range(count)],
                       np.ones(count, dtype=int), np.ones(count, dtype=int))


def fixture():
    cfg = _config({"source_time_scale": 4.})
    guard = np.zeros(7)
    times = np.linspace(0., 6.4, 9)
    original = {"time_s": times, "qpos": np.zeros((9, 7)), "grip_local_m": np.tile([.4, 0., 1.], (9, 1)),
                "axis_local": np.tile([1., 0., 0.], (9, 1)), "source_frame_coordinate": np.linspace(0., 45., 9),
                "pose_success": np.ones(9, dtype=bool), "collision_free": np.ones(9, dtype=bool)}
    source_arrays, skills, reports, source_segments = {}, {}, {}, {}
    for index, name in enumerate(SKILL_IDS):
        source_arrays[name] = copy.deepcopy(original)
        segments = [{"name": "synthetic_source", "start_s": 0., "end_s": 6.4,
                     "start_row": 0, "end_row": 8, "source_frame_interval": [0, 45]}]
        source_segments[name] = segments
        skills[name] = _build_path(name, index, source_arrays[name], segments, guard, cfg)
        reports[name] = {"candidate_passes_skill_geometry": True, "exact_guard_start": True, "exact_guard_end": True}
    reports["library"] = {"status": "executable_candidate", "all_skills_pass_geometry": True,
                           "admitted_to_training": False, "may_start_RL": False}
    assembly = SimpleNamespace(model=SimpleNamespace(jnt_range=np.tile([-np.pi, np.pi], (7, 1))), source_hashes={"fixture": "unchanged"})
    provenance = {"source_pilot_metadata": {"segments": source_segments}, "input_sha256": {"pilot.json": "fixture"},
                  "unchanged_model_source_hashes": assembly.source_hashes, "generator_sha256": "synthetic_fixture"}
    return A1Library(cfg, assembly, guard, skills, reports, source_arrays, provenance)


def fake_continuation(model, base_grips, axes, original_q, guard, bounds, weights, recipe):
    q = original_q.copy()
    q[:, 0] += .01*weights
    return result_from(q), {"all_intermediate_solver_stages": [], "fixture": True}


SCREEN = {"candidate_passes_geometry_screen": True, "all_raw_contacts": [], "strict_static_contacts_free": True}


class OffsetTests(unittest.TestCase):
    def test_compact_support_uses_original_clock_and_preserves_guard(self):
        times = np.array([0., 2.1, 3.45, 4.8, 6.4])
        offsets, weights = authored_grip_offsets(times)
        np.testing.assert_allclose(weights, [0., 0., 1., 0., 0.], atol=1e-12)
        np.testing.assert_allclose(offsets[2], [.160, .0296, -.0848])
        np.testing.assert_array_equal(offsets[[0, -1]], np.zeros((2, 3)))
        original = DEFAULT_RECIPE.copy()
        with self.assertRaises(ValueError):
            authored_grip_offsets(times, {"half_width_source_s": 4.})
        with self.assertRaises(ValueError):
            authored_grip_offsets(times, {"collision_threshold_m": .001})
        with self.assertRaises(ValueError):
            authored_grip_offsets(times, {"amplitude": 100.})
        self.assertEqual(DEFAULT_RECIPE, original)

    def test_row_guidance_uses_each_original_branch_not_previous_time_row(self):
        count = 5
        original = np.zeros((count, 7)); original[:, 1] = np.arange(count)*.1
        before = original.copy()
        grips = np.zeros((count, 3)); axes = np.tile([1., 0., 0.], (count, 1))
        weights = np.array([0., .25, 1., .25, 0.])
        seeds = []
        def solve(model, targets, axes, bounds, *, initial_q, **options):
            seeds.append(initial_q.copy())
            q = initial_q.copy(); q[0] = targets[0, 0]
            return result_from(q[None])
        with patch("sim.a1_adaptation.match_sequence", side_effect=solve), \
             patch("sim.a1_adaptation.screen_joint_path", return_value=([], 2)) as screen:
            fit, history = source_guided_match(None, grips, axes, original, original[0],
                                               np.tile([-3., 3.], (7, 1)), weights)
        self.assertEqual(len(seeds), 8*count)
        self.assertEqual(screen.call_count, count)
        np.testing.assert_array_equal(np.asarray(seeds)[:, 1], np.tile(original[:, 1], 8))
        np.testing.assert_allclose(fit.qpos[:, 0], .8*.20*weights)
        np.testing.assert_array_equal(original, before)
        self.assertTrue(fit.success.all())
        self.assertEqual(len(history["all_intermediate_solver_stages"]), 8)


class AdaptedLibraryTests(unittest.TestCase):
    def test_optional_adapter_keeps_every_original_array_and_declares_authored_runtime(self):
        base = fixture()
        before = copy.deepcopy(base.source_arrays)
        with patch("sim.a1_adaptation.source_guided_match", side_effect=fake_continuation), \
             patch("sim.a1_adaptation._screen", return_value=SCREEN):
            adapted = adapt_a1_library(base)
        self.assertIsNot(adapted, base)
        self.assertEqual(adapted.config, base.config)
        for name in SKILL_IDS:
            for key, value in before[name].items():
                np.testing.assert_array_equal(adapted.source_arrays[name][key], value)
                np.testing.assert_array_equal(base.source_arrays[name][key], value)
            path = adapted.skills[name]
            np.testing.assert_array_equal(path.sample(0.), base.guard_q)
            np.testing.assert_array_equal(path.sample(path.duration_s), base.guard_q)
        self.assertIn("adapted_qpos", adapted.source_arrays["slash_left"])
        self.assertNotIn("adapted_qpos", base.source_arrays["slash_left"])
        self.assertFalse(adapted.reports["library"]["admitted_to_training"])
        self.assertFalse(adapted.reports["library"]["may_start_RL"])
        self.assertTrue(adapted.reports["slash_left"]["authored_strike_enhancement_not_original_ASE"])
        self.assertFalse(adapted.reports["slash_left"]["source_qpos_used_as_runtime_knots"])
        self.assertEqual(adapted.skills["slash_left"].segments[-1]["name"], "authored_joint_return")
        self.assertIn("adaptation_recipe", adapted.provenance)

    def test_failed_pose_or_smooth_collision_keeps_candidate_but_blocks_gate(self):
        for failure in ("pose", "smooth"):
            base = fixture()
            fit = result_from(base.source_arrays["slash_left"]["qpos"], success=failure != "pose")
            screens = [SCREEN, dict(SCREEN, candidate_passes_geometry_screen=failure != "smooth")]
            with patch("sim.a1_adaptation.source_guided_match", return_value=(fit, {})), \
                 patch("sim.a1_adaptation._screen", side_effect=screens):
                adapted = adapt_a1_library(base)
            self.assertFalse(adapted.reports["library"]["all_skills_pass_geometry"])
            self.assertEqual(len(adapted.source_arrays["slash_left"]["adapted_qpos"]), 9)
            self.assertFalse(adapted.reports["library"]["admitted_to_training"])

    def test_existing_numeric_export_load_preserves_original_and_adapted_fields(self):
        base = fixture()
        with patch("sim.a1_adaptation.source_guided_match", side_effect=fake_continuation), \
             patch("sim.a1_adaptation._screen", return_value=SCREEN):
            adapted = adapt_a1_library(base)
        with tempfile.TemporaryDirectory() as output:
            export_a1_library(adapted, output)
            with np.load(Path(output)/"slash_left.npz", allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["source_qpos"], base.source_arrays["slash_left"]["qpos"])
                self.assertIn("source_adapted_qpos", archive.files)
                self.assertIn("source_authored_grip_offset_m", archive.files)
            manifest = json.loads((Path(output)/"bank.json").read_text())
            self.assertIn("adaptation_recipe", manifest["provenance"])
            with patch("sim.a1_skills.build_floor_assembly", return_value=base.single_assembly), \
                 patch("sim.a1_skills._screen", return_value=SCREEN):
                loaded = load_a1_library(output)
            np.testing.assert_array_equal(loaded.source_arrays["slash_left"]["qpos"], base.source_arrays["slash_left"]["qpos"])
            np.testing.assert_array_equal(loaded.skills["slash_left"].qpos, adapted.skills["slash_left"].qpos)

    def test_builder_defaults_uniform_four_but_keeps_explicit_configuration(self):
        with patch("sim.a1_adaptation.build_a1_library", return_value=fixture()) as build, \
             patch("sim.a1_adaptation.adapt_a1_library", return_value="candidate"):
            self.assertEqual(build_adapted_a1_library("pilot"), "candidate")
            self.assertEqual(build.call_args.args[1], {"source_time_scale": 4.})
            build_adapted_a1_library("pilot", {"source_time_scale": 12.})
            self.assertEqual(build.call_args.args[1], {"source_time_scale": 12.})


if __name__ == "__main__":
    unittest.main()
