"""Authored defence construction contracts, not successful physical parries."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import unittest

import numpy as np

from sim.a1_defence import build_defence_candidate, task_targets, validate_recipe
from sim.a1_skills import export_a1_library, load_a1_library
from tests.test_a1_adaptation import SCREEN, fixture, result_from


def recipe():
    return {"schema_version": 1, "skill_id": "parry_right", "samples_per_segment": 9,
        "keyposes": [
            {"name": "upright", "grip_local_m": [.4, 0., .75], "axis_local": [0., 0., 1.], "duration_s": 3.},
            {"name": "finite_hold", "grip_local_m": [.4, 0., .75], "axis_local": [0., 0., 1.], "duration_s": 4.},
        ]}


def fake_fit(model, grips, axes, bounds, **kwargs):
    q = np.tile(kwargs["initial_q"], (len(grips), 1))
    q[:, 0] += .1*np.linspace(0., 1., len(grips))
    return result_from(q)


class DefenceRecipeTests(unittest.TestCase):
    def test_observable_candidate_removes_ready_wait_but_remains_only_a_recipe(self):
        path = Path(__file__).resolve().parents[1]/"config/a1_defence_observable_stab_candidate.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        cfg = validate_recipe(value)
        phases = {p["name"]: p for p in cfg["keyposes"]}
        self.assertNotIn("finite_ready_hold", phases)
        self.assertEqual(phases["finite_protective_hold"]["duration_s"], 9.)
        self.assertEqual(sum(p["duration_s"] for p in cfg["keyposes"]), 24.)
        # The existing explicit 2-second joint return makes this a 26s
        # candidate. This timing contract is NOT a native defence proof.
        self.assertEqual(cfg["skill_id"], "parry_right")
        self.assertNotIn("admitted_to_training", cfg)

    def test_requires_explicit_recipe_and_rejects_physics_or_solver_overrides(self):
        for value in (None, {}, dict(recipe(), torque_caps=[100]*7),
                      dict(recipe(), collision_threshold_m=.001), dict(recipe(), skill_id="stab"),
                      dict(recipe(), samples_per_segment=True), dict(recipe(), solver_max_iterations=True)):
            with self.subTest(recipe=value), self.assertRaises(ValueError):
                validate_recipe(value)

    def test_finite_unit_targets_and_unique_phase_names_required(self):
        for changes in ({"axis_local": [0., 0., 2.]}, {"grip_local_m": [np.nan, 0., 1.]},
                        {"duration_s": 0.}, {"duration_s": True}, {"duration_s": 100.}):
            value = recipe(); value["keyposes"][0].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_recipe(value)
        value = recipe(); value["keyposes"][1]["name"] = "upright"
        with self.assertRaises(ValueError):
            validate_recipe(value)

    def test_task_clock_is_authored_seconds_and_guard_first_target_is_exact_fk(self):
        value = recipe(); before = copy.deepcopy(value)
        with patch("sim.a1_defence.tool_fk", return_value=(np.array([.1, .2, .3]), np.array([1., 0., 0.]), None)):
            time, grip, axis, segments = task_targets(None, np.zeros(7), value)
        self.assertEqual(len(time), 17)
        self.assertEqual(time[0], 0.)
        self.assertEqual(time[-1], 7.)
        self.assertTrue(np.all(np.diff(time) > 0))
        np.testing.assert_array_equal(grip[0], [.1, .2, .3])
        np.testing.assert_array_equal(axis[0], [1., 0., 0.])
        np.testing.assert_allclose(axis[-1], [0., 0., 1.])
        self.assertEqual(segments[1]["start_row"], 8)
        self.assertIsNone(segments[0]["source_frame_interval"])
        self.assertEqual(value, before)


class DefenceBuilderTests(unittest.TestCase):
    def build(self, base, fit=None, screens=None):
        with patch("sim.a1_defence.tool_fk", return_value=(np.array([.1, .2, .3]), np.array([1., 0., 0.]), None)), \
             patch("sim.a1_defence.match_sequence", side_effect=fake_fit if fit is None else None, return_value=fit), \
             patch("sim.a1_defence._screen", side_effect=[SCREEN, SCREEN] if screens is None else screens):
            return build_defence_candidate(base, recipe())

    def test_preserves_existing_sources_and_four_other_runtime_skills(self):
        base = fixture(); originals = copy.deepcopy(base.source_arrays)
        result = self.build(base)
        for name in base.skills:
            for key, source in originals[name].items():
                np.testing.assert_array_equal(result.source_arrays[name][key], source)
                np.testing.assert_array_equal(base.source_arrays[name][key], source)
            if name != "parry_right":
                self.assertIs(result.skills[name], base.skills[name])
        self.assertNotIn("authored_defence_qpos", base.source_arrays["parry_right"])
        self.assertIn("authored_defence_qpos", result.source_arrays["parry_right"])
        self.assertTrue(result.reports["library"]["all_skills_pass_geometry"])
        self.assertFalse(result.reports["library"]["admitted_to_training"])
        self.assertFalse(result.reports["library"]["may_start_RL"])
        self.assertTrue(result.reports["parry_right"]["authored_defence_not_original_ASE"])
        path = result.skills["parry_right"]
        self.assertEqual(path.segments[-1]["name"], "authored_joint_return")
        self.assertEqual(path.segments[-1]["start_s"], 7.)  # NOT 28 seconds
        np.testing.assert_array_equal(path.sample(0.), base.guard_q)
        np.testing.assert_array_equal(path.sample(path.duration_s), base.guard_q)
        self.assertIn("parry_right", result.provenance["authored_defence_recipes"])

    def test_failed_ik_or_smooth_screen_is_retained_not_admitted(self):
        for mode in ("ik", "smooth"):
            base = fixture()
            q = np.zeros((17, 7))
            failed = result_from(q, success=mode != "ik")
            screens = [SCREEN, dict(SCREEN, candidate_passes_geometry_screen=mode != "smooth")]
            result = self.build(base, fit=failed, screens=screens)
            self.assertFalse(result.reports["library"]["all_skills_pass_geometry"])
            self.assertFalse(result.reports["library"]["admitted_to_training"])
            self.assertEqual(len(result.source_arrays["parry_right"]["authored_defence_qpos"]), 17)
            self.assertEqual(result.reports["parry_right"]["ik"]["frames"], 17)

    def test_export_reload_preserves_new_and_original_numeric_data(self):
        base = fixture(); result = self.build(base)
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)/"candidate"
            export_a1_library(result, folder)
            with patch("sim.a1_skills.build_floor_assembly", return_value=base.single_assembly), \
                 patch("sim.a1_skills._screen", return_value=SCREEN):
                loaded = load_a1_library(folder)
            np.testing.assert_array_equal(loaded.skills["parry_right"].qpos, result.skills["parry_right"].qpos)
            np.testing.assert_array_equal(loaded.source_arrays["parry_right"]["qpos"], base.source_arrays["parry_right"]["qpos"])
            self.assertIn("authored_defence_qpos", loaded.source_arrays["parry_right"])
            self.assertEqual(loaded.provenance["authored_defence_recipes"], result.provenance["authored_defence_recipes"])


if __name__ == "__main__":
    unittest.main()
