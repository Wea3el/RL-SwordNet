"""Independent observation, paired-evidence, and native runner regressions.

These short tests do not establish autonomous defence, continuous counterplay,
motor capability, or RL readiness. Full opportunity sweeps remain diagnostics.
"""
from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from sim.duel_scene import build_duel_scene
from sim.shared_control import SkillController, ScriptedPolicy
from sim.shared_skills import build_shared_library
from sim.skill_validation import DEFAULT_CONFIG, V2_CONFIG, compare_defence, observation, run_trial


def paired_fixture():
    baseline = {
        "status": "completed", "failure": None,
        "requested_duration_s": 20., "completed_time_s": 20.,
        "hits": [{"agent": "a1", "attack_id": 1, "time_s": 8., "phase": "thrust", "double_touch": False}],
        "command_sha256": {"a1": "unchanged_attack", "a2": "guard"},
        "initial_qpos": [0.]*14,
        "runtime_config": copy.deepcopy(DEFAULT_CONFIG),
        "scoring_radius_m": .07, "targets_by_attacker": {"a1": [1.43, 0, 1.05], "a2": [.32, 0, 1.05]},
        "unchanged_force_caps_nm": [5, 80, 15, 20, 3, 3, 2]*2,
        "source_hashes": {"robot.xml": "source"}, "geometry_sha256": "same_paths_and_clock",
        "first_positive_blade_contact_time_s": None, "blade_contact_normal_impulse_ns": 0.,
    }
    defended = copy.deepcopy(baseline)
    defended.update(hits=[], first_positive_blade_contact_time_s=6., blade_contact_normal_impulse_ns=.2)
    defended["command_sha256"]["a2"] = "parry"
    return baseline, defended


class PairedEvidenceTests(unittest.TestCase):
    def test_valid_causal_control_and_failed_run_never_becomes_success(self):
        baseline, defended = paired_fixture()
        self.assertTrue(compare_defence(baseline, defended)["successful_physical_defence"])
        for field, value in (("status", "invalid"), ("first_positive_blade_contact_time_s", 9.),
                             ("blade_contact_normal_impulse_ns", .001)):
            bad = copy.deepcopy(defended)
            bad[field] = value
            self.assertFalse(compare_defence(baseline, bad)["successful_physical_defence"], field)
        baseline["hits"] = []
        self.assertFalse(compare_defence(baseline, defended)["successful_physical_defence"])

    def test_attack_hash_initial_state_and_all_conditions_must_match(self):
        baseline, defended = paired_fixture()
        changes = {
            "runtime_config": dict(DEFAULT_CONFIG, physics_timestep_s=.001),
            "scoring_radius_m": .02,
            "targets_by_attacker": {"a1": [1.6, 0, 1.05], "a2": [.32, 0, 1.05]},
            "unchanged_force_caps_nm": [500]*14,
            "requested_duration_s": 8.,
            "source_hashes": {"robot.xml": "different"},
            "geometry_sha256": "different",
            "initial_qpos": [1.]*14,
            "command_sha256": {"a1": "changed_attack", "a2": "parry"},
        }
        for field, value in changes.items():
            bad = copy.deepcopy(defended)
            bad[field] = value
            result = compare_defence(baseline, bad)
            self.assertFalse(result["valid_paired_trial"], field)
            self.assertFalse(result["successful_physical_defence"], field)

    def test_truncated_completed_label_is_not_full_window_evidence(self):
        baseline, defended = paired_fixture()
        defended["completed_time_s"] = 7.
        result = compare_defence(baseline, defended)
        self.assertFalse(result["valid_paired_trial"])
        self.assertFalse(result["successful_physical_defence"])

    def test_late_large_impulse_cannot_turn_early_graze_into_defence(self):
        baseline, defended = paired_fixture()
        defended["blade_contact_impulse_samples"] = [
            {"time_s": 6., "normal_force_n": .1, "interval_s": .002},
            {"time_s": 9., "normal_force_n": 100., "interval_s": .002},
        ]
        result = compare_defence(baseline, defended)
        self.assertAlmostEqual(result["blade_impulse_before_baseline_hit_ns"], .0002)
        self.assertFalse(result["successful_physical_defence"])

    def test_post_contact_shadow_hit_is_failure_even_after_round_end(self):
        baseline, defended = paired_fixture()
        baseline["runtime_config"] = dict(V2_CONFIG)
        defended["runtime_config"] = dict(V2_CONFIG)
        defended["blade_contact_impulse_samples"] = [
            {"time_s": 6., "normal_force_n": 10., "interval_s": .01},
        ]
        self.assertTrue(compare_defence(baseline, defended)["successful_physical_defence"])
        defended["hits"] = [{"agent": "a1", "time_s": 15., "post_round": True}]
        result = compare_defence(baseline, defended)
        self.assertTrue(result["post_contact_target_entry"])
        self.assertFalse(result["successful_physical_defence"])

    def test_v2_missing_early_impulse_evidence_cannot_pass(self):
        baseline, defended = paired_fixture()
        baseline["runtime_config"] = dict(V2_CONFIG)
        defended["runtime_config"] = dict(V2_CONFIG)
        self.assertFalse(compare_defence(baseline, defended)["successful_physical_defence"])


class NativeValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library = build_shared_library()
        cls.scene = build_duel_scene(DEFAULT_CONFIG["base_distance_m"])

    def setUp(self):
        scene = self.scene
        mujoco.mj_resetData(scene.model, scene.data)
        for mapping in scene.agents.values():
            scene.data.qpos[mapping["qpos_ids"]] = self.library.guard_q
        mujoco.mj_forward(scene.model, scene.data)

    def assert_observations_equal(self, first, second):
        self.assertEqual(set(first), set(second))
        for key in first:
            if isinstance(first[key], dict):
                self.assert_observations_equal(first[key], second[key])
            elif isinstance(first[key], np.ndarray):
                np.testing.assert_allclose(first[key], second[key], atol=1e-10)
            else:
                self.assertEqual(first[key], second[key])

    def test_symmetric_current_state_observation_schema_has_no_opponent_clock(self):
        scene = self.scene
        for mapping in scene.agents.values():
            scene.data.qvel[mapping["dof_ids"]] = np.linspace(-.01, .01, 7)
        mujoco.mj_forward(scene.model, scene.data)
        controllers = {k: SkillController(self.library) for k in scene.agents}
        a = observation(scene, "a1", controllers, .7)
        b = observation(scene, "a2", controllers, .7)
        self.assert_observations_equal(a, b)
        self.assertEqual(set(a), {"time_s", "own_skill", "own_q_rad", "own_qvel_rad_s",
            "opponent_tip_minus_own_target_local_m", "opponent_grip_minus_own_target_local_m",
            "opponent_tip_velocity_local_m_s"})
        self.assertEqual(a["own_q_rad"].shape, (7,))
        self.assertEqual(a["opponent_tip_velocity_local_m_s"].shape, (3,))
        # Changing only a hidden opponent controller clock/action must not
        # change this agent's observation when physical state is unchanged.
        controllers["a2"].request(1, .2)
        after = observation(scene, "a1", controllers, .7)
        self.assert_observations_equal(a, after)

    def test_native_smoke_same_guard_no_resets_and_saved_force_matches_ctrl(self):
        scene = self.scene
        native_step = mujoco.mj_step
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"smoke"
            with patch("sim.skill_validation.mujoco.mj_step", wraps=native_step) as step:
                result = run_trial(scene, self.library,
                    {"a1": ScriptedPolicy([(0., 1)]), "a2": ScriptedPolicy()}, .12, output=output)
            self.assertGreaterEqual(step.call_count, 60)
            self.assertEqual(result["status"], "completed")
            self.assertAlmostEqual(result["completed_time_s"], .12, places=9)
            self.assertEqual(result["qpos_resets_after_initialization"], 0)
            for mapping in scene.agents.values():
                np.testing.assert_array_equal(np.asarray(result["initial_qpos"])[mapping["qpos_ids"]], self.library.guard_q)
            self.assertEqual(result["initial_contacts"], [])
            self.assertTrue(result["geometry_sha256"])
            self.assertEqual(result["source_hashes"], scene.source_hashes)
            with np.load(output/"trajectory.npz", allow_pickle=False) as trace:
                probe = mujoco.MjData(scene.model)
                for q, v, ctrl, torque in zip(trace["qpos"], trace["qvel"], trace["ctrl"], trace["actuator_force_nm"]):
                    probe.qpos[:], probe.qvel[:], probe.ctrl[:] = q, v, ctrl
                    mujoco.mj_forward(scene.model, probe)
                    np.testing.assert_allclose(probe.actuator_force, torque, atol=1e-8)
            np.testing.assert_array_equal(scene.data.qfrc_applied, np.zeros(14))
            np.testing.assert_array_equal(scene.model.body_gravcomp, np.zeros(scene.model.nbody))

    def test_failed_geometry_is_rejected_before_a_native_trial(self):
        bad = copy.copy(self.library)
        bad.reports = copy.deepcopy(self.library.reports)
        bad.reports["library"]["all_skills_pass_geometry"] = False
        with self.assertRaisesRegex(ValueError, "geometry preflight"):
            run_trial(self.scene, bad, {"a1": ScriptedPolicy(), "a2": ScriptedPolicy()}, .02)

    def test_v2_native_smoke_has_independent_round_and_shadow_accounting(self):
        result = run_trial(self.scene, self.library,
            {"a1": ScriptedPolicy(), "a2": ScriptedPolicy()}, .02, cfg=V2_CONFIG)
        self.assertEqual(result["status"], "completed")
        self.assertIsNone(result["scoring_failure"])
        self.assertEqual(result["hits"], [])
        self.assertEqual(result["round_first_touch_events"], [])
        self.assertTrue(result["validation_continues_after_round_end"])
        self.assertEqual(set(result["minimum_blade_target_clearance_m"]), {"a1", "a2"})

    def test_v2_native_invalid_contact_cannot_be_a_zero_point_draw(self):
        bad = {"geom1": "a1_shinai_blade_collision_0", "geom2": "floor",
               "category": "illegal_floor_contact", "distance_m": -.001,
               "normal_force_n": 2.}
        with patch("sim.skill_validation.duel_contact_records", return_value=[bad]):
            result = run_trial(self.scene, self.library,
                {"a1": ScriptedPolicy(), "a2": ScriptedPolicy()}, .02, cfg=V2_CONFIG)
        self.assertEqual(result["status"], "invalid")
        self.assertEqual(result["failure"], "illegal_contact_force")
        self.assertEqual(result["round_result"]["status"], "invalid")


if __name__ == "__main__":
    unittest.main()
