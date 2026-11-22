"""Independent observation, paired-evidence, and native runner regressions.

These short tests do not establish autonomous defence, continuous counterplay,
motor capability, or RL readiness. Full opportunity sweeps remain diagnostics.
"""
from __future__ import annotations

import copy
import functools
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from sim.duel_scene import build_duel_scene
from sim.shared_control import SkillController, ScriptedPolicy
from sim.shared_skills import build_shared_library
from sim.skill_validation import (DEFAULT_CONFIG, V2_CONFIG, compare_defence,
                                  freeze_experiment, observation, run_trial)


class FeedbackControllerFixture(SkillController):
    """Native runner adapter fixture, not an A1 fencing-skill implementation."""
    def __init__(self, library):
        super().__init__(library)
        self.failure = None
        self.feedback = []
        self.action_map = {0: "guard", 1: "attack", 2: "parry"}
        self.feedback_gates = {"scope": "current_own_joint_state_only"}

    def observe_own_state(self, q, qvel, now):
        self.feedback.append({"q": q.copy(), "qvel": qvel.copy(), "time_s": now})

    def diagnostics(self):
        return {"action_map": self.action_map, "feedback_gates": self.feedback_gates,
                "feedback_samples": len(self.feedback), "ignored_requests": self.ignored_busy_requests}


class CapturePolicyFixture:
    def __init__(self, action=0):
        self.action = action
        self.observations = []

    def decide(self, observation):
        self.observations.append(copy.deepcopy(observation))
        return self.action


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

    def test_custom_factory_builds_independent_controllers_and_records_evidence(self):
        created = []

        def factory(library):
            controller = FeedbackControllerFixture(library)
            created.append(controller)
            return controller

        policies = {owner: CapturePolicyFixture() for owner in ("a1", "a2")}
        result = run_trial(self.scene, self.library, policies, .02, cfg=V2_CONFIG,
                           controller_factory=factory)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(created), 2)
        self.assertIsNot(created[0], created[1])
        for owner, controller in zip(policies, created):
            self.assertEqual(len(controller.feedback), 11)
            self.assertEqual(result["controller_diagnostics"][owner]["feedback_samples"], 11)
            self.assertEqual(result["controller_diagnostics"][owner]["action_map"], controller.action_map)
            self.assertEqual(result["controller_diagnostics"][owner]["events"], controller.events)
        self.assertEqual(result["controller_failures"], {})
        relative = "tests/test_skill_validation.py"
        expected = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.assertEqual(result["controller_implementation_sha256"][relative], expected)
        # Custom adapter evidence must remain JSON-serializable for frozen runs.
        json.dumps(result["controller_diagnostics"], allow_nan=False)

    def test_feedback_receives_current_own_state_before_policy_without_leaking_opponent_clock(self):
        created = []

        class CheckingController(FeedbackControllerFixture):
            def observe_own_state(controller, q, qvel, now):
                owner = ("a1", "a2")[created.index(controller)]
                mapping = self.scene.agents[owner]
                np.testing.assert_array_equal(q, self.scene.data.qpos[mapping["qpos_ids"]])
                np.testing.assert_array_equal(qvel, self.scene.data.qvel[mapping["dof_ids"]])
                self.assertEqual(now, self.scene.data.time)
                super(CheckingController, controller).observe_own_state(q, qvel, now)
                # In-place edits must not touch native qpos or qvel. Subsequent
                # comparisons and the policy snapshot would expose aliasing.
                q[:] = 123.
                qvel[:] = 456.

        def factory(library):
            controller = CheckingController(library)
            created.append(controller)
            return controller

        class CheckingPolicy(CapturePolicyFixture):
            def decide(policy, obs):
                owner_index = list(policies.values()).index(policy)
                own = created[owner_index]
                self.assertEqual(own.feedback[-1]["time_s"], obs["time_s"])
                np.testing.assert_array_equal(own.feedback[-1]["q"], obs["own_q_rad"])
                np.testing.assert_array_equal(own.feedback[-1]["qvel"], obs["own_qvel_rad_s"])
                self.assertEqual(set(obs), {"time_s", "own_skill", "own_q_rad", "own_qvel_rad_s",
                    "opponent_tip_minus_own_target_local_m", "opponent_grip_minus_own_target_local_m",
                    "opponent_tip_velocity_local_m_s"})
                self.assertNotIn("opponent_skill", obs)
                self.assertNotIn("opponent_action_id", obs)
                return super(CheckingPolicy, policy).decide(obs)

        policies = {"a1": CheckingPolicy(1), "a2": CheckingPolicy()}
        result = run_trial(self.scene, self.library, policies, .12, cfg=V2_CONFIG,
                           controller_factory=factory)
        self.assertEqual(result["status"], "completed")
        self.assertFalse(np.any(self.scene.data.qpos == 123.))
        self.assertFalse(np.any(self.scene.data.qvel == 456.))
        self.assertEqual(len(created[0].feedback), 61)

    def test_feedback_fault_invalidates_before_new_actions_are_accepted(self):
        created = []

        class FaultController(FeedbackControllerFixture):
            def observe_own_state(controller, q, qvel, now):
                super(FaultController, controller).observe_own_state(q, qvel, now)
                controller.failure = "guard_return_timeout"

        def factory(library):
            controller = FaultController(library)
            created.append(controller)
            return controller

        policies = {owner: CapturePolicyFixture(1) for owner in ("a1", "a2")}
        result = run_trial(self.scene, self.library, policies, .02, cfg=V2_CONFIG,
                           controller_factory=factory)
        self.assertEqual(result["status"], "invalid")
        self.assertEqual(result["failure"], "controller_failure")
        self.assertEqual(result["round_result"]["status"], "invalid")
        self.assertEqual(result["controller_failures"], {"a1": "guard_return_timeout", "a2": "guard_return_timeout"})
        self.assertEqual(result["decisions"], [])
        self.assertTrue(all(controller.events == [] for controller in created))
        self.assertTrue(all(policy.observations == [] for policy in policies.values()))

    def test_custom_factory_is_rejected_in_v1_and_cannot_share_clock_instances(self):
        with self.assertRaisesRegex(ValueError, "requires explicit whole_blade_first_touch_v2"):
            run_trial(self.scene, self.library, {"a1": ScriptedPolicy(), "a2": ScriptedPolicy()}, .02,
                      controller_factory=FeedbackControllerFixture)
        shared = FeedbackControllerFixture(self.library)
        with self.assertRaisesRegex(ValueError, "independent instances"):
            run_trial(self.scene, self.library, {"a1": ScriptedPolicy(), "a2": ScriptedPolicy()}, .02,
                      cfg=V2_CONFIG, controller_factory=lambda library: shared)

    def test_partial_custom_factory_retains_real_implementation_hash(self):
        result = run_trial(self.scene, self.library,
            {"a1": ScriptedPolicy(), "a2": ScriptedPolicy()}, .004, cfg=V2_CONFIG,
            controller_factory=functools.partial(FeedbackControllerFixture))
        self.assertEqual(result["status"], "completed")
        self.assertIn("tests/test_skill_validation.py", result["controller_implementation_sha256"])

    def test_freeze_extra_implementation_hashes_add_without_changing_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            first = freeze_experiment(Path(directory)/"default", self.library, self.scene)
            second = freeze_experiment(Path(directory)/"extra", self.library, self.scene,
                                       implementation_paths=("tests/test_skill_validation.py",))
            old = json.loads((first/"experiment.json").read_text())["implementation_sha256"]
            new = json.loads((second/"experiment.json").read_text())["implementation_sha256"]
            self.assertEqual(set(old), {"sim/shared_skills.py", "sim/shared_control.py", "sim/skill_validation.py",
                                        "sim/duel_scene.py", "sim/manual_duel.py", "sim/fencing_rules.py"})
            self.assertEqual({key: new[key] for key in old}, old)
            self.assertEqual(new["tests/test_skill_validation.py"], hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
