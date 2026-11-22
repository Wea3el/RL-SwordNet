"""A1 verification must count actual accepted AND completed executions."""
from __future__ import annotations

import copy
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sim.a1_validation import defence_gate, execution_gate, main, validate_a1


def fixture():
    return {"status": "completed", "requested_duration_s": 10., "completed_time_s": 10.,
        "runtime_config": {"physics_timestep_s": .002},
        "action_events": {"a1": [{"action_id": 1}, {"action_id": 3}], "a2": []},
        "controller_diagnostics": {"a1": {"completions": [{"action_id": 1}, {"action_id": 3}]},
                                   "a2": {"completions": []}}}


class ExecutionGateTests(unittest.TestCase):
    def test_complete_native_sequence_is_only_execution_evidence(self):
        result = execution_gate(fixture(), {"a1": [1, 3], "a2": []})
        self.assertTrue(result["native_execution_pass"])
        self.assertFalse(result["effective_attack_or_defence_certification"])

    def test_missing_or_wrong_order_acceptance_or_completion_cannot_pass(self):
        original = fixture()
        variants = []
        for field in ("action_events", "controller_diagnostics"):
            bad = copy.deepcopy(original)
            if field == "action_events":
                bad[field]["a1"] = [{"action_id": 3}, {"action_id": 1}]
            else:
                bad[field]["a1"]["completions"] = [{"action_id": 1}]
            variants.append(bad)
        bad = copy.deepcopy(original)
        bad["action_events"]["a2"] = [{"action_id": 4}]
        variants.append(bad)
        for bad in variants:
            self.assertFalse(execution_gate(bad, {"a1": [1, 3], "a2": []})["native_execution_pass"])

    def test_truncation_or_native_invalid_overrides_good_action_counts(self):
        for change in ({"completed_time_s": 9.}, {"status": "invalid"}):
            bad = dict(fixture(), **change)
            self.assertFalse(execution_gate(bad, {"a1": [1, 3], "a2": []})["native_execution_pass"])

    def test_guard_control_requires_no_unexpected_skill_execution(self):
        r = fixture()
        self.assertFalse(execution_gate(r, {"a1": [], "a2": []})["native_execution_pass"])

    def test_physical_comparison_without_complete_expected_skills_is_not_defence(self):
        physical = {"successful_physical_defence": True}
        base = {"a1_execution_gate": {"native_execution_pass": True}}
        for tested in ({}, {"a1_execution_gate": {"native_execution_pass": False}}):
            with patch("sim.a1_validation.compare_defence", return_value=physical.copy()):
                self.assertFalse(defence_gate(base, tested, "a1")["successful_physical_defence"])
        with patch("sim.a1_validation.compare_defence", return_value=physical.copy()):
            self.assertTrue(defence_gate(base, base, "a1")["successful_physical_defence"])


class CommandLineTests(unittest.TestCase):
    def test_authored_bank_freezes_its_generator_without_claiming_native_pass(self):
        for provenance in ({}, {"adaptation_recipe": {"enabled": True}},
                {"authored_defence_recipes": {"parry_right": {}}},
                {"adaptation_recipe": {"enabled": True}, "authored_defence_recipes": {"parry_left": {}}}):
            library = SimpleNamespace(provenance=provenance, config={},
                reports={"library": {"all_skills_pass_geometry": False}})
            with self.subTest(provenance=provenance), \
                 patch("sim.a1_validation.build_duel_scene", return_value="scene"), \
                 patch("sim.a1_validation.freeze_experiment", return_value=Path("frozen")) as freeze, \
                 patch("pathlib.Path.read_text", return_value='{"implementation_sha256": {}}'), \
                 patch("sim.a1_validation.save_json"), \
                 patch("sim.a1_skills.export_a1_library"):
                result = validate_a1(library, "fresh", progress=None)
            implementations = freeze.call_args.kwargs["implementation_paths"]
            self.assertEqual("sim/a1_adaptation.py" in implementations, bool(provenance.get("adaptation_recipe")))
            self.assertEqual("sim/a1_defence.py" in implementations, bool(provenance.get("authored_defence_recipes")))
            self.assertEqual(result["status"], "geometry_rejected_before_native")
            self.assertFalse(result["may_start_RL"])

    def test_frozen_bank_load_does_not_rebuild_or_retime_pilot(self):
        with patch("sys.argv", ["a1_validation", "--bank", "frozen", "--output", "fresh"]), \
             patch("sim.a1_skills.load_a1_library", return_value="frozen_library") as load, \
             patch("sim.a1_skills.build_a1_library") as build, \
             patch("sim.a1_validation.validate_a1", return_value={"status": "test_only"}) as validate, \
             contextlib.redirect_stdout(io.StringIO()):
            main()
        self.assertEqual(str(load.call_args.args[0]), "frozen")
        build.assert_not_called()
        self.assertEqual(validate.call_args.args[0], "frozen_library")

    def test_frozen_bank_cannot_silently_override_clock_or_mix_input_types(self):
        cases = (["--bank", "frozen", "--source-time-scale", "4"],
                 ["--bank", "frozen", "--bank-config", "config.json"],
                 ["--bank", "frozen", "--pilot", "pilot"])
        for case in cases:
            with self.subTest(arguments=case), \
                 patch("sys.argv", ["a1_validation", "--output", "fresh", *case]), \
                 patch("sim.a1_skills.load_a1_library") as load, \
                 contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main()
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
