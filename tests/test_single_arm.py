"""Numerical and fault-injection checks for the current single-arm milestone."""

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

from sim.audit import audit_model
from sim.demo import SimulationFailure, run_demo
from sim.model import duration, load_config, load_debug_model


class SingleArmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()

    def test_source_model_is_not_mistaken_for_a_calibrated_fencing_environment(self):
        report = audit_model()
        self.assertTrue(report["fixed_base"])
        self.assertEqual(report["nq"], 7)
        self.assertEqual(report["nu"], 7)
        self.assertEqual(report["robot_collision_geom_count"], 0)
        self.assertEqual(report["sites"], [])
        self.assertTrue(all(body["positive_inertia"] and body["inertia_triangle_valid"] for body in report["bodies"]))
        self.assertTrue(all(joint["range_rad"] is None for joint in report["joints"]))
        self.assertTrue(all(actuator["force_range"] is None for actuator in report["actuators"]))

    def test_default_full_motion_under_gravity(self):
        robot = load_debug_model(copy.deepcopy(self.config))
        with tempfile.TemporaryDirectory() as directory:
            summary = run_demo(robot, directory)
            self.assertEqual(summary["status"], "passed")
            self.assertAlmostEqual(summary["duration_s"], duration(self.config), places=7)
            self.assertFalse(any(summary["warnings"]))
            self.assertEqual(summary["sample_count"], 3001)
            for result in summary["joints"].values():
                self.assertLess(result["tracking_rmse_rad"], 0.005)
                self.assertLess(result["peak_tracking_error_rad"], 0.01)
                self.assertLess(result["peak_speed_rad_s"], 0.05)
                self.assertLessEqual(result["peak_actuator_torque_nm"], result["effective_debug_torque_cap_nm"] + 1e-7)
            with (Path(directory) / "trajectory.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), summary["sample_count"])
            for name in robot.names:
                targets = np.asarray([float(row[f"{name}_target_rad"]) for row in rows])
                self.assertAlmostEqual(float(targets.max()), self.config["trajectory"]["amplitude_rad"], places=7)
                self.assertAlmostEqual(float(targets[0]), 0)
                self.assertAlmostEqual(float(targets[-1]), 0)
                self.assertLessEqual(float(np.max(np.abs(np.diff(targets)))) / self.config["control_period_s"],
                                     self.config["joints"][name]["max_target_speed_rad_s"] + 1e-10)

    def test_low_torque_fails_instead_of_bypassing_the_actuator_cap(self):
        config = copy.deepcopy(self.config)
        config["joints"]["J1"]["debug_torque_cap_nm"] = 0.1
        robot = load_debug_model(config)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(SimulationFailure):
                run_demo(robot, directory)
            report = json.loads((Path(directory) / "summary.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertLess(report["duration_s"], duration(config))
        self.assertTrue(any(result["peak_speed_rad_s"] > config["max_observed_speed_rad_s"]
                            or result["peak_tracking_error_rad"] > config["max_tracking_error_rad"]
                            for result in report["joints"].values()))
        self.assertLessEqual(np.abs(robot.data.qfrc_actuator[robot.dof_ids[1]]), 0.1 + 1e-7)
        np.testing.assert_array_equal(robot.data.qfrc_applied, 0)
        np.testing.assert_array_equal(robot.data.xfrc_applied, 0)
        np.testing.assert_array_equal(robot.model.body_gravcomp, 0)

    def test_unexpected_step_error_replaces_a_previous_passed_summary(self):
        robot = load_debug_model(copy.deepcopy(self.config))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text('{"status": "passed"}')
            with patch("sim.demo.mujoco.mj_step", side_effect=RuntimeError("injected physics failure")):
                with self.assertRaises(SimulationFailure):
                    run_demo(robot, directory)
            report = json.loads(path.read_text())
        self.assertEqual(report["status"], "failed")
        self.assertIn("injected physics failure", report["failure"])

    def test_command_slew_limits_and_does_not_teleport_the_arm(self):
        robot = load_debug_model(copy.deepcopy(self.config))
        initial_q = robot.data.qpos.copy()
        applied = robot.command(robot.home + 0.05)
        np.testing.assert_array_equal(robot.data.qpos, initial_q)
        self.assertTrue(np.all(applied - robot.home <= robot.target_speed_caps * self.config["control_period_s"] + 1e-12))
        again = robot.command(robot.home + 0.05)
        np.testing.assert_array_equal(again, applied)
        mujoco.mj_step(robot.model, robot.data)
        self.assertGreater(robot.data.time, 0)

    def test_invalid_windows_and_rate_settings_are_rejected(self):
        for mutation in ("amplitude", "period", "speed", "nan"):
            with self.subTest(mutation=mutation):
                config = copy.deepcopy(self.config)
                if mutation == "amplitude":
                    config["trajectory"]["amplitude_rad"] = 0.5
                elif mutation == "period":
                    config["control_period_s"] = 0.003
                elif mutation == "speed":
                    config["joints"]["J0"]["max_target_speed_rad_s"] = 0.0001
                else:
                    config["joints"]["J0"]["debug_torque_cap_nm"] = float("nan")
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "invalid.json"
                    path.write_text(json.dumps(config))
                    with self.assertRaises(ValueError):
                        load_config(path)

    def test_name_mapping_does_not_assume_configuration_order(self):
        config = copy.deepcopy(self.config)
        config["joints"] = dict(reversed(list(config["joints"].items())))
        robot = load_debug_model(config)
        robot.command(robot.home)
        mujoco.mj_forward(robot.model, robot.data)
        for index, name in enumerate(robot.names):
            self.assertEqual(mujoco.mj_id2name(robot.model, mujoco.mjtObj.mjOBJ_JOINT, robot.joint_ids[index]), name)
            self.assertEqual(mujoco.mj_id2name(robot.model, mujoco.mjtObj.mjOBJ_ACTUATOR, robot.actuator_ids[index]), name)
        # The reordered controller still supplies the gravity load to J1, not J5.
        np.testing.assert_allclose(robot.data.qfrc_actuator, robot.gravity_data.qfrc_bias, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
