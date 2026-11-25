"""J3 straight-arm offset is kinematic provenance, not motion certification."""
from __future__ import annotations

import hashlib
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from sim.j3_zero_calibration import ROOT, calibration_report, measure_j3_zero


class J3CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_path(str(ROOT / "scene.xml"))

    def test_measured_source_offset_and_straight_axes(self):
        result = measure_j3_zero(self.model)
        self.assertTrue(result["geometry_consistent"])
        self.assertAlmostEqual(result["offset_deg"], 12.491944940769413, places=11)
        self.assertLess(result["offset_spread_deg"], 1e-10)
        self.assertLess(result["maximum_straight_axis_alignment_residual_deg"], 1e-10)

    def test_does_not_depend_on_joint_limits_or_integrate(self):
        before = self.model.jnt_range.copy()
        limited = self.model.jnt_limited.copy()
        try:
            self.model.jnt_range[:] = [-.001, .001]
            self.model.jnt_limited[:] = True
            with patch("mujoco.mj_step", side_effect=AssertionError("not a physics experiment")):
                result = measure_j3_zero(self.model)
            self.assertTrue(result["geometry_consistent"])
            np.testing.assert_array_equal(self.model.jnt_range, np.tile([-.001, .001], (7, 1)))
        finally:
            self.model.jnt_range[:] = before
            self.model.jnt_limited[:] = limited

    def test_probe_wrap_is_not_model_coordinate_wrapping(self):
        probes = np.zeros((2, 7))
        probes[:, 3] = [0., 2 * np.pi + .2]
        result = measure_j3_zero(self.model, qposes=probes)
        self.assertTrue(result["geometry_consistent"])
        self.assertEqual(result["samples"][1]["probe_qpos_rad_J0_to_J6"][3], probes[1, 3])

    def test_bad_inputs_rejected(self):
        for values in ([], np.zeros(7), np.zeros((1, 6)), [[float("nan")] * 7]):
            with self.assertRaises(ValueError):
                measure_j3_zero(self.model, qposes=values)
        with self.assertRaises(ValueError):
            measure_j3_zero(self.model, prefix="missing_")

    def test_report_both_slots_source_hashes_and_no_shared_constraints(self):
        with patch("mujoco.mj_step", side_effect=AssertionError("no integration")), \
             patch("sim.joint_constraints.load_joint_constraints", side_effect=AssertionError("no mapping read")):
            report = calibration_report()
        self.assertTrue(report["geometry_consistent"])
        self.assertEqual(set(report["measurements"]), {"floor", "a1", "a2"})
        self.assertLess(report["cross_model_offset_spread_deg"], 1e-10)
        self.assertEqual(report["physics_integration_steps"], 0)
        self.assertFalse(report["shared_joint_constraints_read"])
        self.assertFalse(report["limits_or_model_parameters_modified"])
        self.assertFalse(report["admitted_to_training"])
        np.testing.assert_allclose(report["derived_CAD_bounds_deg"],
                                   [-32.49194494076941, 107.50805505923059], atol=1e-11, rtol=0.)
        self.assertIn("not_hardware_calibration", report["status"])
        for path, digest in report["source_sha256"].items():
            self.assertEqual(digest, hashlib.sha256((ROOT / path).read_bytes()).hexdigest())

    def test_current_mapping_agrees_with_measured_source_geometry(self):
        import json
        from sim.joint_constraints import DEFAULT_PATH, load_joint_constraints
        document = json.loads(DEFAULT_PATH.read_text())
        self.assertAlmostEqual(load_joint_constraints().offset_deg[3],
                               measure_j3_zero(self.model)['offset_deg'], places=10)
        for path, digest in document['j3_zero_definition']['cad_source_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
