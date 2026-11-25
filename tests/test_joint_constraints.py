from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from sim.joint_constraints import (DEFAULT_PATH, audit_joint_trajectory,
    enforce_joint_trajectory, load_joint_constraints, model_joint_ranges_rad,
    hardware_to_model_rad, model_to_hardware_deg)


class JointConstraintsTests(unittest.TestCase):
    def setUp(self):
        self.constraints = load_joint_constraints()
        self.middle = np.mean(self.constraints.bounds_deg, axis=1)

    def test_observed_table_and_user_confirmed_mapping(self):
        self.assertEqual(self.constraints.bounds_deg,
                         ((0, 360), (0, 180), (0, 360), (-20, 120),
                          (0, 360), (15, 145), (-90, 90)))
        self.assertTrue(self.constraints.confirmed)
        self.assertEqual(self.constraints.sign, (1,) * 7)
        np.testing.assert_allclose(self.constraints.offset_deg,
            [0, 0, 0, 12.491944940769413, 0, 0, 0], rtol=0, atol=1e-12)

    def test_inclusive_limits_and_every_sample(self):
        values = np.array(self.constraints.bounds_deg).T
        result = enforce_joint_trajectory(hardware_to_model_rad(values, self.constraints), self.constraints)
        self.assertTrue(result["within_limits"])
        self.assertEqual(result["sample_count"], 2)
        np.testing.assert_allclose(result["values_deg"], values, atol=1e-12)
        np.testing.assert_allclose(result["raw_min_deg"], values[0], atol=1e-12)

    def test_reports_all_violations_without_clipping(self):
        values = np.tile(self.middle, (3, 1))
        values[0, 1], values[2, 5] = -1, 146
        q = hardware_to_model_rad(values, self.constraints)
        before = q.copy()
        result = audit_joint_trajectory(q, self.constraints)
        self.assertEqual([(x["sample_index"], x["joint"]) for x in result["violations"]],
                         [(0, "J1"), (2, "J5")])
        np.testing.assert_array_equal(q, before)
        with self.assertRaises(ValueError):
            enforce_joint_trajectory(q, self.constraints)

    def test_sign_and_fixed_offset_are_applied_once_to_whole_path(self):
        changed = replace(self.constraints, sign=(-1,) * 7, offset_deg=(40,) * 7)
        values = np.tile(self.middle, (2, 1))
        result = enforce_joint_trajectory(np.deg2rad(40 - values), changed)
        np.testing.assert_allclose(result["values_deg"], values)
        ranges = model_joint_ranges_rad(changed)
        np.testing.assert_allclose(np.rad2deg(ranges),
                                   np.sort(40 - np.asarray(changed.bounds_deg), axis=1))
        self.assertEqual(ranges.shape, (7, 2))

    def test_unconfirmed_rejects_enforcement_but_allows_labeled_diagnostic(self):
        unknown = replace(self.constraints, confirmed=False, sign=None, offset_deg=None)
        with self.assertRaises(ValueError):
            enforce_joint_trajectory(np.deg2rad(self.middle), unknown)
        with self.assertRaises(ValueError):
            model_joint_ranges_rad(unknown)
        result = audit_joint_trajectory(np.deg2rad(self.middle), unknown,
                                        diagnostic_identity=True)
        self.assertTrue(result["within_limits"])
        self.assertTrue(result["diagnostic_identity_assumption"])
        self.assertFalse(result["eligible_for_enforcement"])

    def test_multi_turn_jump_is_not_hidden_by_per_sample_modulo(self):
        values = np.tile(self.middle, (2, 1))
        values[1, 0] += 360
        result = audit_joint_trajectory(hardware_to_model_rad(values, self.constraints), self.constraints)
        self.assertEqual(result["violations"][0]["joint"], "J0")
        self.assertAlmostEqual(result["raw_max_deg"][0], 540)
        self.assertAlmostEqual(result["violations"][0]["excess_deg"], 180)

    def test_bad_bounds_mapping_and_nonfinite_inputs_rejected(self):
        for changes in ({"bounds_deg": [[1, 0]] * 7}, {"bounds_deg": [[0, float("nan")]] * 7},
                        {"sign": [0] * 7}, {"sign": [True] * 7}, {"offset_deg": [0] * 6},
                        {"confirmed": 1}, {"sign": None}, {"confirmation_source": ""}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.constraints, **changes)
        for value in ([0] * 6, [], [float("nan")] * 7, [float("inf")] * 7,
                      [True] * 7, ["1"] * 7, [1e308] * 7):
            with self.subTest(value=value), self.assertRaises(ValueError):
                audit_joint_trajectory(value, self.constraints)

    def test_loader_rejects_reordered_joint_names_and_wrong_schema(self):
        document = json.loads(DEFAULT_PATH.read_text())
        for changes in ({"joint_names": list(reversed(document["joint_names"]))},
                        {"schema_version": True}, {"mapping": None}):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "constraints.json"
                path.write_text(json.dumps(dict(document, **changes)))
                with self.assertRaises(ValueError):
                    load_joint_constraints(path)

    def test_straight_j3_zero_and_corrected_bounds_roundtrip_without_clipping(self):
        hardware = np.tile(self.middle, (3, 1))
        hardware[:, 3] = [-20., 0., 120.]
        q = hardware_to_model_rad(hardware, self.constraints)
        np.testing.assert_allclose(np.rad2deg(q[:, 3]),
            [-32.49194494076941, -12.491944940769413, 107.50805505923059], atol=1e-12)
        np.testing.assert_allclose(model_to_hardware_deg(q, self.constraints), hardware, atol=1e-12)
        np.testing.assert_allclose(q[[0, 2], 3], model_joint_ranges_rad(self.constraints)[3], atol=1e-15)
        self.assertTrue(enforce_joint_trajectory(q, self.constraints)['within_limits'])
        old_high_pose = q[1].copy(); old_high_pose[3] = np.deg2rad(115.)
        with self.assertRaises(ValueError): enforce_joint_trajectory(old_high_pose, self.constraints)

    def test_coordinate_conversion_validates_mapping_shape_and_finiteness(self):
        unknown = replace(self.constraints, confirmed=False, sign=None, offset_deg=None)
        for convert in (hardware_to_model_rad, model_to_hardware_deg):
            with self.assertRaises(ValueError): convert(self.middle, unknown)
            for value in ([], [1]*6, [True]*7, [np.nan]*7, [np.inf]*7, ['1']*7):
                with self.subTest(function=convert.__name__, value=value), self.assertRaises(ValueError):
                    convert(value, self.constraints)

    def test_historical_zero_map_remains_explicitly_available_not_current(self):
        old = load_joint_constraints(DEFAULT_PATH.with_name('joint_constraints_cad_zero_v1_historical.json'))
        self.assertEqual(old.offset_deg, (0.,)*7)
        self.assertNotEqual(old.offset_deg, self.constraints.offset_deg)


if __name__ == "__main__":
    unittest.main()
