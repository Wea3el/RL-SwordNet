"""Fast retiming/controller checks; not long skill or hardware tests."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np

from sim.floor import build_floor_assembly
from sim.floor_playback import (CONTROL_DT, apply_position_gravity_target,
    load_sequence, quintic_position, run_playback, segment_duration)


class FloorPlaybackTests(unittest.TestCase):
    def test_quintic_duration_bounds_and_endpoints(self):
        for delta in (0., .001, .2, 2.):
            duration = segment_duration([delta])
            self.assertGreaterEqual(duration, .02)
            self.assertLessEqual(1.875*delta/duration, .05+1e-12)
            self.assertLessEqual(5.774*delta/duration**2, .1+1e-12)
            self.assertAlmostEqual(duration/CONTROL_DT, round(duration/CONTROL_DT))
        np.testing.assert_array_equal(quintic_position([1, 2], [3, 4], 0), [1, 2])
        np.testing.assert_array_equal(quintic_position([1, 2], [3, 4], 1), [3, 4])
        with self.assertRaises(ValueError):
            segment_duration([np.nan])

    def test_native_actuator_feedforward_uses_same_cap_and_no_applied_force(self):
        a = build_floor_assembly(np.tile([-np.pi, np.pi], (7, 1)))
        m, d = a.model, a.data
        gravity = mujoco.MjData(m)
        original_q = d.qpos.copy()
        apply_position_gravity_target(m, d, gravity, np.ones(7)*2)
        mujoco.mj_forward(m, d)
        np.testing.assert_array_equal(d.qpos, original_q)
        np.testing.assert_array_equal(d.qfrc_applied, np.zeros(7))
        np.testing.assert_array_equal(m.body_gravcomp, np.zeros(m.nbody))
        self.assertTrue(np.all(np.abs(d.qfrc_actuator) <= m.actuator_forcerange[:, 1]+1e-9))

    def test_bad_input_records_failed_and_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"result"
            result = run_playback(Path(directory)/"missing", "stab", output)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(json.loads((output/"summary.json").read_text())["status"], "failed")
            with self.assertRaises(FileExistsError):
                run_playback(directory, "stab", output)

    def test_saved_path_seams_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            q = np.zeros((2, 7))
            for name in ("guard", "guard_to_stab", "stab", "stab_to_guard"):
                np.savez(path/f"{name}.npz", qpos=q)
            sequence = load_sequence(path, "stab")
            self.assertAlmostEqual(sequence["duration_s"], 2.06)
            q[0, 0] = .1
            np.savez(path/"stab.npz", qpos=q)
            with self.assertRaisesRegex(ValueError, "Discontinuous"):
                load_sequence(path, "stab")


if __name__ == "__main__":
    unittest.main()
