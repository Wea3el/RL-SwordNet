import unittest

import numpy as np

from sim.a1_triangle_parry_video import entry_counts, nearest_indices, paired_schedule, status_lines


class PairedNativeVideoTests(unittest.TestCase):
    def test_nearest_samples_are_actual_and_ties_choose_earlier(self):
        self.assertEqual(nearest_indices([0., .1, .2], [0., .05, .16, .2]).tolist(), [0, 0, 2, 2])
        with self.assertRaises(ValueError):
            nearest_indices([0., .1], [.11])

    def test_full_window_uses_both_actual_streams_no_interpolation(self):
        schedule = paired_schedule([0., .04, .12, .2], [0., .08, .16, .2], .2, fps=10)
        np.testing.assert_allclose(schedule["requested_time_s"], [0., .1])
        self.assertEqual(schedule["baseline_indices"].tolist(), [0, 2])
        self.assertEqual(schedule["defended_indices"].tolist(), [0, 1])
        self.assertFalse(schedule["truncated"])

    def test_failed_short_trace_caps_both_sides_and_never_freezes(self):
        schedule = paired_schedule([0., .1, .2, .3], [0., .1, .15], .3, fps=10)
        self.assertTrue(schedule["truncated"])
        self.assertEqual(schedule["common_end_s"], .15)
        np.testing.assert_allclose(schedule["requested_time_s"], [0., .1])
        self.assertLessEqual(max(schedule["defended_indices"]), 2)

    def test_rejects_invalid_timing_and_nonoverlap(self):
        for times in ([0., 0.], [0., float("nan")], [1., 0.]):
            with self.assertRaises(ValueError):
                paired_schedule(times, [0., 1.], 1.)
        with self.assertRaises(ValueError):
            paired_schedule([0., .1], [.2, .3], .3)
        with self.assertRaises(ValueError):
            paired_schedule([0., 1.], [0., 1.], 1., fps=True)

    def test_entries_use_both_owners_all_regions_and_keep_late_leak(self):
        audit = {"all_entries": [
            {"agent": "a1", "region": "upper", "time_s": .2},
            {"agent": "a2", "region": "left", "time_s": .3},
            {"agent": "a1", "region": "right", "time_s": .8},
            {"agent": "a1", "region": "left", "time_s": .4, "step_invalid": True}]}
        early = entry_counts(audit, .3)
        self.assertEqual(early["a1"], {"left": 0, "right": 0, "upper": 1})
        self.assertEqual(early["a2"]["left"], 1)
        self.assertEqual(entry_counts(audit, 1.)["a1"]["right"], 1)
        with self.assertRaises(ValueError):
            entry_counts({"all_entries": [{"agent": "a1", "region": "old_target", "time_s": 0.}]}, 1.)

    def test_status_does_not_call_failed_or_partial_trace_qualified(self):
        clean = {name: {"summary": {"status": "completed", "native_execution_pass": True}}
                 for name in ("baseline", "defended")}
        title, detail = status_lines({"status": "qualified"}, clean, {"truncated": False})
        self.assertIn("qualified", title)
        self.assertIn("LIMITED CONDITION", detail)
        with self.assertRaises(ValueError):
            status_lines({"status": "qualified"}, clean, {"truncated": True})
        clean["defended"]["summary"] = {"status": "failed", "native_execution_pass": False}
        title, detail = status_lines({"status": "invalid"}, clean, {"truncated": True})
        self.assertIn("TRUNCATED EVIDENCE", title)
        self.assertIn("SOURCE FAILURE: defended", detail)
        self.assertIn("NOT CERTIFIED", detail)

    def test_observation_cue_is_labelled_separately_from_open_loop_or_rl(self):
        clean={name:{'summary':{'status':'completed','native_execution_pass':True}}
               for name in ('baseline','defended')}
        _,detail=status_lines({'status':'qualified','physical_trigger_events':[]},clean,{'truncated':False})
        self.assertIn('CURRENT-OBSERVATION',detail)
        self.assertIn('NOT learned policy',detail)
        self.assertNotIn('open-loop',detail)


if __name__ == "__main__":
    unittest.main()
