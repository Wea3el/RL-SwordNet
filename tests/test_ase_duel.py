"""Full-clip clocks/programs: all tests are numerical, no physics rollouts."""
from __future__ import annotations

import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from sim.ase_duel import AseClipProgram, evaluate_ase_pair, join_role, retime_samples, runtime_config, sample_timed
from sim.ase_skills import guard_transition


def fixture_library():
    attack = np.array([[.1, .2], [.8, -.1], [.3, .7], [.12, .22]])
    parry = np.array([[-.11, .04], [-.6, .5], [-.12, .02]])
    ag, dg = np.array([0., 0.]), np.array([-.1, 0.])
    return SimpleNamespace(
        guard_attack_q=ag, guard_defend_q=dg, attack_q=attack, parry_q=parry,
        attack_phase=np.array([0., .2, .7, 1.]), parry_phase=np.array([0., .4, 1.]),
        transitions={"attack_in": guard_transition(ag, attack[0], 3),
                     "attack_out": guard_transition(attack[-1], ag, 3),
                     "parry_in": guard_transition(dg, parry[0], 3),
                     "parry_out": guard_transition(parry[-1], dg, 3)})


class FullClipClockTests(unittest.TestCase):
    def setUp(self):
        self.library = fixture_library()

    def test_each_native_row_appears_once_between_exact_transition_seams(self):
        for role, prefix, native, native_phase in (
                ("attack", "attack", self.library.attack_q, self.library.attack_phase),
                ("defend", "parry", self.library.parry_q, self.library.parry_phase)):
            with self.subTest(role=role):
                path = join_role(self.library, role, .35, .01)
                incoming = self.library.transitions[prefix+"_in"]["qpos"]
                outgoing = self.library.transitions[prefix+"_out"]["qpos"]
                expected = np.vstack([incoming, native[1:], outgoing[1:]])
                np.testing.assert_array_equal(path["qpos"], expected)
                self.assertEqual(len(path["qpos"]), len(incoming)+len(native)+len(outgoing)-2)
                indices = path["native_row_indices"]
                np.testing.assert_array_equal(indices, np.arange(len(incoming)-1, len(incoming)+len(native)-1))
                np.testing.assert_array_equal(path["qpos"][indices], native)
                np.testing.assert_array_equal(path["native_phase"], native_phase)
                for row in native:
                    self.assertEqual(np.all(path["qpos"] == row, axis=1).sum(), 1)
                self.assertEqual(path["native_start_s"], path["time_s"][indices[0]])
                self.assertEqual(path["native_end_s"], path["time_s"][indices[-1]])

    def test_control_aligned_times_preserve_duplicate_rows_and_speed_bound(self):
        q = np.array([[0., 0.], [0., 0.], [.001, -.7], [.8, -.7]])
        time = retime_samples(q, .35, .01)
        self.assertEqual(len(time), len(q))
        self.assertTrue(np.all(np.diff(time) >= .01-1e-12))
        np.testing.assert_allclose(time/.01, np.round(time/.01), atol=1e-10, rtol=0.)
        speed = np.max(np.abs(np.diff(q, axis=0)/np.diff(time)[:, None]))
        self.assertLessEqual(speed, .35)

    def test_discontinuous_transition_is_rejected(self):
        for transition, at in (("attack_in", -1), ("attack_out", 0)):
            broken = copy.deepcopy(self.library)
            broken.transitions[transition]["qpos"][at] += .01
            with self.subTest(transition=transition), self.assertRaisesRegex(ValueError, "continuously"):
                join_role(broken, "attack", .35, .01)

    def test_joining_does_not_modify_original_geometry_or_phase(self):
        before = copy.deepcopy(self.library)
        path = join_role(self.library, "attack", .35, .01)
        path["qpos"][0] += 100.
        path["native_phase"][0] = .5
        for field in ("attack_q", "parry_q", "attack_phase", "parry_phase", "guard_attack_q", "guard_defend_q"):
            np.testing.assert_array_equal(getattr(self.library, field), getattr(before, field))
        for key in self.library.transitions:
            np.testing.assert_array_equal(self.library.transitions[key]["qpos"], before.transitions[key]["qpos"])

    def test_invalid_clock_inputs_rejected(self):
        for speed, period in ((0., .01), (-1., .01), (np.nan, .01), (.35, 0.), (.35, np.inf)):
            with self.subTest(speed=speed, period=period), self.assertRaises(ValueError):
                retime_samples(self.library.attack_q, speed, period)
        with self.assertRaises(ValueError):
            retime_samples([[0., 0.]], .35, .01)


class FullClipProgramTests(unittest.TestCase):
    def setUp(self):
        self.library = fixture_library()
        self.cfg = runtime_config()

    def test_every_native_sample_is_reachable_at_its_control_clock(self):
        program = AseClipProgram()
        timing = program.schedule(self.library, self.cfg, 1., .2)
        for at, q in zip(program.attack_path["time_s"], program.attack_path["qpos"]):
            np.testing.assert_allclose(program.attack_command(self.library, timing, timing["attack_start_s"]+at), q, atol=1e-12)
        for at, q in zip(program.parry_path["time_s"], program.parry_path["qpos"]):
            np.testing.assert_allclose(program.parry_command(self.library, timing, timing["parry_start_s"]+at, True), q, atol=1e-12)

    def test_commands_return_to_guard_without_manual_full_clip_reverse(self):
        program = AseClipProgram()
        timing = program.schedule(self.library, self.cfg, 1., .2)
        with patch("sim.manual_duel.attack_command", side_effect=AssertionError("No manual reverse protocol")), patch("sim.manual_duel.parry_command", side_effect=AssertionError("No manual reverse protocol")):
            for time in (-1., 0., timing["duration_s"], timing["duration_s"]+1.):
                np.testing.assert_array_equal(program.attack_command(self.library, timing, time), self.library.guard_attack_q)
                np.testing.assert_array_equal(program.parry_command(self.library, timing, time, True), self.library.guard_defend_q)
            at = timing["attack_start_s"]+program.attack_path["native_end_s"]
            np.testing.assert_allclose(program.attack_command(self.library, timing, at), self.library.attack_q[-1])

    def test_attack_targets_identical_for_enabled_and_disabled_defence(self):
        baseline, defended = AseClipProgram(), AseClipProgram()
        tb = baseline.schedule(self.library, self.cfg, 1., .2)
        td = defended.schedule(self.library, self.cfg, 1., .2)
        self.assertEqual(tb, td)
        for time in np.linspace(0., tb["duration_s"], 50):
            baseline.parry_command(self.library, tb, time, False)
            defended.parry_command(self.library, td, time, True)
            np.testing.assert_array_equal(baseline.attack_command(self.library, tb, time), defended.attack_command(self.library, td, time))

    def test_speed_scale_is_capped_and_invalid_speed_delay_are_rejected(self):
        program = AseClipProgram()
        timing = program.schedule(self.library, self.cfg, 1e6, 0.)
        self.assertEqual(timing["attack_target_speed_bound_rad_s"], self.cfg["command_speed_ceiling_rad_s"])
        for path, bound in ((program.attack_path, timing["attack_target_speed_bound_rad_s"]),
                            (program.parry_path, timing["parry_target_speed_bound_rad_s"])):
            peak = np.max(np.abs(np.diff(path["qpos"], axis=0)/np.diff(path["time_s"])[:, None]))
            self.assertLessEqual(peak, bound)
        for speed, delay in ((0., 0.), (-1., 0.), (np.inf, 0.), (1., np.nan), (1., -100.)):
            with self.subTest(speed=speed, delay=delay), self.assertRaises(ValueError):
                program.schedule(self.library, self.cfg, speed, delay)

    def test_interpolation_is_per_joint_and_clamps_endpoints(self):
        path = {"time_s": np.array([0., 2.]), "qpos": np.array([[0., 1.], [4., -1.]])}
        np.testing.assert_allclose(sample_timed(path, .5), [1., .5])
        np.testing.assert_array_equal(sample_timed(path, -1.), path["qpos"][0])
        np.testing.assert_array_equal(sample_timed(path, 3.), path["qpos"][-1])


class NativeSourceAttributionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = runtime_config()

    def pair(self):
        baseline = {"status": "completed", "attack_target_command_sha256": "same",
                    "initial_qpos": [0., 1.], "target_hit": True,
                    "first_target_hit_time_s": 3., "first_target_hit_during_native_attack": True,
                    "first_positive_blade_contact_time_s": None,
                    "first_native_clip_blade_contact_time_s": None,
                    "blade_contact_normal_impulse_ns": 0., "native_clip_overlap_blade_impulse_ns": 0.}
        defended = copy.deepcopy(baseline)
        defended.update(target_hit=False, first_target_hit_time_s=None,
                        first_target_hit_during_native_attack=False,
                        first_positive_blade_contact_time_s=2.,
                        first_native_clip_blade_contact_time_s=2.,
                        blade_contact_normal_impulse_ns=.2, native_clip_overlap_blade_impulse_ns=.2)
        return baseline, defended

    def test_native_stab_hit_and_first_native_contact_before_hit_are_accepted(self):
        comparison = evaluate_ase_pair(*self.pair(), self.cfg)
        self.assertTrue(comparison["successful_coupled_parry_regardless_of_motion_source"])
        self.assertTrue(comparison["baseline_hit_in_native_stab"])
        self.assertTrue(comparison["first_blade_contact_in_both_native_clips"])
        self.assertTrue(comparison["successful_physical_parry"])

    def test_artificial_guard_transition_hit_cannot_be_attributed_to_ase_stab(self):
        baseline, defended = self.pair()
        baseline["first_target_hit_during_native_attack"] = False
        comparison = evaluate_ase_pair(baseline, defended, self.cfg)
        self.assertTrue(comparison["successful_coupled_parry_regardless_of_motion_source"])
        self.assertFalse(comparison["baseline_hit_in_native_stab"])
        self.assertFalse(comparison["successful_physical_parry"])

    def test_first_contact_in_authored_connection_is_rejected_despite_later_native_impulse(self):
        baseline, defended = self.pair()
        defended["first_positive_blade_contact_time_s"] = 1.
        defended["first_native_clip_blade_contact_time_s"] = 2.
        defended["native_clip_overlap_blade_impulse_ns"] = 10.
        comparison = evaluate_ase_pair(baseline, defended, self.cfg)
        self.assertTrue(comparison["successful_coupled_parry_regardless_of_motion_source"])
        self.assertFalse(comparison["first_blade_contact_in_both_native_clips"])
        self.assertFalse(comparison["successful_physical_parry"])

    def test_total_impulse_cannot_replace_native_overlap_impulse(self):
        baseline, defended = self.pair()
        defended["blade_contact_normal_impulse_ns"] = 10.
        defended["native_clip_overlap_blade_impulse_ns"] = .5*self.cfg["minimum_blade_contact_impulse_ns"]
        comparison = evaluate_ase_pair(baseline, defended, self.cfg)
        self.assertTrue(comparison["successful_coupled_parry_regardless_of_motion_source"])
        self.assertFalse(comparison["successful_physical_parry"])
        defended["native_clip_overlap_blade_impulse_ns"] = self.cfg["minimum_blade_contact_impulse_ns"]
        self.assertTrue(evaluate_ase_pair(baseline, defended, self.cfg)["successful_physical_parry"])

    def test_no_native_contact_is_not_a_native_parry(self):
        baseline, defended = self.pair()
        defended["first_native_clip_blade_contact_time_s"] = None
        defended["native_clip_overlap_blade_impulse_ns"] = 0.
        comparison = evaluate_ase_pair(baseline, defended, self.cfg)
        self.assertTrue(comparison["successful_coupled_parry_regardless_of_motion_source"])
        self.assertFalse(comparison["successful_physical_parry"])

    def test_contact_at_or_after_baseline_hit_cannot_be_causal_block(self):
        for contact_time in (3., 3.1):
            baseline, defended = self.pair()
            defended["first_positive_blade_contact_time_s"] = contact_time
            defended["first_native_clip_blade_contact_time_s"] = contact_time
            with self.subTest(contact_time=contact_time):
                self.assertFalse(evaluate_ase_pair(baseline, defended, self.cfg)["successful_physical_parry"])


if __name__ == "__main__":
    unittest.main()
