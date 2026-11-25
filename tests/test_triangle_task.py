"""Offline contracts, not physical attack/defence certificates (stdlib runner)."""

import math
import random
import unittest

from sim.triangle_task import (
    REGIONS, BalancedRegionScheduler, InvalidTickError, NativeTickEvents,
    QualifiedFoul, TargetRegion, TriangleGeometry, TriangleTask,
)


def geometry_mapping():
    return {
        "base_distance": 1.75,
        "regions": {
            "left": {"center": [0.32, 0.23, 0.9], "radius": 0.1},
            "right": {"center": [0.32, -0.23, 0.9], "radius": 0.1},
            "upper": {"center": [0.32, 0.0, 1.27], "radius": 0.1},
        },
    }


class TriangleTaskTests(unittest.TestCase):
    def setUp(self):
        self.geometry = TriangleGeometry.from_mapping(geometry_mapping())

    def test_world_mapping_and_mirror_involution(self):
        self.assertEqual(self.geometry.world_center("left", "a1"), (0.32, 0.23, 0.9))
        self.assertEqual(self.geometry.world_center("left", "a2"), (1.43, -0.23, 0.9))
        for point in [(0.3, 0.2, 1), (-0.4, -0.2, 0.4), (1.75, 0, -3)]:
            recovered = self.geometry.mirror_point(self.geometry.mirror_point(point))
            for actual, expected in zip(recovered, point):
                self.assertAlmostEqual(actual, expected)

    def test_geometry_owns_immutable_input_and_fresh_output(self):
        mapping = geometry_mapping()
        geometry = TriangleGeometry.from_mapping(mapping)
        mapping["regions"]["left"]["center"][1] = 100
        self.assertEqual(geometry.region("left").center[1], 0.23)
        exported = geometry.to_mapping()
        exported["regions"]["left"]["center"][1] = 200
        self.assertEqual(geometry.region("left").center[1], 0.23)
        with self.assertRaises(AttributeError):
            geometry.region("left").radius = 2

    def test_nonfinite_and_nonnumeric_geometry(self):
        for invalid in [math.nan, math.inf, -math.inf, True, "0.1", None]:
            for field in ["base_distance", "radius", "coordinate"]:
                with self.subTest(invalid=invalid, field=field):
                    mapping = geometry_mapping()
                    if field == "base_distance":
                        mapping[field] = invalid
                    elif field == "radius":
                        mapping["regions"]["left"][field] = invalid
                    else:
                        mapping["regions"]["left"]["center"][0] = invalid
                    with self.assertRaises(ValueError):
                        TriangleGeometry.from_mapping(mapping)

    def test_nonpositive_sizes(self):
        for field in ["base_distance", "radius"]:
            for value in [0, -1]:
                mapping = geometry_mapping()
                if field == "base_distance":
                    mapping[field] = value
                else:
                    mapping["regions"]["upper"][field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    TriangleGeometry.from_mapping(mapping)

    def test_nontriangular_layout(self):
        cases = [("right", 0, 0.4), ("upper", 0, 0.4), ("right", 1, -0.2),
                 ("left", 1, -0.23), ("right", 2, 0.8), ("upper", 1, 0.1),
                 ("upper", 2, 0.9), ("upper", 2, 0.8)]
        for region, coordinate, value in cases:
            mapping = geometry_mapping()
            mapping["regions"][region]["center"][coordinate] = value
            with self.subTest(case=(region, coordinate, value)), self.assertRaises(ValueError):
                TriangleGeometry.from_mapping(mapping)

    def test_strict_nonoverlap(self):
        for radius in [0.23, 0.24]:
            mapping = geometry_mapping()
            for name in ["left", "right"]:
                mapping["regions"][name]["radius"] = radius
            with self.assertRaisesRegex(ValueError, "non-overlapping"):
                TriangleGeometry.from_mapping(mapping)

    def test_malformed_geometry(self):
        mapping = geometry_mapping()
        del mapping["regions"]["upper"]
        with self.assertRaises(ValueError):
            TriangleGeometry.from_mapping(mapping)
        with self.assertRaises(ValueError):
            TriangleGeometry((self.geometry.region("left"), self.geometry.region("right"), self.geometry.region("left")), 1.75)
        with self.assertRaises(ValueError):
            TargetRegion("head", (0, 0, 1), 0.1)
        with self.assertRaises(ValueError):
            TriangleGeometry.from_mapping({**geometry_mapping(), "unsupported": True})
        with self.assertRaises(ValueError):
            self.geometry.world_center("left", "unknown")
        for point in [(0, 1), (0, math.nan, 1), None]:
            with self.assertRaises(ValueError):
                self.geometry.mirror_point(point)

    def test_balanced_independent_seeded_scheduler(self):
        global_state = random.getstate()
        one, two = BalancedRegionScheduler(12), BalancedRegionScheduler(12)
        sequence = [one.next() for _ in range(30)]
        self.assertEqual(sequence, [two.next() for _ in range(30)])
        for start in range(0, 30, 3):
            self.assertEqual(set(sequence[start:start + 3]), set(REGIONS))
        self.assertEqual(global_state, random.getstate())
        one.reset()
        self.assertEqual([one.next() for _ in range(30)], sequence)
        one.reset(13)
        self.assertNotEqual([one.next() for _ in range(30)], sequence)
        one.next()
        two.reset()
        self.assertEqual([two.next() for _ in range(30)], sequence)
        for seed in [True, "1", 1.2, None]:
            with self.assertRaises(ValueError):
                BalancedRegionScheduler(seed)

    def test_two_live_schedulers_do_not_share_pending_state(self):
        first, second, reference = (BalancedRegionScheduler(7) for _ in range(3))
        first.next()
        first.next()
        self.assertEqual([second.next() for _ in range(12)], [reference.next() for _ in range(12)])

    def test_public_task_identity_and_reset_distribution(self):
        task = TriangleTask(self.geometry, mode="region_targeted", seed=21)
        goals = []
        for _ in range(9):
            info = task.reset()
            goals.append(info["active_goal"])
            self.assertTrue(info["goal_public_to_both"])
            for flag in ["runtime_activation_allowed", "admitted_to_training", "live_scorer_replaced"]:
                self.assertFalse(info[flag])
            info["geometry"]["regions"]["left"]["radius"] = 9
            self.assertEqual(task.public_metadata()["geometry"]["regions"]["left"]["radius"], 0.1)
        for start in range(0, 9, 3):
            self.assertEqual(set(goals[start:start + 3]), set(REGIONS))
        free_info = TriangleTask(self.geometry).reset()
        self.assertIsNone(free_info["active_goal"])
        self.assertNotEqual(free_info["mode"], task.mode)

    def test_inactive_then_active_independent_entries(self):
        task = TriangleTask(self.geometry, mode="region_targeted", seed=3)
        active = task.reset()["active_goal"]
        inactive = next(name for name in REGIONS if name != active)
        first = task.step(NativeTickEvents(1, entries_a1={inactive}, entries_a2={inactive}))
        self.assertFalse(first.terminated)
        self.assertEqual(first.rewards, (0, 0))
        second = task.step(NativeTickEvents(2, entries_a1={active}))
        self.assertTrue(second.terminated)
        self.assertEqual(second.outcome, "target_hit")
        self.assertEqual(second.winner, "a1")
        self.assertEqual(second.rewards, (1, -1))

    def test_same_tick_targeted_and_unrestricted_double_hits(self):
        for mode in ["region_targeted", "unrestricted"]:
            task = TriangleTask(self.geometry, mode=mode)
            info = task.reset()
            active = info["active_goal"] or "left"
            result = task.step(NativeTickEvents(0, entries_a1=set(REGIONS), entries_a2={active}))
            self.assertEqual(result.outcome, "double_hit")
            self.assertIsNone(result.winner)
            self.assertEqual(result.rewards, (0, 0))

    def test_all_regions_same_reward_attacker_named(self):
        for region in REGIONS:
            for attacker in ["a1", "a2"]:
                task = TriangleTask(self.geometry)
                task.reset()
                result = task.step(NativeTickEvents(0, **{f"entries_{attacker}": {region}}))
                self.assertEqual(result.winner, attacker)
                self.assertEqual(result.rewards, (1, -1) if attacker == "a1" else (-1, 1))

    def test_foul_priority_over_hits_and_timeout(self):
        cases = [({"a1"}, (-1, 1), "a2"), ({"a2"}, (1, -1), "a1"),
                 ({"a1", "a2"}, (-1, -1), None)]
        for responsible, rewards, winner in cases:
            task = TriangleTask(self.geometry)
            task.reset()
            result = task.step(NativeTickEvents(10, entries_a1={"left"}, entries_a2={"upper"},
                fouls=(QualifiedFoul(responsible, "offline evidence fixture"),), timeout=True))
            self.assertTrue(result.terminated)
            self.assertEqual(result.rewards, rewards)
            self.assertEqual(result.winner, winner)
            self.assertIn("foul", result.outcome)
        task.reset()
        result = task.step(NativeTickEvents(0, fouls=(QualifiedFoul({"a1"}, "self contact"), QualifiedFoul({"a2"}, "floor contact"))))
        self.assertEqual(result.rewards, (-1, -1))

    def test_neutral_timeout_and_hit_priority(self):
        task = TriangleTask(self.geometry)
        task.reset()
        result = task.step(NativeTickEvents(0, neutral_contacts=8))
        self.assertEqual(result.outcome, "ongoing")
        self.assertEqual(result.rewards, (0, 0))
        result = task.step(NativeTickEvents(5, timeout=True, neutral_contacts=9))
        self.assertEqual(result.outcome, "timeout")
        self.assertEqual(result.rewards, (0, 0))
        self.assertIsNone(result.winner)
        task.reset()
        self.assertEqual(task.step(NativeTickEvents(0, entries_a1={"left"}, timeout=True)).outcome, "target_hit")

    def test_error_not_draw_and_precedes_foul(self):
        for field in ["numerical_error", "evidence_error"]:
            task = TriangleTask(self.geometry)
            task.reset()
            event = NativeTickEvents(0, entries_a1={"left"}, fouls=(QualifiedFoul({"a2"}, "fixture"),), **{field: "invalid fixture"})
            with self.assertRaisesRegex(InvalidTickError, "invalid fixture"):
                task.step(event)
            with self.assertRaises(RuntimeError):
                task.step(NativeTickEvents(1))
            task.reset()
            self.assertFalse(task.step(NativeTickEvents(0)).terminated)

    def test_round_lifecycle_ticks_and_seed_replay(self):
        task = TriangleTask(self.geometry, mode="region_targeted", seed=37)
        with self.assertRaises(RuntimeError):
            task.step(NativeTickEvents(0))
        first = task.reset()
        task.step(NativeTickEvents(5))
        for tick in [5, 4]:
            with self.assertRaises(ValueError):
                task.step(NativeTickEvents(tick))
        task.step(NativeTickEvents(6, timeout=True))
        with self.assertRaises(RuntimeError):
            task.step(NativeTickEvents(7))
        self.assertEqual(task.reset(seed=37)["active_goal"], first["active_goal"])
        self.assertFalse(task.step(NativeTickEvents(0)).terminated)

    def test_malformed_events_and_unqualified_fouls(self):
        cases = [{"tick": True}, {"tick": -1}, {"tick": math.nan}, {"entries_a1": "left"},
                 {"entries_a1": {"head"}}, {"entries_a2": [["left"]]}, {"fouls": ("raw contact",)},
                 {"timeout": 1}, {"neutral_contacts": -1}, {"neutral_contacts": True},
                 {"numerical_error": ""}, {"evidence_error": 2}]
        for kwargs in cases:
            with self.subTest(case=kwargs), self.assertRaises(ValueError):
                NativeTickEvents(**{"tick": 0, **kwargs})
        for responsible, evidence in [(set(), "fixture"), ({"a3"}, "fixture"), ("a1", "fixture"), ({"a1"}, ""), ({"a1"}, None)]:
            with self.assertRaises(ValueError):
                QualifiedFoul(responsible, evidence)

    def test_events_own_input_collections(self):
        hits, sides = {"left"}, {"a1"}
        fouls = [QualifiedFoul(sides, "fixture")]
        event = NativeTickEvents(0, entries_a1=hits, fouls=fouls)
        hits.add("right")
        sides.add("a2")
        fouls.clear()
        self.assertEqual(event.entries_a1, frozenset({"left"}))
        self.assertEqual(event.fouls[0].responsible, frozenset({"a1"}))

    def test_bad_task_config_and_event_type(self):
        with self.assertRaises(ValueError):
            TriangleTask(self.geometry, mode="secret_bonus")
        with self.assertRaises(ValueError):
            TriangleTask(geometry_mapping())
        task = TriangleTask(self.geometry)
        task.reset()
        with self.assertRaises(ValueError):
            task.step({"tick": 0})


if __name__ == "__main__":
    unittest.main()
