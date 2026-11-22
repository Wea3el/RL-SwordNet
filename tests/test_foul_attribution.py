"""Offline proposed-rule contracts, NOT native validation or game deployment."""
from dataclasses import replace
import itertools
import json
import unittest

import numpy as np

from sim.foul_attribution import (
    ACTIVE_NORMAL_SPEED_EPSILON_M_S, ContactEvidence, RULE_VERSION,
    attribute_contact, candidate_rule_config, point_velocity_world, settle_tick)


def evidence(**changes):
    base = ContactEvidence(physics_tick=17, contact_id="pair_1",
        category="illegal_cross_robot_nonblade_contact", owner_a="a1", owner_b="a2",
        velocity_a_world_m_s=(.1, 0., 0.), velocity_b_world_m_s=(0., 0., 0.),
        normal_a_to_b_world=(1., 0., 0.), preimpact_reliable=True,
        prior_legal_weapon_influence=False)
    return replace(base, **changes)


class PointKinematicsTests(unittest.TestCase):
    def test_rotating_body_uses_contact_point_not_stationary_center(self):
        point_speed = point_velocity_world([0, 0, 0], [0, 0, 2], [0, -1, 0], [0, 0, 0])
        np.testing.assert_array_equal(point_speed, [2, 0, 0])
        verdict = attribute_contact(evidence(velocity_a_world_m_s=tuple(point_speed)))
        self.assertEqual(verdict["responsible_owners"], ["a1"])

    def test_translation_and_yaw_pi_preserve_point_velocity(self):
        v, omega, point, center = map(np.asarray, ([.1, .3, -.2], [1., 2., 3.],
            [.4, .5, .7], [.1, -.2, .3]))
        base = point_velocity_world(v, omega, point, center)
        offset = np.array([100., -35., 11.])
        np.testing.assert_allclose(point_velocity_world(v, omega, point+offset, center+offset), base)
        yaw = np.diag([-1., -1., 1.])
        np.testing.assert_allclose(point_velocity_world(yaw@v, yaw@omega, yaw@point, yaw@center), yaw@base)

    def test_nonfinite_and_malformed_kinematics_are_not_silently_used(self):
        for bad in ([np.nan, 0., 0.], [np.inf, 0., 0.], [1., 2.]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                point_velocity_world(bad, [0, 0, 0], [0, 0, 0], [0, 0, 0])


class MotionAttributionTests(unittest.TestCase):
    def test_single_approacher_vs_stationary_or_retreating(self):
        for retreat in (0., .03):
            verdict = attribute_contact(evidence(velocity_b_world_m_s=(retreat, 0., 0.)))
            self.assertEqual(verdict["status"], "single_foul")
            self.assertEqual(verdict["responsible_owners"], ["a1"])
        other = attribute_contact(evidence(velocity_a_world_m_s=(0., 0., 0.),
                                           velocity_b_world_m_s=(-.1, 0., 0.)))
        self.assertEqual(other["responsible_owners"], ["a2"])

    def test_faster_retreat_is_not_mislabelled_as_closing(self):
        verdict = attribute_contact(evidence(velocity_b_world_m_s=(.2, 0., 0.)))
        self.assertEqual(verdict["status"], "ambiguous_foul")
        self.assertLess(verdict["relative_closing_m_s"], 0.)

    def test_head_on_motion_is_double_foul(self):
        verdict = attribute_contact(evidence(velocity_b_world_m_s=(-.04, 0., 0.)))
        self.assertEqual(verdict["status"], "double_foul")
        self.assertEqual(verdict["responsible_owners"], ["a1", "a2"])

    def test_pure_tangential_and_near_zero_normal_motion_are_ambiguous(self):
        for va, vb in (((0., 3., 0.), (0., -2., 0.)),
                       ((.0009, 0., 0.), (-.0009, 0., 0.)),
                       ((0., 0., 0.), (0., 0., 0.))):
            with self.subTest(va=va, vb=vb):
                self.assertEqual(attribute_contact(evidence(velocity_a_world_m_s=va,
                    velocity_b_world_m_s=vb))["status"], "ambiguous_foul")

    def test_active_threshold_is_fixed_explicit_and_strict(self):
        eps = ACTIVE_NORMAL_SPEED_EPSILON_M_S
        low = attribute_contact(evidence(velocity_a_world_m_s=(eps, 0., 0.)))
        high = attribute_contact(evidence(velocity_a_world_m_s=(eps+1e-6, 0., 0.)))
        self.assertEqual(low["status"], "ambiguous_foul")
        self.assertEqual(high["status"], "single_foul")
        self.assertEqual(candidate_rule_config()["active_normal_speed_epsilon_m_s"], .001)

    def test_geom_order_swap_and_yaw_pi_do_not_change_owner_verdict(self):
        base = evidence(velocity_a_world_m_s=(.1, .03, .02),
                        velocity_b_world_m_s=(.02, -.04, .01))
        swapped = replace(base, owner_a=base.owner_b, owner_b=base.owner_a,
            velocity_a_world_m_s=base.velocity_b_world_m_s,
            velocity_b_world_m_s=base.velocity_a_world_m_s,
            normal_a_to_b_world=tuple(-np.asarray(base.normal_a_to_b_world)))
        yaw = np.diag([-1., -1., 1.])
        rotated = replace(base,
            velocity_a_world_m_s=tuple(yaw@base.velocity_a_world_m_s),
            velocity_b_world_m_s=tuple(yaw@base.velocity_b_world_m_s),
            normal_a_to_b_world=tuple(yaw@base.normal_a_to_b_world))
        verdicts = [attribute_contact(x) for x in (base, swapped, rotated)]
        self.assertTrue(all(x["status"] == "single_foul" for x in verdicts))
        self.assertTrue(all(x["responsible_owners"] == ["a1"] for x in verdicts))
        self.assertTrue(all(np.isclose(x["relative_closing_m_s"], .08) for x in verdicts))

    def test_prior_deflection_or_unknown_history_never_gives_single_winner(self):
        for changes in ({"prior_legal_weapon_influence": True},
                        {"prior_legal_weapon_influence": None}):
            with self.subTest(changes=changes):
                verdict = attribute_contact(evidence(**changes))
                self.assertEqual(verdict["status"], "ambiguous_foul")
                settled = settle_tick(17, [verdict])
                self.assertIsNone(settled["match_winner"])
                self.assertEqual(settled["rl_reward"], {"a1": -1., "a2": -1.})

    def test_unreliable_input_provenance_is_not_a_fabricated_foul_or_reward(self):
        for changes in ({}, {"category": "cross_blade_guard"},
                        {"category": "illegal_self_contact", "owner_b": "a1"},
                        {"category": "illegal_floor_contact", "owner_b": None}):
            verdict = attribute_contact(evidence(preimpact_reliable=False, **changes))
            settled = settle_tick(17, [verdict], hit_agents=["a1"])
            self.assertEqual(settled["status"], "evidence_error")
            self.assertIsNone(settled["match_winner"])
            self.assertIsNone(settled["rl_reward"])

    def test_legal_weapon_contact_awards_nothing_even_with_closing(self):
        for category in ("cross_blade_blade", "cross_blade_guard"):
            verdict = attribute_contact(evidence(category=category,
                prior_legal_weapon_influence=True))
            self.assertEqual(verdict["status"], "legal_neutral")
            settled = settle_tick(17, [verdict])
            self.assertFalse(settled["terminal"])
            self.assertEqual(settled["rl_reward"], {"a1": 0., "a2": 0.})

    def test_self_and_floor_contacts_use_explicit_ownership_convention(self):
        for category, owners in (("illegal_self_contact", ("a1", "a1")),
                                  ("illegal_floor_contact", (None, "a2"))):
            verdict = attribute_contact(evidence(category=category, owner_a=owners[0], owner_b=owners[1]))
            self.assertEqual(verdict["responsible_owners"], [next(o for o in owners if o)])
            self.assertIn("convention", verdict["reason"])
            externally_deflected = attribute_contact(evidence(category=category,
                owner_a=owners[0], owner_b=owners[1], prior_legal_weapon_influence=True))
            self.assertEqual(externally_deflected["responsible_owners"], verdict["responsible_owners"])

    def test_nonfinite_and_invalid_normals_never_manufacture_a_winner(self):
        for changes, expected in (({"velocity_a_world_m_s": (np.nan, 0., 0.)}, "numerical_fault"),
                                  ({"normal_a_to_b_world": (0., 0., 0.)}, "evidence_error"),
                                  ({"normal_a_to_b_world": (2., 0., 0.)}, "evidence_error")):
            verdict = attribute_contact(evidence(**changes))
            self.assertEqual(verdict["status"], expected)
            settled = settle_tick(17, [verdict], hit_agents=["a2"])
            self.assertEqual(settled["status"], expected)
            self.assertIsNone(settled["match_winner"])
            self.assertIsNone(settled["rl_reward"])
            self.assertFalse(settled["valid_for_reward"])

    def test_malformed_topology_flags_and_vectors_are_contract_errors(self):
        for changes in ({"category": "unknown"}, {"owner_a": "robotA"},
                        {"owner_a": None}, {"category": "illegal_self_contact"},
                        {"physics_tick": True}, {"contact_id": ""},
                        {"velocity_a_world_m_s": (1., 2.)}, {"preimpact_reliable": 1},
                        {"prior_legal_weapon_influence": "unknown"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                attribute_contact(evidence(**changes))


class TickSettlementTests(unittest.TestCase):
    def test_single_foul_separates_reward_match_winner_and_undefined_competition_points(self):
        settled = settle_tick(17, [attribute_contact(evidence())])
        self.assertEqual(settled["status"], "foul_win")
        self.assertEqual(settled["match_winner"], "a2")
        self.assertEqual(settled["rl_reward"], {"a1": -1., "a2": 1.})
        self.assertTrue(settled["zero_sum_reward"])
        self.assertIsNone(settled["competition_points"])
        self.assertFalse(settled["live_runner_changed"])

    def test_double_foul_is_not_a_zero_reward_draw(self):
        verdict = attribute_contact(evidence(velocity_b_world_m_s=(-.1, 0., 0.)))
        settled = settle_tick(17, [verdict])
        self.assertEqual(settled["status"], "double_foul")
        self.assertEqual(settled["rl_reward"], {"a1": -1., "a2": -1.})
        self.assertEqual(settled["reward_sum"], -2.)
        self.assertFalse(settled["zero_sum_reward"])
        self.assertIsNone(settled["match_winner"])

    def test_same_tick_contacts_are_unioned_once_independent_of_order_or_duplicates(self):
        a = attribute_contact(evidence(contact_id="pair_a"))
        b = attribute_contact(evidence(contact_id="pair_b", velocity_a_world_m_s=(0., 0., 0.),
            velocity_b_world_m_s=(-.1, 0., 0.)))
        legal = attribute_contact(evidence(contact_id="legal", category="cross_blade_guard"))
        outcomes = [settle_tick(17, perm) for perm in itertools.permutations((a, b, legal, a))]
        self.assertTrue(all(x == outcomes[0] for x in outcomes))
        self.assertEqual(outcomes[0]["rl_reward"], {"a1": -1., "a2": -1.})
        self.assertEqual(outcomes[0]["contact_ids"], ["legal", "pair_a", "pair_b"])
        duplicate = settle_tick(17, [a, a, legal])
        self.assertEqual(duplicate["rl_reward"], {"a1": -1., "a2": 1.})

    def test_same_tick_foul_precedes_hit_even_by_otherwise_winning_agent(self):
        settled = settle_tick(17, [attribute_contact(evidence())], hit_agents=["a1"])
        self.assertEqual(settled["match_winner"], "a2")
        self.assertTrue(settled["foul_precedes_same_tick_hit"])
        double = attribute_contact(evidence(velocity_b_world_m_s=(-.1, 0., 0.)))
        self.assertIsNone(settle_tick(17, [double], hit_agents=["a1", "a2"])["match_winner"])

    def test_contradictory_duplicate_contact_ids_are_rejected_not_merged_into_a_foul(self):
        a = attribute_contact(evidence())
        b = attribute_contact(evidence(velocity_a_world_m_s=(0., 0., 0.),
            velocity_b_world_m_s=(-.1, 0., 0.)))
        for pair in ((a, b), (b, a), (a, dict(a, status="ambiguous_foul", responsible_owners=["a1", "a2"]))):
            with self.subTest(pair=pair), self.assertRaisesRegex(ValueError, "Conflicting duplicate"):
                settle_tick(17, pair)

    def test_numerical_fault_precedes_foul_and_hit_without_fake_reward(self):
        settled = settle_tick(17, [attribute_contact(evidence())], hit_agents=["a1"],
            numerical_faults=["mujoco_warning", "mujoco_warning"])
        self.assertEqual(settled["status"], "numerical_fault")
        self.assertIsNone(settled["match_winner"])
        self.assertIsNone(settled["rl_reward"])
        self.assertFalse(settled["valid_for_reward"])
        self.assertEqual(settled["numerical_faults"], ["mujoco_warning"])

    def test_no_contact_and_legal_only_hit_settlements(self):
        self.assertFalse(settle_tick(17)["terminal"])
        self.assertEqual(settle_tick(17, hit_agents=["a1"])["match_winner"], "a1")
        both = settle_tick(17, hit_agents=["a1", "a2", "a1"])
        self.assertEqual(both["status"], "double_touch")
        self.assertEqual(both["rl_reward"], {"a1": 0., "a2": 0.})

    def test_mixed_ticks_and_invalid_owners_cannot_overturn_completed_round(self):
        verdict = attribute_contact(evidence())
        for kwargs in ({"physics_tick": 18, "contacts": [verdict]},
                       {"physics_tick": 17, "hit_agents": ["robotA"]},
                       {"physics_tick": 17, "numerical_faults": [""]},
                       {"physics_tick": 17, "contacts": [dict(verdict, rule_version="other")]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                settle_tick(**kwargs)

    def test_json_serializable_candidate_config_and_settlement(self):
        config = candidate_rule_config()
        self.assertEqual(config["rule_version"], RULE_VERSION)
        self.assertIn("NOT PERFORMED", config["threshold_calibration"])
        self.assertTrue(config["upstream_qualifying_forbidden_event_required"])
        self.assertIn("NOT PROVIDED", config["contact_and_force_threshold_detection"])
        json.dumps(config, allow_nan=False)
        json.dumps(settle_tick(17, [attribute_contact(evidence())]), allow_nan=False)


if __name__ == "__main__":
    unittest.main()
