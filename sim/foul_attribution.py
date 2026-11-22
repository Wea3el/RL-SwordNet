"""Offline candidate motion-attribution rule, NOT deployed game/scoring code.

The rule describes a *convention* in the fixed-floor world frame, not intent
recognition or proof of physical causation. Callers must supply FIRST-contact,
PRE-impact velocities of the two material points corresponding to the same
identified contact, including angular motion. Their earlier world positions
need not coincide. The normal points from geom A to geom B. A previous-2-ms
sample is only a coarse candidate approximation, not an exact time-of-impact.
Contact force is not an input to attribution.

The caller must FIRST establish that the existing physical foul gate (for
example forbidden-contact force > 0.5 N) was triggered. This module does not
detect contacts, evaluate that gate, or turn a zero-force geometric contact
into a foul. ``physics_tick`` is the qualifying event's SETTLEMENT tick; cached
pre-impact velocities may come from an earlier first geometric contact. Never
backdate the penalty to that earlier zero-force first-contact tick.

The fixed candidate thresholds below have not been calibrated to hardware,
noise, timestep or native examples. Do not tune them to obtain desired labels.
Old native acceptance, collision masks, scoring, rewards and runners are not
modified or imported here. Output rewards are a proposed offline settlement,
not rewards applied to an environment; match winner is a separate field.

Reliable self/floor evidence follows the explicit owner-fault convention even
after external deflection; that convention does not reconstruct causation.
Unreliable input provenance is an evidence error, not a fabricated foul winner.
For cross-robot contact, reliable but previously deflected or unknown-history
motion is attribution-ambiguous and receives the proposed bilateral penalty.
Double/ambiguous foul rewards (-1, -1) are explicitly NON-zero-sum.
"""
from __future__ import annotations

from dataclasses import dataclass
import numbers
from typing import Iterable

import numpy as np


RULE_VERSION = "motion_foul_offline_candidate_v1"
ACTIVE_NORMAL_SPEED_EPSILON_M_S = .001
RELATIVE_CLOSING_EPSILON_M_S = .001
UNIT_NORMAL_TOLERANCE = 1e-6
OWNERS = ("a1", "a2")
LEGAL_CONTACTS = frozenset(("cross_blade_blade", "cross_blade_guard"))
FORBIDDEN_CONTACTS = frozenset(("illegal_cross_robot_nonblade_contact",
                              "illegal_self_contact", "illegal_floor_contact"))


@dataclass(frozen=True)
class ContactEvidence:
    """One externally established forbidden/legal first-contact observation.

    ``physics_tick`` is the externally qualified event settlement tick, which
    may be later than the cached first-geometric-contact pre-impact sample.
    ``preimpact_reliable`` describes the velocity/normal provenance, not merely
    whether values are finite. ``prior_legal_weapon_influence=None`` means the
    preceding history is unknown, not that its influence was ruled out.
    Ownership is the physical robot slot; it is not A1/A2 policy family.
    """

    physics_tick: int
    contact_id: str
    category: str
    owner_a: str | None
    owner_b: str | None
    velocity_a_world_m_s: tuple[float, float, float]
    velocity_b_world_m_s: tuple[float, float, float]
    normal_a_to_b_world: tuple[float, float, float]
    preimpact_reliable: bool
    prior_legal_weapon_influence: bool | None


def candidate_rule_config():
    """Return a fresh, JSON-compatible record of the fixed candidate rule."""
    return {"rule_version": RULE_VERSION,
            "status": "offline_candidate_not_deployed_or_calibrated",
            "frame": "fixed-floor world frame; not Galilean-invariant intent inference",
            "normal_convention": "unit world normal from geom A to geom B",
            "active_normal_speed_epsilon_m_s": ACTIVE_NORMAL_SPEED_EPSILON_M_S,
            "relative_closing_epsilon_m_s": RELATIVE_CLOSING_EPSILON_M_S,
            "unit_normal_tolerance": UNIT_NORMAL_TOLERANCE,
            "threshold_calibration": "NOT PERFORMED; constants fixed for this candidate version",
            "contact_force_used_for_attribution": False,
            "first_contact_preimpact_data_required": True,
            "upstream_qualifying_forbidden_event_required": True,
            "contact_and_force_threshold_detection": "NOT PROVIDED; caller retains existing physical gates",
            "physics_tick": "qualifying event settlement tick, not necessarily cached first-geometric-contact tick",
            "native_previous_step_velocities": "coarse candidate evidence, not exact time-of-impact or causal certification",
            "unreliable_input_provenance": "evidence_error; no settlement or reward",
            "cross_prior_legal_influence_or_unknown_history": "ambiguous; bilateral penalty; no winner",
            "self_and_floor": "reliable contact owner is responsible by convention even after prior deflection",
            "priority": ["numerical/evidence error", "foul", "target first touch"],
            "same_tick_multiple_contacts": "union of responsibility; never additive penalties; conflicting duplicate IDs rejected",
            "proposed_rewards": {"single_foul": {"responsible": -1., "other": 1.},
                                 "double_or_ambiguous_foul": {"a1": -1., "a2": -1.}},
            "double_foul_is_non_zero_sum": True,
            "live_runner_changed": False}


def _tick(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or value < 0:
        raise ValueError("A nonnegative integer physics tick is required")
    return int(value)


def _vector(value, name):
    try:
        result = np.asarray(value, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be a numeric three-vector") from exc
    if result.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    return result


def point_velocity_world(linear_velocity_world_m_s, angular_velocity_world_rad_s,
                         point_world_m, body_center_world_m):
    """v(point) = v(reference center) + omega cross (point - reference center).

    The linear velocity MUST be for the supplied reference center, not another
    site or center of mass. This helper neither reconstructs pre-impact state
    nor asserts that a caller's current velocity is an unperturbed velocity.
    """
    v, omega, point, center = (_vector(x, name) for x, name in (
        (linear_velocity_world_m_s, "linear velocity"),
        (angular_velocity_world_rad_s, "angular velocity"),
        (point_world_m, "world point"), (body_center_world_m, "reference center")))
    if not all(np.isfinite(x).all() for x in (v, omega, point, center)):
        raise ValueError("Nonfinite rigid-point kinematics are not attribution evidence")
    with np.errstate(over="ignore", invalid="ignore"):
        result = v + np.cross(omega, point-center)
    if not np.isfinite(result).all():
        raise ValueError("Overflowed rigid-point kinematics are not attribution evidence")
    return result


def _topology(evidence):
    if evidence.category not in LEGAL_CONTACTS | FORBIDDEN_CONTACTS:
        raise ValueError("Unknown contact category; do not silently assume legality or fault")
    if any(owner not in (*OWNERS, None) for owner in (evidence.owner_a, evidence.owner_b)):
        raise ValueError("Owners must be physical slots a1/a2, or None for the floor")
    if evidence.category == "illegal_floor_contact":
        if ((evidence.owner_a is None) == (evidence.owner_b is None)
                or not any(owner in OWNERS for owner in (evidence.owner_a, evidence.owner_b))):
            raise ValueError("Floor contact needs exactly one robot owner and one unowned floor")
    elif evidence.category == "illegal_self_contact":
        if evidence.owner_a not in OWNERS or evidence.owner_a != evidence.owner_b:
            raise ValueError("Self contact needs two geoms of the same robot")
    elif set((evidence.owner_a, evidence.owner_b)) != set(OWNERS):
        raise ValueError("Cross-robot contact needs distinct robot owners")


def attribute_contact(evidence: ContactEvidence):
    """Return one auditable offline verdict; never infer intent from force.

    Cross contact: cA = vA dot n, cB = -vB dot n, closing = cA+cB.
    One positive contribution and a stationary/retreating counterpart can give
    a single foul. Both positive gives a double foul; insufficient closing,
    possible earlier legal deflection or unknown history is ambiguous.
    Unreliable pre-impact evidence produces an error, not a fabricated winner.

    Self/floor fault is an explicit ownership convention, NOT motion-derived
    causation. Unreliable evidence is conservatively not settled at all.
    """
    if not isinstance(evidence, ContactEvidence):
        raise ValueError("ContactEvidence is required")
    tick = _tick(evidence.physics_tick)
    if not isinstance(evidence.contact_id, str) or not evidence.contact_id.strip():
        raise ValueError("Nonempty contact_id required for auditable evidence")
    if type(evidence.preimpact_reliable) is not bool:
        raise ValueError("preimpact_reliable must be explicitly True or False")
    if evidence.prior_legal_weapon_influence is not None and type(evidence.prior_legal_weapon_influence) is not bool:
        raise ValueError("prior_legal_weapon_influence must be True, False or unknown None")
    _topology(evidence)
    va = _vector(evidence.velocity_a_world_m_s, "velocity A")
    vb = _vector(evidence.velocity_b_world_m_s, "velocity B")
    normal = _vector(evidence.normal_a_to_b_world, "normal A to B")
    result = {"rule_version": RULE_VERSION, "physics_tick": tick,
              "contact_id": evidence.contact_id, "category": evidence.category,
              "owners": [evidence.owner_a, evidence.owner_b],
              "status": None, "responsible_owners": [], "reason": None,
              "preimpact_reliable": evidence.preimpact_reliable,
              "prior_legal_weapon_influence": evidence.prior_legal_weapon_influence,
              "normal_contribution_m_s": None, "relative_closing_m_s": None,
              "candidate_only": True, "causal_or_intent_identification": False}

    def verdict(status, owners, reason):
        result.update(status=status, responsible_owners=sorted(set(owners)), reason=reason)
        return result

    # Do not let an invalid native state manufacture either a foul winner or
    # immunity from a foul. Numerical evidence requires a separate error path.
    if not all(np.isfinite(x).all() for x in (va, vb, normal)):
        return verdict("numerical_fault", (), "nonfinite_contact_kinematics")
    if abs(float(np.linalg.norm(normal))-1.) > UNIT_NORMAL_TOLERANCE:
        return verdict("evidence_error", (), "normal_must_be_unit_world_A_to_B")
    with np.errstate(over="ignore", invalid="ignore"):
        ca, cb = float(va@normal), float(-vb@normal)
        closing = ca+cb
    if not np.isfinite((ca, cb, closing)).all():
        return verdict("numerical_fault", (), "nonfinite_derived_contact_motion")
    result["normal_contribution_m_s"] = {"geom_a": ca, "geom_b": cb}
    result["relative_closing_m_s"] = closing

    if not evidence.preimpact_reliable:
        return verdict("evidence_error", (), "unreliable_first_contact_preimpact_data_not_settled")
    if evidence.category in LEGAL_CONTACTS:
        return verdict("legal_neutral", (), "legal_weapon_contact_awards_no_reward_or_immunity")
    if evidence.category in {"illegal_self_contact", "illegal_floor_contact"}:
        owner = next(owner for owner in (evidence.owner_a, evidence.owner_b) if owner in OWNERS)
        return verdict("single_foul", (owner,), "explicit_self_or_floor_owner_convention_not_causal_inference")
    if evidence.prior_legal_weapon_influence is not False:
        return verdict("ambiguous_foul", OWNERS,
                       "possible_prior_legal_deflection" if evidence.prior_legal_weapon_influence
                       else "prior_legal_weapon_history_unknown")
    if closing <= RELATIVE_CLOSING_EPSILON_M_S:
        return verdict("ambiguous_foul", OWNERS, "insufficient_relative_normal_closing")
    active_a = ca > ACTIVE_NORMAL_SPEED_EPSILON_M_S
    active_b = cb > ACTIVE_NORMAL_SPEED_EPSILON_M_S
    if active_a and active_b:
        return verdict("double_foul", OWNERS, "both_contact_points_approach_along_normal")
    if active_a != active_b:
        owner = evidence.owner_a if active_a else evidence.owner_b
        return verdict("single_foul", (owner,), "single_normal_approach_other_stationary_or_retreating")
    return verdict("ambiguous_foul", OWNERS, "normal_motion_below_active_threshold")


def settle_tick(physics_tick: int, contacts: Iterable[dict] = (), *,
                hit_agents: Iterable[str] = (), numerical_faults: Iterable[str] = ()):
    """Settle ONE offline tick, once, order-independently.

    Numerical/evidence errors do not create wins or RL rewards. Otherwise any
    foul in this tick precedes a target touch in that SAME tick. No later tick
    may be supplied here to overturn a completed earlier first-touch round.
    The caller, not this stateless pure function, owns the round clock.
    """
    tick = _tick(physics_tick)
    contacts = list(contacts)
    hits = tuple(sorted(set(hit_agents)))
    if any(owner not in OWNERS for owner in hits):
        raise ValueError("Target hit agents must be physical slots a1/a2")
    faults = list(numerical_faults)
    if any(not isinstance(reason, str) or not reason.strip() for reason in faults):
        raise ValueError("Numerical faults need nonempty reason strings")
    errors, guilty, contact_statuses, ids = [], set(), [], []
    seen = {}
    statuses = {"legal_neutral", "single_foul", "double_foul", "ambiguous_foul",
                "numerical_fault", "evidence_error"}
    for item in contacts:
        if (not isinstance(item, dict) or item.get("rule_version") != RULE_VERSION
                or _tick(item.get("physics_tick")) != tick or item.get("status") not in statuses):
            raise ValueError("Only same-tick verdicts from this candidate rule may be merged")
        owners = item.get("responsible_owners")
        if not isinstance(owners, list) or any(owner not in OWNERS for owner in owners):
            raise ValueError("Malformed responsible owners in contact verdict")
        status = item["status"]
        if ((status == "single_foul" and len(set(owners)) != 1)
                or (status in {"double_foul", "ambiguous_foul"} and set(owners) != set(OWNERS))
                or (status in {"legal_neutral", "numerical_fault", "evidence_error"} and owners)):
            raise ValueError("Contact verdict responsibility contradicts its status")
        contact_id = item.get("contact_id")
        if not isinstance(contact_id, str) or not contact_id.strip():
            raise ValueError("Contact verdict requires contact_id")
        fingerprint = (status, tuple(sorted(set(owners))), item.get("category"))
        if contact_id in seen and seen[contact_id] != fingerprint:
            raise ValueError("Conflicting duplicate contact_id evidence cannot become a double foul")
        seen[contact_id] = fingerprint
        ids.append(contact_id)
        contact_statuses.append(status)
        if status == "numerical_fault":
            faults.append(f"contact:{contact_id}:{item['reason']}")
        elif status == "evidence_error":
            errors.append(f"contact:{contact_id}:{item['reason']}")
        else:
            guilty.update(owners)

    result = {"rule_version": RULE_VERSION, "physics_tick": tick,
              "status": "ongoing", "terminal": False, "match_winner": None,
              "rl_reward": {"a1": 0., "a2": 0.}, "reward_sum": 0.,
              "zero_sum_reward": True, "responsible_owners": sorted(guilty),
              "contact_ids": sorted(set(ids)), "target_hits_this_tick": list(hits),
              "foul_precedes_same_tick_hit": bool(guilty and hits),
              "numerical_faults": sorted(set(faults)), "evidence_errors": sorted(set(errors)),
              "candidate_only": True, "live_runner_changed": False,
              "training_admission": False,
              "competition_points": None,
              "competition_points_note": "Not defined here; RL reward and match winner are separate"}
    if faults or errors:
        result.update(status="numerical_fault" if faults else "evidence_error",
                      terminal=True, rl_reward=None, reward_sum=None, zero_sum_reward=None,
                      responsible_owners=[], valid_for_reward=False)
        return result
    result["valid_for_reward"] = True
    if guilty:
        if guilty == set(OWNERS):
            result.update(status="ambiguous_foul" if "ambiguous_foul" in contact_statuses else "double_foul",
                          terminal=True, rl_reward={"a1": -1., "a2": -1.},
                          reward_sum=-2., zero_sum_reward=False)
        else:
            owner = next(iter(guilty))
            other = OWNERS[1] if owner == OWNERS[0] else OWNERS[0]
            result.update(status="foul_win", terminal=True, match_winner=other,
                          rl_reward={owner: -1., other: 1.})
    elif len(hits) == 2:
        result.update(status="double_touch", terminal=True)
    elif hits:
        owner = hits[0]
        other = OWNERS[1] if owner == OWNERS[0] else OWNERS[0]
        result.update(status="target_win", terminal=True, match_winner=owner,
                      rl_reward={owner: 1., other: -1.})
    return result
