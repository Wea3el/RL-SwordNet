"""Skill-independent, auditable first-touch rules for the fixed-arm game.

Version 2 scores the *physical flat-ended blade cylinders*, not a tip, an
animation label, a capsule extended behind the guard, or a declared block.
Blade/weapon contacts themselves never award points or immunity. The native
adapter reads every numerically indexed blade collision proxy in the model:
their first endpoint is the blade-side ``shinai_guard_far``, NOT guard_near.

The sweep interpolates both cylinder-axis endpoints between successive native
physics states. It is continuous for that interpolation, not a claim that two
samples reconstruct an arbitrary fast curved rigid-body trajectory exactly.
Call it every physics step, and repeat boundary cases at a smaller timestep.
An adaptive Lipschitz search certifies empty intervals or finds an actual
interior witness. Exhausted/ambiguous searches invalidate the round; they are
not silently converted into a hit or a harmless draw.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import weakref

import numpy as np


RULES_VERSION = "whole_blade_first_touch_v2"
BLADE_START_SITE = "shinai_guard_far"
_DISTANCE_EPS_M = 1e-10
_FRACTION_EPS = 1e-9
_TIME_EPS_S = 1e-10
_MAX_EVALUATIONS = 8192
_ADAPTER_TOPOLOGY_CACHE = weakref.WeakKeyDictionary()


@dataclass(frozen=True)
class BladeSection:
    """World-space axis endpoints and actual radius of a finite cylinder.

    Arrays are copied so a subsequent MuJoCo forward/step cannot mutate an
    earlier snapshot. Nonfinite endpoint values are retained for the scorer
    to invalidate with evidence, rather than discarded by a filtering adapter.
    """
    start: np.ndarray
    end: np.ndarray
    radius_m: float
    name: str = ""

    def __post_init__(self):
        for field in ("start", "end"):
            value = np.asarray(getattr(self, field), dtype=float).copy()
            if value.shape != (3,):
                raise ValueError("Blade endpoints must have shape (3,)")
            value.flags.writeable = False
            object.__setattr__(self, field, value)
        radius = float(self.radius_m)
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("A positive finite native blade radius is required")
        object.__setattr__(self, "radius_m", radius)
        object.__setattr__(self, "name", str(self.name))


def blade_sections_from_scene(scene):
    """Read all native flat-ended blade proxies for each agent.

    Call ``mj_forward`` after changing state and before taking a snapshot.
    No radius is invented or borrowed from the old 10mm tip-scoring setting.
    This function neither changes collision masks nor classifies contacts.
    Model-local enumeration also supports historical three-cylinder XMLs;
    it never assumes that a current profile has only three scoring sections.
    """
    from .blade_geometry import blade_collision_geom_ids

    result = {}
    topology = _ADAPTER_TOPOLOGY_CACHE.setdefault(scene.model, {})
    for owner, mapping in scene.agents.items():
        prefix = mapping["prefix"]
        if prefix not in topology:
            ids = np.asarray(blade_collision_geom_ids(scene.model, prefix), dtype=int)
            topology[prefix] = ids, tuple(scene.model.geom(int(i)).name for i in ids)
        ids, names = topology[prefix]
        # Cache only immutable model topology, never q-dependent positions,
        # orientations or radii. Every snapshot still owns its endpoint data.
        radii = scene.model.geom_size[ids, 0]
        half = scene.model.geom_size[ids, 1, None]
        directions = scene.data.geom_xmat[ids].reshape(-1, 3, 3)[:, :, 2]
        centers = scene.data.geom_xpos[ids]
        starts, ends = centers-half*directions, centers+half*directions
        result[owner] = tuple(BladeSection(a, b, r, name)
                              for a, b, r, name in zip(starts, ends, radii, names))
    return result


def point_cylinder_distance(point, section):
    """Euclidean point-to-solid-finite-cylinder distance and closest point.

    Unlike point-to-capsule distance, the caps do not protrude past either
    blade endpoint. This matters at the guard where neutral weapon contact
    must not be turned into an artificial blade hit.
    """
    point = np.asarray(point, dtype=float)
    axis = section.end-section.start
    length = float(np.linalg.norm(axis))
    if point.shape != (3,) or not np.isfinite(point).all() or not np.isfinite(length) or length <= 1e-12:
        raise ValueError("Finite point and nondegenerate cylinder required")
    direction = axis/length
    offset = point-section.start
    axial = float(offset@direction)
    radial = offset-axial*direction
    radial_distance = float(np.linalg.norm(radial))
    closest = section.start+np.clip(axial, 0., length)*direction
    if radial_distance > 0:
        closest = closest+radial*min(1., section.radius_m/radial_distance)
    distance = float(np.hypot(max(0., radial_distance-section.radius_m),
                              max(0., -axial, axial-length)))
    return distance, closest


def _point_cylinder_distances(point, starts, ends, radii):
    """Batch the unchanged finite-cylinder formula; preserve row order/ties.

    This is the same solid side/cap/corner distance as the public scalar
    function, not a capsule, bounding sphere, profile interpolation or new
    contact tolerance. The first minimal row remains the witness on a tie.
    """
    point = np.asarray(point, dtype=float)
    starts, ends, radii = (np.asarray(v, dtype=float) for v in (starts, ends, radii))
    if (point.shape != (3,) or starts.ndim != 2 or starts.shape[1:] != (3,)
            or not len(starts) or ends.shape != starts.shape or radii.shape != (len(starts),)
            or not all(np.isfinite(v).all() for v in (point, starts, ends, radii))
            or np.any(radii <= 0)):
        raise ValueError("Finite point and nondegenerate cylinder required")
    axis = ends-starts
    length = np.linalg.norm(axis, axis=1)
    if not np.isfinite(length).all() or np.any(length <= 1e-12):
        raise ValueError("Finite point and nondegenerate cylinder required")
    direction = axis/length[:, None]
    offset = point-starts
    axial = np.sum(offset*direction, axis=1)
    radial = offset-axial[:, None]*direction
    radial_distance = np.linalg.norm(radial, axis=1)
    closest = starts+np.clip(axial, 0., length)[:, None]*direction
    ratio = np.ones_like(radial_distance)
    np.divide(radii, radial_distance, out=ratio, where=radial_distance > 0)
    closest += radial*np.minimum(1., ratio)[:, None]
    distance = np.hypot(np.maximum(0., radial_distance-radii),
                        np.maximum(np.maximum(0., -axial), axial-length))
    return distance, closest


class _UnresolvedSweep(RuntimeError):
    pass


def _motion_bound(before, after):
    """Hausdorff-motion Lipschitz bound per unit physics-step fraction.

    Centerline points move no faster than the faster endpoint. A parallel-
    transported radial vector moves at most radius * |axis'| / min|axis|.
    The minimum interpolated axis length is solved, not assumed from frames.
    """
    first_start = np.asarray([s.start for s in before])
    first_end = np.asarray([s.end for s in before])
    last_start = np.asarray([s.start for s in after])
    last_end = np.asarray([s.end for s in after])
    return _motion_bound_arrays(first_start, first_end, last_start, last_end,
                                np.asarray([s.radius_m for s in before]))


def _motion_bound_arrays(first_start, first_end, last_start, last_end, radii):
    axis = first_end-first_start
    delta = (last_end-last_start)-axis
    dd = np.sum(delta*delta, axis=1)
    fraction = np.zeros_like(dd)
    np.divide(-np.sum(axis*delta, axis=1), dd, out=fraction, where=dd != 0)
    fraction = np.clip(fraction, 0., 1.)
    minimum_length = np.linalg.norm(axis+fraction[:, None]*delta, axis=1)
    if np.any(minimum_length <= 1e-10):
        raise _UnresolvedSweep("interpolated_blade_axis_degenerate")
    endpoint_speed = np.maximum(np.linalg.norm(last_start-first_start, axis=1),
                                np.linalg.norm(last_end-first_end, axis=1))
    return float(np.max(endpoint_speed+radii*np.linalg.norm(delta, axis=1)/minimum_length))


def _first_strictly_negative(evaluate, left, right, lipschitz, *, evaluation_budget):
    """Earliest certified negative witness, with a bounded entry bracket.

    ``evaluate(t)`` is distance clearance, or its negative when looking for a
    full exit. Returns None only when every searched interval is certified to
    contain no value below -distance_epsilon. The search visits left first.
    A tiny uncertified interval without an interior witness does not by
    itself prove a miss OR make a later witnessed entry invalid: retain its
    left bound and keep searching. A subsequent genuine interior witness
    proves an entry, with its bracket widened to cover that earlier interval.
    If no witness exists and any interval remains uncertified, invalidate.
    """
    eps = _DISTANCE_EPS_M
    first, last = evaluate(left), evaluate(right)
    if first < -eps:
        return left, (left, left)
    if lipschitz <= 1e-15:
        return None
    stack = [(left, right, first, last)]
    earliest_uncertified = None
    while stack:
        lo, hi, flo, fhi = stack.pop()
        width = hi-lo
        if flo < -eps:
            bracket_lo = lo if earliest_uncertified is None else earliest_uncertified
            return lo, (bracket_lo, lo)
        # Two endpoint cones give this global interval lower bound. Use a
        # small roundoff slack: an optimistic bound must never skip a hit.
        lower = .5*(flo+fhi-lipschitz*width)-1e-14
        if lower >= -eps:
            continue
        if width <= _FRACTION_EPS:
            if fhi < -eps:
                bracket_lo = lo if earliest_uncertified is None else earliest_uncertified
                return hi, (bracket_lo, hi)
            if earliest_uncertified is None:
                earliest_uncertified = lo
            continue
        mid = .5*(lo+hi)
        evaluation_budget[0] += 1
        if evaluation_budget[0] > _MAX_EVALUATIONS:
            raise _UnresolvedSweep("swept_distance_evaluation_budget_exhausted")
        fmid = evaluate(mid)
        stack.append((mid, hi, fmid, fhi))
        stack.append((lo, mid, flo, fmid))
    if earliest_uncertified is not None:
        raise _UnresolvedSweep("unresolved_boundary_or_substep_entry")
    return None


class FirstTouchScorer:
    """One independent round; geometry-only attacks, no animation state input.

    ``centers`` maps ATTACKER to the opponent's target center. Target radii
    are actual sphere radii, not already inflated blade/tip radii. ``update``
    returns all physical outside->entry events in that step. ``events`` holds
    the first-touch deciding event(s); ``all_entries`` also retains subsequent
    late entries, so a block's recovery cannot hide a delayed target hit.

    Starting overlapped disarms that blade until it is fully outside all its
    sections. Both owners entering in the same physics step is double touch,
    even if estimated within-step times differ. Numerical failures override a
    provisional result as invalid; they are never a zero-point timeout draw.
    """
    def __init__(self, centers: Mapping, target_radius):
        if len(centers) != 2:
            raise ValueError("Exactly two agents/attacker target centers required")
        self.centers = {k: np.asarray(v, dtype=float).copy() for k, v in centers.items()}
        if any(c.shape != (3,) or not np.isfinite(c).all() for c in self.centers.values()):
            raise ValueError("Finite target centers with shape (3,) required")
        radii = target_radius if isinstance(target_radius, Mapping) else {k: target_radius for k in centers}
        if set(radii) != set(centers):
            raise ValueError("A target radius is required for each attacker")
        self.target_radii = {k: float(v) for k, v in radii.items()}
        if any(not np.isfinite(r) or r <= 0 for r in self.target_radii.values()):
            raise ValueError("Positive finite target radii required")
        self.events, self.all_entries, self.initial_overlap_events = [], [], []
        self.failure = None
        self.minimum_clearance_m = {k: np.inf for k in centers}
        self.armed = {k: None for k in centers}
        self._last, self._last_time = None, None
        self._outcome = {"status": "ongoing", "winner": None,
                         "points": {k: 0 for k in centers}, "time_s": None}

    @property
    def round_result(self):
        result = dict(self._outcome, points=dict(self._outcome["points"]), rules_version=RULES_VERSION)
        if self.failure:
            result.update(status="invalid", winner=None, points={k: 0 for k in self.centers},
                          failure=self.failure, provisional_result=dict(self._outcome))
        return result

    def invalidate(self, reason, time_s, details=None):
        """Latch a native numerical/safety fault without losing touch evidence."""
        if self.failure is None:
            time = float(time_s)
            self.failure = {"reason": str(reason), "time_s": time if np.isfinite(time) else None,
                            "details": details or {}}

    def finish(self, time_s):
        """Close a valid no-touch round as timeout; not a numeric-fault escape."""
        if not np.isfinite(time_s) or (self._last_time is not None and time_s < self._last_time-_TIME_EPS_S):
            self.invalidate("invalid_finish_time", time_s)
        elif self._outcome["status"] == "ongoing" and self.failure is None:
            self._outcome.update(status="timeout_draw", time_s=float(time_s))
        return self.round_result

    def _check_frames(self, before, after, t0, t1):
        if not np.isfinite([t0, t1]).all() or t0 < 0 or t1 < t0:
            raise ValueError("nonnegative_finite_step_interval_required")
        initialization = self._last_time is None and t0 == t1
        if t0 == t1 and not initialization:
            raise ValueError("zero_length_step_only_allowed_for_initial_snapshot")
        if self._last_time is not None and abs(t0-self._last_time) > _TIME_EPS_S:
            raise ValueError("noncontiguous_physics_step_times")
        if set(before) != set(self.centers) or set(after) != set(self.centers):
            raise ValueError("both_blade_snapshots_required")
        packed = {}
        for owner in self.centers:
            first, last = tuple(before[owner]), tuple(after[owner])
            if not first or len(first) != len(last):
                raise ValueError("matching_nonempty_blade_sections_required")
            if self._last is not None and len(self._last[owner]) != len(first):
                raise ValueError("blade_section_count_changed")
            if not all(isinstance(s, BladeSection) for s in (*first, *last)):
                raise ValueError("BladeSection_snapshots_required")
            starts = np.asarray([s.start for s in first])
            ends = np.asarray([s.end for s in first])
            new_starts = np.asarray([s.start for s in last])
            new_ends = np.asarray([s.end for s in last])
            radii = np.asarray([s.radius_m for s in first])
            new_radii = np.asarray([s.radius_m for s in last])
            names = tuple(s.name for s in first)
            new_names = tuple(s.name for s in last)
            valid = (all(np.isfinite(v).all() for v in (starts, ends, new_starts, new_ends))
                     and names == new_names and np.all(np.abs(radii-new_radii) <= 1e-12))
            if initialization:
                valid = valid and np.array_equal(starts, new_starts) and np.array_equal(ends, new_ends)
            if self._last is not None:
                old = self._last[owner]
                valid = (valid and tuple(s.name for s in old) == names
                         and np.array_equal(np.asarray([s.radius_m for s in old]), radii)
                         and np.allclose(np.asarray([s.start for s in old]), starts, rtol=0., atol=1e-10)
                         and np.allclose(np.asarray([s.end for s in old]), ends, rtol=0., atol=1e-10))
            packed[owner] = starts, ends, new_starts, new_ends, radii
            if valid:
                continue
            # Preserve the scalar invalid-input error ordering. The common
            # valid case does its identical comparisons in batches instead
            # of 100+ per-section allclose calls every physics tick.
            for index, (a, b) in enumerate(zip(first, last)):
                if not isinstance(a, BladeSection) or not isinstance(b, BladeSection):
                    raise ValueError("BladeSection_snapshots_required")
                if not all(np.isfinite(v).all() for v in (a.start, a.end, b.start, b.end)):
                    raise ValueError("nonfinite_blade_state")
                if a.name != b.name or abs(a.radius_m-b.radius_m) > 1e-12:
                    raise ValueError("blade_geometry_changed_during_step")
                if initialization and (not np.array_equal(a.start, b.start) or not np.array_equal(a.end, b.end)):
                    raise ValueError("initial_snapshot_cannot_move_at_zero_time")
                if self._last is not None:
                    old = self._last[owner][index]
                    if (old.name != a.name or old.radius_m != a.radius_m
                            or not np.allclose(old.start, a.start, rtol=0., atol=1e-10)
                            or not np.allclose(old.end, a.end, rtol=0., atol=1e-10)):
                        raise ValueError("blade_state_discontinuity_between_steps")
        return packed

    def update(self, before_segments, after_segments, t0, t1):
        if self.failure:
            return []
        pending = []
        try:
            packed = self._check_frames(before_segments, after_segments, t0, t1)
            for owner, center in self.centers.items():
                before, after = tuple(before_segments[owner]), tuple(after_segments[owner])
                starts, ends, new_starts, new_ends, radii = packed[owner]
                delta_start, delta_end = new_starts-starts, new_ends-ends
                bound = _motion_bound_arrays(starts, ends, new_starts, new_ends, radii)
                cache = {}

                def sample(fraction):
                    if fraction not in cache:
                        sample_starts = starts+fraction*delta_start
                        sample_ends = ends+fraction*delta_end
                        distances, points = _point_cylinder_distances(center, sample_starts, sample_ends, radii)
                        selected = int(np.argmin(distances))
                        section = BladeSection(sample_starts[selected], sample_ends[selected],
                                               radii[selected], before[selected].name)
                        cache[fraction] = (float(distances[selected]-self.target_radii[owner]),
                                           points[selected], section)
                        self.minimum_clearance_m[owner] = min(self.minimum_clearance_m[owner], cache[fraction][0])
                    return cache[fraction][0]

                initial_gap = sample(0.)
                if self.armed[owner] is None:
                    self.armed[owner] = bool(initial_gap > _DISTANCE_EPS_M)
                    if not self.armed[owner]:
                        self.initial_overlap_events.append({"agent": owner, "time_s": float(t0),
                                                            "clearance_m": float(initial_gap)})
                cursor, budget = 0., [0]
                for _ in range(32):
                    # Armed: find entry. Unarmed: find a complete exit before
                    # allowing another entry, including within the same step.
                    sign = 1. if self.armed[owner] else -1.
                    crossing = _first_strictly_negative(lambda f: sign*sample(f), cursor, 1., bound,
                                                         evaluation_budget=budget)
                    if crossing is None:
                        break
                    fraction, bracket = crossing
                    if self.armed[owner]:
                        clearance, closest, section = cache[fraction]
                        pending.append({"agent": owner, "time_s": float(t0+fraction*(t1-t0)),
                            "step_t0_s": float(t0), "step_t1_s": float(t1),
                            "fraction": float(fraction), "entry_fraction_bracket": list(map(float, bracket)),
                            "blade_section": section.name, "blade_radius_m": section.radius_m,
                            "blade_closest_point_world_m": closest.tolist(), "target_center_world_m": center.tolist(),
                            "target_radius_m": self.target_radii[owner], "clearance_m": float(clearance),
                            "method": "finite_cylinder_linear_endpoints_certified_interior_witness",
                            "double_touch": False, "decides_round": False})
                    self.armed[owner] = not self.armed[owner]
                    cursor = fraction
                    if cursor >= 1.:
                        break
                else:
                    raise _UnresolvedSweep("too_many_entry_exit_transitions_in_one_step")
            self._last = {k: tuple(v) for k, v in after_segments.items()}
            self._last_time = float(t1)
        except (ValueError, IndexError, _UnresolvedSweep) as error:
            self.invalidate(str(error), t1, {"step_t0_s": float(t0) if np.isfinite(t0) else None,
                                            "step_t1_s": float(t1) if np.isfinite(t1) else None})
            # Candidates computed before a fault remain diagnostic only.
            for event in pending:
                event["step_invalid"] = True
            self.all_entries.extend(pending)
            return []
        pending.sort(key=lambda event: (event["time_s"], event["agent"]))
        if self._outcome["status"] == "ongoing" and pending:
            first_by_owner = {}
            for event in pending:
                first_by_owner.setdefault(event["agent"], event)
            double = len(first_by_owner) == 2
            for event in first_by_owner.values():
                event.update(double_touch=double, decides_round=True)
                self.events.append(event)
            winner = None if double else next(iter(first_by_owner))
            self._outcome.update(status="double_touch" if double else "win", winner=winner,
                                 points={k: int(k == winner) for k in self.centers},
                                 time_s=min(event["time_s"] for event in first_by_owner.values()))
        self.all_entries.extend(pending)
        return pending
