"""Offline contract for a candidate three-region fencing task.

This module does not detect collisions, certify a defence, run a controller, or
replace the live scorer. A native-physics adapter must supply *independent*
finite-blade entry events for every region, plus already-adjudicated fouls.
Filtering the first entry into the union is not a valid adapter for the targeted
mode: an inactive entry must not hide a subsequent active-region entry.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Literal, Mapping


REGIONS = ("left", "right", "upper")
SIDES = ("a1", "a2")
Side = Literal["a1", "a2"]
Mode = Literal["unrestricted", "region_targeted"]
CONTRACT_VERSION = "triangle_task_v1_offline"


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be a finite number")
    return converted


def _names(values: Any, allowed: tuple[str, ...], name: str) -> frozenset[str]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a collection, not a string")
    try:
        result = frozenset(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a collection of names") from exc
    if not result.issubset(allowed):
        raise ValueError(f"{name} contains an unknown name")
    return result


@dataclass(frozen=True)
class TargetRegion:
    """A finite spherical region in the defender-local target frame (metres)."""

    name: str
    center: tuple[float, float, float]
    radius: float

    def __post_init__(self) -> None:
        if self.name not in REGIONS:
            raise ValueError(f"unknown region {self.name!r}")
        if isinstance(self.center, (str, bytes)):
            raise ValueError("center must have three finite coordinates")
        try:
            center = tuple(self.center)
        except TypeError as exc:
            raise ValueError("center must have three finite coordinates") from exc
        if len(center) != 3:
            raise ValueError("center must have three finite coordinates")
        center = tuple(_finite(v, "center coordinate") for v in center)
        radius = _finite(self.radius, "radius")
        if radius <= 0:
            raise ValueError("radius must be positive")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "radius", radius)


@dataclass(frozen=True)
class TriangleGeometry:
    """Candidate geometry only; validity here does not establish reachability."""

    regions: tuple[TargetRegion, ...]
    base_distance: float

    def __post_init__(self) -> None:
        try:
            regions = tuple(self.regions)
        except TypeError as exc:
            raise ValueError("regions must contain exactly three TargetRegions") from exc
        object.__setattr__(self, "regions", regions)
        object.__setattr__(self, "base_distance", _finite(self.base_distance, "base_distance"))
        self.validate()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TriangleGeometry":
        """Read ``base_distance`` and ``regions[name] = {center, radius}``."""
        if not isinstance(value, Mapping) or set(value) != {"base_distance", "regions"}:
            raise ValueError("geometry needs exactly base_distance and regions")
        regions = value["regions"]
        if not isinstance(regions, Mapping) or set(regions) != set(REGIONS):
            raise ValueError("geometry needs exactly left, right, upper regions")
        parsed = []
        for name in REGIONS:
            region = regions[name]
            if not isinstance(region, Mapping) or set(region) != {"center", "radius"}:
                raise ValueError(f"region {name} needs exactly center and radius")
            parsed.append(TargetRegion(name, region["center"], region["radius"]))
        return cls(tuple(parsed), value["base_distance"])

    def validate(self) -> None:
        if self.base_distance <= 0:
            raise ValueError("base_distance must be positive")
        if len(self.regions) != 3 or any(not isinstance(r, TargetRegion) for r in self.regions):
            raise ValueError("regions must contain exactly three TargetRegions")
        if {r.name for r in self.regions} != set(REGIONS):
            raise ValueError("regions must contain left, right, upper exactly once")
        left, right, upper = (self.region(name) for name in REGIONS)
        close = lambda a, b: math.isclose(a, b, rel_tol=0.0, abs_tol=1e-9)
        if not (close(left.center[0], right.center[0]) and close(left.center[0], upper.center[0])):
            raise ValueError("all target centers must share one local x plane")
        if not (left.center[1] > 0 and right.center[1] < 0 and close(left.center[1], -right.center[1])):
            raise ValueError("left/right centers must have opposite symmetric local y")
        if not close(left.center[2], right.center[2]):
            raise ValueError("left/right centers must have equal height")
        if not close(upper.center[1], 0.0) or upper.center[2] <= left.center[2]:
            raise ValueError("upper center must be centered and above both side centers")
        for i, a in enumerate(self.regions):
            for b in self.regions[i + 1:]:
                if math.dist(a.center, b.center) <= a.radius + b.radius:
                    raise ValueError(f"regions {a.name} and {b.name} must be strictly non-overlapping")

    def region(self, name: str) -> TargetRegion:
        for region in self.regions:
            if region.name == name:
                return region
        raise ValueError(f"unknown region {name!r}")

    def mirror_point(self, point: tuple[float, float, float]) -> tuple[float, float, float]:
        """The rigid 180-degree slot transform, which is its own inverse."""
        try:
            coords = tuple(point)
        except TypeError as exc:
            raise ValueError("point must have three coordinates") from exc
        if len(coords) != 3:
            raise ValueError("point must have three coordinates")
        x, y, z = (_finite(v, "point coordinate") for v in coords)
        return self.base_distance - x, -y, z

    def world_center(self, name: str, defender: Side) -> tuple[float, float, float]:
        if defender not in SIDES:
            raise ValueError("defender must be a1 or a2")
        center = self.region(name).center
        return center if defender == "a1" else self.mirror_point(center)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "base_distance": self.base_distance,
            "regions": {name: {"center": list(self.region(name).center), "radius": self.region(name).radius}
                        for name in REGIONS},
        }


class BalancedRegionScheduler:
    """Independent seeded RNG; each consecutive shuffled block covers all regions."""

    def __init__(self, seed: int = 0):
        self.reset(seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is None:
            if not hasattr(self, "_seed"):
                raise ValueError("seed must be an integer")
            seed = self._seed
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        self._seed = seed
        self._rng = random.Random(seed)
        self._pending: list[str] = []

    def next(self) -> str:
        if not self._pending:
            self._pending = list(REGIONS)
            self._rng.shuffle(self._pending)
        return self._pending.pop()


@dataclass(frozen=True)
class QualifiedFoul:
    """External adjudicator's established responsibility, not a raw contact."""

    responsible: frozenset[Side]
    evidence: str

    def __post_init__(self) -> None:
        responsible = _names(self.responsible, SIDES, "foul responsible")
        if not responsible:
            raise ValueError("qualified foul needs at least one responsible side")
        if not isinstance(self.evidence, str) or not self.evidence.strip():
            raise ValueError("qualified foul needs nonempty evidence")
        object.__setattr__(self, "responsible", responsible)


@dataclass(frozen=True)
class NativeTickEvents:
    """Physics-adapter evidence for exactly one native simulation tick.

    ``entries_a1`` / ``entries_a2`` are independent region-entry events by each
    ATTACKER's valid finite blade, not defender-side names or occupancy flags.
    A later active-region entry must be reported even after an inactive entry.
    Foul responsibility must already have been adjudicated by the common rule.
    """

    tick: int
    entries_a1: frozenset[str] = frozenset()
    entries_a2: frozenset[str] = frozenset()
    fouls: tuple[QualifiedFoul, ...] = ()
    timeout: bool = False
    neutral_contacts: int = 0
    numerical_error: str | None = None
    evidence_error: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.tick, bool) or not isinstance(self.tick, int) or self.tick < 0:
            raise ValueError("tick must be a nonnegative integer")
        object.__setattr__(self, "entries_a1", _names(self.entries_a1, REGIONS, "entries_a1"))
        object.__setattr__(self, "entries_a2", _names(self.entries_a2, REGIONS, "entries_a2"))
        try:
            fouls = tuple(self.fouls)
        except TypeError as exc:
            raise ValueError("fouls must be QualifiedFoul events") from exc
        if any(not isinstance(foul, QualifiedFoul) for foul in fouls):
            raise ValueError("fouls must be QualifiedFoul events")
        object.__setattr__(self, "fouls", fouls)
        if not isinstance(self.timeout, bool):
            raise ValueError("timeout must be boolean")
        if isinstance(self.neutral_contacts, bool) or not isinstance(self.neutral_contacts, int) or self.neutral_contacts < 0:
            raise ValueError("neutral_contacts must be a nonnegative integer")
        for name in ("numerical_error", "evidence_error"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be None or a nonempty error description")


class InvalidTickError(RuntimeError):
    """A round is invalid: never translate numerical/evidence faults into draws."""


@dataclass(frozen=True)
class RoundTransition:
    terminated: bool
    rewards: tuple[float, float]
    outcome: str
    winner: Side | None
    tick: int
    mode: Mode
    active_goal: str | None


class TriangleTask:
    """Offline equal-reward event translator, shared by both policy families.

    ``reset`` starts a round and returns public task identity. ``step`` consumes
    one native tick, with errors > qualified fouls > active hits > timeout.
    Targeted mode is a separate diagnostic/training task, not free-duel proof.
    """

    def __init__(self, geometry: TriangleGeometry, mode: Mode = "unrestricted", seed: int = 0):
        if not isinstance(geometry, TriangleGeometry):
            raise ValueError("geometry must be a validated TriangleGeometry")
        if mode not in ("unrestricted", "region_targeted"):
            raise ValueError("unknown task mode")
        self.geometry = geometry
        self.mode = mode
        self.scheduler = BalancedRegionScheduler(seed)
        self.active_goal: str | None = None
        self._ready = False
        self._last_tick = -1

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.scheduler.reset(seed)
        self.active_goal = self.scheduler.next() if self.mode == "region_targeted" else None
        self._last_tick = -1
        self._ready = True
        return self.public_metadata()

    def public_metadata(self) -> dict[str, Any]:
        """The same public task identity must be available to both agents."""
        return {
            "contract_version": CONTRACT_VERSION,
            "mode": self.mode,
            "active_goal": self.active_goal,
            "goal_public_to_both": True,
            "region_hit_reward": 1.0,
            "geometry": self.geometry.to_mapping(),
            "runtime_activation_allowed": False,
            "admitted_to_training": False,
            "physical_validation": "not_performed_by_this_contract",
            "live_scorer_replaced": False,
        }

    def step(self, events: NativeTickEvents) -> RoundTransition:
        if not self._ready:
            raise RuntimeError("reset required before a new or invalid/terminated round")
        if not isinstance(events, NativeTickEvents):
            raise ValueError("step needs NativeTickEvents")
        if events.tick <= self._last_tick:
            raise ValueError("native ticks must be strictly increasing within a round")
        if events.numerical_error is not None or events.evidence_error is not None:
            self._ready = False
            raise InvalidTickError(events.numerical_error or events.evidence_error)
        self._last_tick = events.tick
        responsible = frozenset(side for foul in events.fouls for side in foul.responsible)
        winner: Side | None = None
        rewards = (0.0, 0.0)
        outcome = "ongoing"
        if len(responsible) == 2:
            rewards, outcome = (-1.0, -1.0), "bilateral_foul"
        elif responsible:
            winner = "a2" if "a1" in responsible else "a1"
            rewards = (1.0, -1.0) if winner == "a1" else (-1.0, 1.0)
            outcome = "unilateral_foul"
        else:
            active = frozenset(REGIONS) if self.mode == "unrestricted" else frozenset((self.active_goal,))
            hit_a1 = bool(events.entries_a1 & active)
            hit_a2 = bool(events.entries_a2 & active)
            if hit_a1 and hit_a2:
                outcome = "double_hit"
            elif hit_a1 or hit_a2:
                winner = "a1" if hit_a1 else "a2"
                rewards = (1.0, -1.0) if hit_a1 else (-1.0, 1.0)
                outcome = "target_hit"
            elif events.timeout:
                outcome = "timeout"
        terminated = outcome != "ongoing"
        if terminated:
            self._ready = False
        return RoundTransition(terminated, rewards, outcome, winner, events.tick, self.mode, self.active_goal)
