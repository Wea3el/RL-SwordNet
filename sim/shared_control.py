"""Symmetric non-cancelling skill clocks and outside-entry strike accounting.

No controller here receives the other agent's action, future clock or score.
Guard is an idle choice; finite Attack/Parry must recover and cool down before
another action can be accepted. A controller changes commands, NEVER qpos.
"""
from __future__ import annotations

import numpy as np

from .manual_duel import swept_sphere_fraction


ACTION_NAMES = {0: "guard", 1: "attack", 2: "parry"}
SCORING_PHASES = frozenset(("thrust", "extended_hold", "retract"))


class SkillController:
    def __init__(self, library):
        self.library = library
        self.action_id = 0
        self.started_s = None
        self.attack_id = 0
        self.events = []
        self.ignored_busy_requests = 0

    def _advance(self, now):
        if not np.isfinite(now) or now < 0:
            raise ValueError("Nonnegative finite own time required")
        if self.action_id and now-self.started_s >= self.library.skills[ACTION_NAMES[self.action_id]].duration_s-1e-9:
            self.action_id, self.started_s = 0, None

    def request(self, action_id, now):
        if isinstance(action_id, bool) or action_id not in ACTION_NAMES:
            raise ValueError("Action must be 0, 1 or 2")
        self._advance(now)
        if self.action_id:
            self.ignored_busy_requests += 1
            return False
        if action_id == 0:
            return True
        self.action_id, self.started_s = int(action_id), float(now)
        if action_id == 1:
            self.attack_id += 1
        self.events.append({"time_s": float(now), "action_id": int(action_id),
                            "name": ACTION_NAMES[action_id], "attack_id": self.attack_id})
        return True

    def state(self, now):
        self._advance(now)
        elapsed = 0. if self.started_s is None else max(0., float(now)-self.started_s)
        phase = "guard" if not self.action_id else self.library.skills[ACTION_NAMES[self.action_id]].phase_at(elapsed)
        return {"action_id": self.action_id, "phase": phase, "elapsed_s": elapsed,
                # 0 is also the valid busy-state continue/no-op. Never expose
                # an empty categorical mask to a future policy.
                "busy": bool(self.action_id), "action_mask": [True, not self.action_id, not self.action_id],
                "attack_id": self.attack_id,
                "strike_active": self.action_id == 1 and phase in SCORING_PHASES}

    def target(self, now):
        self._advance(now)
        if not self.action_id:
            return self.library.guard_q.copy()
        return self.library.skills[ACTION_NAMES[self.action_id]].sample(now-self.started_s)


class EntryScorer:
    """One outside-to-inside swept tip entry per accepted Attack, symmetrically.

    A new attack starting with its tip inside is not armed until it leaves.
    Both entrants in one physics step are a double touch (no winner assigned).
    Inactive accidental entries remain diagnostics, not extra attack points.
    """
    def __init__(self, centers, radius):
        self.centers = {k: np.asarray(v, dtype=float) for k, v in centers.items()}
        if not np.isfinite(radius) or radius <= 0 or any(v.shape != (3,) or not np.isfinite(v).all() for v in self.centers.values()):
            raise ValueError("Finite target centers and positive radius required")
        self.radius = float(radius)
        self.last_attack = {k: 0 for k in centers}
        self.armed = {k: False for k in centers}
        self.scored = set()
        self.events = []
        self.inactive_entries = []

    def update(self, states, before, after, t0, t1):
        pending = []
        for owner, center in self.centers.items():
            state = states[owner]
            start, end = np.asarray(before[owner]), np.asarray(after[owner])
            outside_start = np.linalg.norm(start-center) > self.radius+1e-10
            outside_end = np.linalg.norm(end-center) > self.radius+1e-10
            attack = int(state["attack_id"])
            if attack != self.last_attack[owner]:
                self.last_attack[owner] = attack
                self.armed[owner] = bool(outside_start)
            fraction = swept_sphere_fraction(start, end, center, self.radius) if outside_start else None
            key = owner, attack
            if state["strike_active"] and self.armed[owner] and key not in self.scored and fraction is not None:
                pending.append({"agent": owner, "attack_id": attack, "time_s": float(t0+fraction*(t1-t0)),
                                "phase": state["phase"]})
                self.scored.add(key)
                self.armed[owner] = False
            elif fraction is not None and not state["strike_active"]:
                self.inactive_entries.append({"agent": owner, "time_s": float(t0+fraction*(t1-t0)), "phase": state["phase"]})
            if outside_end and key not in self.scored:
                self.armed[owner] = True
        double = len(pending) == 2
        for event in pending:
            event["double_touch"] = double
        self.events.extend(pending)
        return pending


class ScriptedPolicy:
    """Controlled experiment only: attempts each fixed event once; never retries."""
    def __init__(self, events=()):
        self.events = sorted((float(t), int(a)) for t, a in events)
        if any(t < 0 or not np.isfinite(t) or a not in ACTION_NAMES for t, a in self.events):
            raise ValueError("Finite nonnegative event times and legal actions required")
        self.next = 0

    def decide(self, observation):
        now = observation["time_s"]
        if self.next < len(self.events) and self.events[self.next][0] <= now+1e-9:
            action = self.events[self.next][1]
            self.next += 1
            return action
        return 0


class ConstantPolicy:
    def __init__(self, action_id):
        if action_id not in ACTION_NAMES:
            raise ValueError("Legal action required")
        self.action_id = action_id

    def decide(self, observation):
        return self.action_id


class RandomPolicy:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def decide(self, observation):
        return int(self.rng.integers(3)) if not observation["own_skill"]["busy"] else 0


class ReactivePolicy:
    """Simple baseline using visible current opponent sword geometry/velocity.

    No opponent phase/action/clock/script. Parameters are validation candidates,
    not a proven fencer. Attack wait is independently drawn after becoming idle.
    """
    def __init__(self, seed=0, threat_horizon_s=4., attack_wait_s=(.5, 2.), preparation_height_band_m=(.22, .27)):
        self.rng = np.random.default_rng(seed)
        self.horizon = threat_horizon_s
        self.wait_range = attack_wait_s
        self.preparation_height_band = preparation_height_band_m
        self.attack_after = None
        self.last_reason = "idle"

    def decide(self, observation):
        if observation["own_skill"]["busy"]:
            self.attack_after = None
            return 0
        p = np.asarray(observation["opponent_tip_minus_own_target_local_m"])
        v = np.asarray(observation["opponent_tip_velocity_local_m_s"])
        # A visibly descending weapon approaching the horizontal attack line.
        # This is an explicit, tuned geometric rule, NOT inferred opponent
        # elapsed time. It can be fooled and must be tested against controls.
        low, high = self.preparation_height_band
        preparation_threat = (low <= p[2] <= high and .25 < p[0] < .65
                              and abs(p[1]) < .25 and v[2] < -.03)
        vv = float(v@v)
        ttc = -float(p@v)/vv if vv > 1e-8 else np.inf
        miss = float(np.linalg.norm(p+max(0., min(self.horizon, ttc))*v)) if np.isfinite(ttc) else np.inf
        straight_threat = (abs(p[2]) < .07 and 0 < ttc < self.horizon
                           and miss < .18 and np.linalg.norm(v) > .015)
        if preparation_threat or straight_threat:
            self.last_reason = "visible_approaching_sword"
            self.attack_after = None
            return 2
        if self.attack_after is None:
            self.attack_after = observation["time_s"]+float(self.rng.uniform(*self.wait_range))
        if observation["time_s"] >= self.attack_after:
            self.last_reason = "own_randomized_attack_wait_expired"
            self.attack_after = None
            return 1
        self.last_reason = "wait"
        return 0


class ReactiveCounterPolicy(ReactivePolicy):
    """A fixed baseline candidate: observable defence, then own-clock counter.

    After OWN finite Parry, wait for visible axial weapon withdrawal before
    countering. This requires no opponent action id, timer or future deadline.
    Immediate release was an independently recorded failed candidate: it let
    the original attacker hit during retraction. This rule still needs tests.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.was_parrying = False

    def decide(self, observation):
        own = observation["own_skill"]
        if own["busy"]:
            self.was_parrying = own["action_id"] == 2
        elif self.was_parrying:
            p = np.asarray(observation["opponent_tip_minus_own_target_local_m"])
            v = np.asarray(observation["opponent_tip_velocity_local_m_s"])
            # Axial clearance, not Euclidean distance: a laterally deflected
            # sword can still be pushed back toward the target by its servo.
            withdrawn = p[0] > .21 and (v[0] > .005 or p[2] > .10 and v[2] > .01)
            if withdrawn:
                self.was_parrying = False
                self.attack_after = None
                self.last_reason = "visible_weapon_withdrawn_counterattack"
                return 1
            self.last_reason = "wait_for_visible_weapon_withdrawal"
            return 0
        return super().decide(observation)
