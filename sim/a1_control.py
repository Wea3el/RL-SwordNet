"""A1 finite-skill execution; no tactics, opponent clock, scoring or learning.

Instances are policy-family A1 executors in EITHER physical robot slot. A path
is never cancelled/restarted while busy. Its reference returns continuously to
the common Guard; release then requires measured own joint position/velocity
to settle there. No method can write simulator qpos or supply external forces.
This is deliberately usable for candidate validation, NOT training admission.
"""
from __future__ import annotations

import numbers

import numpy as np


class A1Controller:
    def __init__(self, library, *, position_tolerance_rad=.05,
                 velocity_tolerance_rad_s=.10, settle_dwell_s=.10,
                 settle_timeout_s=3., maximum_feedback_age_s=.021):
        self.library = library
        self.guard_q = np.asarray(library.guard_q, dtype=float).copy()
        if self.guard_q.shape != (7,) or not np.isfinite(self.guard_q).all():
            raise ValueError("Seven finite common Guard joints required")
        values = (position_tolerance_rad, velocity_tolerance_rad_s,
                  settle_dwell_s, settle_timeout_s, maximum_feedback_age_s)
        if (any(isinstance(v, (bool, str)) for v in values)
                or not np.isfinite(values).all() or min(values) <= 0
                or settle_timeout_s < settle_dwell_s):
            raise ValueError("Positive finite entry/exit feedback limits required")
        self.position_tolerance_rad = float(position_tolerance_rad)
        self.velocity_tolerance_rad_s = float(velocity_tolerance_rad_s)
        self.settle_dwell_s = float(settle_dwell_s)
        self.settle_timeout_s = float(settle_timeout_s)
        self.maximum_feedback_age_s = float(maximum_feedback_age_s)
        self.action_names = {}
        for name, path in library.skills.items():
            if (not isinstance(path.action_id, numbers.Integral)
                    or isinstance(path.action_id, (bool, np.bool_))
                    or int(path.action_id) in self.action_names or path.name != name):
                raise ValueError("Unique integer skill IDs and matching names required")
            q, t = np.asarray(path.qpos), np.asarray(path.time_s)
            if (q.ndim != 2 or q.shape[1] != 7 or len(q) < 2
                    or t.shape != (len(q),) or not np.isfinite(q).all()
                    or not np.isfinite(t).all() or t[0] != 0.
                    or not np.all(np.diff(t) > 0)
                    or not np.array_equal(q[0], self.guard_q)
                    or not np.array_equal(q[-1], self.guard_q)
                    or not np.array_equal(path.sample(0.), self.guard_q)
                    or not np.array_equal(path.sample(float(t[-1])), self.guard_q)):
                raise ValueError("Every finite skill must start/end at EXACT common Guard")
            self.action_names[int(path.action_id)] = name
        if (len(self.action_names) < 2 or self.action_names.get(0) != "guard"
                or set(self.action_names) != set(range(len(self.action_names)))):
            raise ValueError("Contiguous IDs with Guard=0 required")
        self.action_id, self.started_s, self.attack_id = 0, None, 0
        self.failure = None
        self.events, self.completions = [], []
        self.ignored_busy_requests = 0
        self.rejected_entry_requests = 0
        self._last_time = -np.inf
        self._feedback_time = None
        self._q, self._v = None, None
        self._settled_since = None

    def _time(self, now):
        if (isinstance(now, (bool, str)) or not np.isscalar(now)
                or not np.isfinite(now) or now < 0 or now < self._last_time-1e-9):
            raise ValueError("Finite nonnegative monotonic own time required")
        self._last_time = max(self._last_time, float(now))
        return float(now)

    def _guard_errors(self):
        if self._q is None:
            return None, None
        return float(np.max(np.abs(self._q-self.guard_q))), float(np.max(np.abs(self._v)))

    def _entry_ready(self, now):
        if self.failure or self._feedback_time is None:
            return False
        fresh = -1e-9 <= now-self._feedback_time <= self.maximum_feedback_age_s+1e-9
        position, velocity = self._guard_errors()
        return bool(fresh and position <= self.position_tolerance_rad
                    and velocity <= self.velocity_tolerance_rad_s)

    def _advance(self, now):
        if self.action_id and not self.failure:
            end = self.started_s+self.library.skills[self.action_names[self.action_id]].duration_s
            # Completion can ONLY be established by fresh measured feedback,
            # never by a future call to target/state or the source clock alone.
            if now >= end+self.settle_timeout_s-1e-9:
                self.failure = "a1_guard_settle_timeout"

    def observe_own_state(self, q, qvel, now):
        now = self._time(now)
        q, qvel = np.asarray(q, dtype=float), np.asarray(qvel, dtype=float)
        if q.shape != (7,) or qvel.shape != (7,) or not np.isfinite(q).all() or not np.isfinite(qvel).all():
            raise ValueError("Finite measured own seven-joint position/velocity required")
        if (self._feedback_time is not None
                and now-self._feedback_time > self.maximum_feedback_age_s+1e-9):
            # Two good endpoint measurements cannot establish that an
            # unobserved interval stayed inside the exit contract.
            self._settled_since = None
        # A late measurement cannot retroactively complete an expired action.
        self._advance(now)
        self._q, self._v, self._feedback_time = q.copy(), qvel.copy(), now
        if self.action_id and not self.failure:
            path = self.library.skills[self.action_names[self.action_id]]
            end = self.started_s+path.duration_s
            if now >= end-1e-9:
                if self._entry_ready(now):
                    if self._settled_since is None:
                        self._settled_since = now
                    if now-self._settled_since >= self.settle_dwell_s-1e-9:
                        position, velocity = self._guard_errors()
                        self.completions.append({"time_s": now, "action_id": self.action_id,
                            "name": path.name, "started_s": self.started_s,
                            "nominal_end_s": end, "settle_duration_s": now-end,
                            "guard_error_rad": position, "speed_rad_s": velocity})
                        self.action_id, self.started_s = 0, None
                        self._settled_since = None
                else:
                    self._settled_since = None
        self._advance(now)

    def request(self, action_id, now):
        if (isinstance(action_id, (bool, np.bool_))
                or not isinstance(action_id, numbers.Integral)
                or int(action_id) not in self.action_names):
            raise ValueError("An integer ID from this A1 library required")
        now = self._time(now)
        self._advance(now)
        if self.failure:
            return False
        # Guard is also the busy-state continue/no-op, not a cancellation.
        if action_id == 0:
            return True
        if self.action_id:
            self.ignored_busy_requests += 1
            return False
        if not self._entry_ready(now):
            self.rejected_entry_requests += 1
            return False
        self.action_id, self.started_s = int(action_id), now
        self._settled_since = None
        self.attack_id += 1  # execution serial only; NEVER a scoring gate
        position, velocity = self._guard_errors()
        self.events.append({"time_s": now, "action_id": self.action_id,
            "name": self.action_names[self.action_id], "execution_id": self.attack_id,
            "entry_guard_error_rad": position, "entry_speed_rad_s": velocity})
        return True

    def state(self, now):
        now = self._time(now)
        self._advance(now)
        elapsed = 0. if self.started_s is None else max(0., now-self.started_s)
        path = self.library.skills[self.action_names[self.action_id]]
        settling = bool(self.action_id and elapsed >= path.duration_s-1e-9)
        ready = not self.action_id and self._entry_ready(now)
        return {"policy_family": "A1", "action_id": self.action_id,
                "phase": "fault" if self.failure else "return_settling" if settling
                         else path.phase_at(elapsed) if self.action_id else "guard",
                "elapsed_s": elapsed, "busy": bool(self.action_id),
                "action_mask": [True]+[bool(ready)]*(len(self.action_names)-1),
                "attack_id": self.attack_id,
                "strike_active": bool(self.action_id),  # legacy distance diagnostic ONLY
                "settling": settling, "failure": self.failure}

    def target(self, now):
        now = self._time(now)
        self._advance(now)
        if not self.action_id:
            return self.guard_q.copy()
        path = self.library.skills[self.action_names[self.action_id]]
        return path.sample(min(now-self.started_s, path.duration_s)).copy()

    def diagnostics(self):
        return {"policy_family": "A1", "robot_slot_independent": True,
            "action_names": dict(self.action_names), "failure": self.failure,
            "entry_exit_contract": {"position_tolerance_rad": self.position_tolerance_rad,
                "velocity_tolerance_rad_s": self.velocity_tolerance_rad_s,
                "settle_dwell_s": self.settle_dwell_s,
                "settle_timeout_s": self.settle_timeout_s,
                "maximum_feedback_age_s": self.maximum_feedback_age_s},
            "ignored_busy_requests": self.ignored_busy_requests,
            "rejected_entry_requests": self.rejected_entry_requests,
            "completions": list(self.completions),
            "feedback_time_s": self._feedback_time,
            "last_guard_error_rad": self._guard_errors()[0],
            "last_speed_rad_s": self._guard_errors()[1],
            "opponent_inputs": False, "qpos_resets_after_initialization": 0,
            "admitted_to_training": False,
            "scope": "Own-feedback finite command executor; not policy learning or skill-effectiveness certification"}


class A1ScriptedPolicy:
    """One-shot requests for controlled validation, NOT A1's learned strategy."""
    def __init__(self, library, events=()):
        valid = {path.action_id for path in library.skills.values()}
        checked = []
        for time, action in events:
            if (isinstance(time, (bool, str)) or not np.isfinite(time) or time < 0
                    or isinstance(action, (bool, np.bool_))
                    or not isinstance(action, numbers.Integral) or action not in valid):
                raise ValueError("Finite own scheduled times and integer library actions required")
            checked.append((float(time), int(action)))
        self.events, self.next = sorted(checked), 0

    def decide(self, observation):
        now = observation["time_s"]
        if self.next < len(self.events) and self.events[self.next][0] <= now+1e-9:
            action = self.events[self.next][1]
            self.next += 1
            return action
        return 0


class A1SequencePolicy:
    """Diagnostic sequence of independent requests at measured ready states.

    This constructs reproducible transition tests, NOT a whole-combat sequence
    or self-play policy. It sees the same own mask as a future learned chooser.
    """
    def __init__(self, library, actions, start_s=.75, gap_s=.1):
        valid = {path.action_id for path in library.skills.values()}-{0}
        self.actions = tuple(actions)
        if any(isinstance(a, (bool, np.bool_)) or not isinstance(a, numbers.Integral) or a not in valid for a in self.actions):
            raise ValueError("Non-Guard integer library actions required")
        if not np.isfinite([start_s, gap_s]).all() or min(start_s, gap_s) < 0:
            raise ValueError("Finite nonnegative own waiting times required")
        self.start_s, self.gap_s = float(start_s), float(gap_s)
        self.next, self._last_request, self._was_busy = 0, None, False
        self._ready_at = self.start_s

    def decide(self, observation):
        now, own = float(observation["time_s"]), observation["own_skill"]
        if own["busy"]:
            self._was_busy = True
            return 0
        if self._was_busy:
            self._ready_at, self._was_busy = now+self.gap_s, False
        if self.next >= len(self.actions) or now < self._ready_at-1e-9:
            return 0
        action = int(self.actions[self.next])
        if not own["action_mask"][action]:
            return 0
        self.next += 1
        self._last_request = now
        return action
