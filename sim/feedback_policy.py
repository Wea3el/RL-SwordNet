"""Small explicit skill graph and observation-only diagnostic policies.

This is not a learned A1 policy, and does not admit any motion to training.
The executor never resets physical state. Nodes are exact joint coordinates;
an action can only start at its node after measured own-state settling.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .joint_constraints import enforce_joint_trajectory, load_joint_constraints, model_joint_ranges_rad
from .motion_spline import C2JointSpline


@dataclass(frozen=True)
class GraphAction:
    name: str
    start_node: str
    end_node: str
    curve: C2JointSpline


class SkillGraphExecutor:
    def __init__(self, nodes, actions, initial_node):
        constraints = load_joint_constraints()
        ranges = model_joint_ranges_rad(constraints)
        self.nodes = {k: np.array(v, dtype=float, copy=True) for k, v in nodes.items()}
        if not self.nodes or initial_node not in self.nodes:
            raise ValueError('Explicit nodes and initial node required')
        for q in self.nodes.values():
            if q.shape != (7,): raise ValueError('Seven coordinates per node required')
            enforce_joint_trajectory(q, constraints)
        self.actions = dict(actions)
        for name, action in self.actions.items():
            if not isinstance(action, GraphAction) or action.name != name:
                raise ValueError('Named graph actions required')
            if action.start_node not in self.nodes or action.end_node not in self.nodes:
                raise ValueError('Unknown node')
            if action.curve.knot_time_s[0] != 0:
                raise ValueError('Action curve must start at zero')
            if not np.array_equal(action.curve.sample(0), self.nodes[action.start_node]) \
                    or not np.array_equal(action.curve.sample(action.curve.duration_s), self.nodes[action.end_node]):
                raise ValueError('Action must preserve exact joint-space node identity')
            if np.any(action.curve.bezier_controls < ranges[:, 0]) or np.any(action.curve.bezier_controls > ranges[:, 1]):
                raise ValueError('Whole reference must respect confirmed limits')
        self.node = initial_node
        self.active = None
        self.started_s = None
        self.ready = False
        self.stable_since_s = None
        self.events = []
        self._trial_consumed = False

    @property
    def initial_q(self):
        return self.nodes[self.node].copy()

    def target(self, now):
        if self.active is None:
            return self.nodes[self.node].copy()
        return self.actions[self.active].curve.sample(float(now)-self.started_s)

    def observe(self, now, q, velocity):
        q, velocity = np.asarray(q), np.asarray(velocity)
        if q.shape != (7,) or velocity.shape != (7,) or not np.isfinite([q, velocity]).all():
            raise ValueError('Finite own-state observation required')
        destination = self.node if self.active is None else self.actions[self.active].end_node
        finished_reference = self.active is None or now >= self.started_s+self.actions[self.active].curve.duration_s-1e-9
        stable = bool(finished_reference and np.max(np.abs(q-self.nodes[destination])) <= .05
                      and np.max(np.abs(velocity)) <= .10)
        if not stable:
            self.stable_since_s = None
            self.ready = False
            return
        if self.stable_since_s is None: self.stable_since_s = float(now)
        self.ready = bool(now-self.stable_since_s >= .1-1e-9)
        if self.active is not None and self.ready:
            self.events.append({'event': 'completed', 'action': self.active, 'time_s': float(now),
                                'node': destination})
            self.active, self.started_s, self.node = None, None, destination

    def available(self):
        return tuple(name for name, a in self.actions.items() if self.ready and self.active is None and a.start_node == self.node)

    def request(self, action, now):
        if action not in self.available():
            raise ValueError('Unavailable action; requests may not bypass own-state readiness')
        self.events.append({'event': 'started', 'action': action, 'time_s': float(now), 'node': self.node})
        self.active, self.started_s = action, float(now)
        self.ready, self.stable_since_s = False, None


class TimedRequests:
    """A predeclared attacker/test script; never called autonomous self-play."""
    def __init__(self, requests):
        self.requests = tuple((int(tick), str(action)) for tick, action in requests)
        if any(t < 0 for t, _ in self.requests) or list(self.requests) != sorted(self.requests):
            raise ValueError('Ordered nonnegative decision ticks required')
        self.tick, self.next = 0, 0
        self.events = []

    def choose(self, observation, available):
        result = None
        if self.next < len(self.requests):
            tick, action = self.requests[self.next]
            if self.tick >= tick and action in available:
                result = action
                self.next += 1
        self.tick += 1
        return result


class ThreatParry:
    """One-shot cue using current observed tip motion, not time/skill labels.

    Noise-free simulation diagnostic only. Thresholds are not a perception
    precision or transfer-to-hardware claim. Own executor availability remains
    authoritative; a threat while busy cannot teleport into a parry.
    """
    def __init__(self, action='parry', displacement_m=.001, approaching_speed_m_s=.005,
                 consecutive_samples=2, require_approach=True):
        if displacement_m <= 0 or approaching_speed_m_s <= 0 or consecutive_samples < 1:
            raise ValueError('Positive cue thresholds and confirmation count required')
        self.action = action
        self.displacement_m = float(displacement_m)
        self.approaching_speed_m_s = float(approaching_speed_m_s)
        self.consecutive_samples = int(consecutive_samples)
        if not isinstance(require_approach, bool): raise ValueError('Explicit direction-cue flag required')
        self.require_approach = require_approach
        self.reference_tip = None
        self.count, self.fired = 0, False
        self.events = []

    def choose(self, observation, available):
        tip = np.asarray(observation['opponent_tip_minus_own_target_local_m'])
        velocity = np.asarray(observation['opponent_tip_velocity_local_m_s'])
        if tip.shape != (3,) or velocity.shape != (3,) or not np.isfinite([tip, velocity]).all():
            raise ValueError('Finite current tip observation required')
        if self.reference_tip is None: self.reference_tip = tip.copy()
        motion = float(np.linalg.norm(tip-self.reference_tip))
        # In observer coordinates an approaching front-side sword moves -X.
        speed_cue = (velocity[0] <= -self.approaching_speed_m_s if self.require_approach
                     else np.linalg.norm(velocity) >= self.approaching_speed_m_s)
        threat = motion >= self.displacement_m and speed_cue
        self.count = self.count+1 if threat else 0
        if not self.fired and self.count >= self.consecutive_samples and self.action in available:
            self.fired = True
            self.events.append({'event': 'observed_threat', 'displacement_m': motion,
                'observed_tip_m': tip.tolist(), 'observed_tip_velocity_m_s': velocity.tolist()})
            return self.action
        return None


class WaitPolicy:
    events = ()

    def choose(self, observation, available):
        return None


class RearmingThreatParry(ThreatParry):
    """Rearm only after observed opponent recovery and own action readiness.

    This is still a conservative movement cue, not intent recognition. A
    different opponent resting pose need not rearm: that limitation is explicit
    rather than using an opponent's hidden action-completed flag.
    """
    def __init__(self, *args, recovery_distance_m=.005, recovery_speed_m_s=.01,
                 recovery_samples=5, **kwargs):
        super().__init__(*args, **kwargs)
        if recovery_distance_m <= 0 or recovery_speed_m_s <= 0 or recovery_samples < 1:
            raise ValueError('Positive measured recovery thresholds required')
        self.recovery_distance_m=float(recovery_distance_m)
        self.recovery_speed_m_s=float(recovery_speed_m_s)
        self.recovery_samples=int(recovery_samples)
        self.recovery_count=0

    def choose(self, observation, available):
        if self.fired:
            tip=np.asarray(observation['opponent_tip_minus_own_target_local_m'])
            velocity=np.asarray(observation['opponent_tip_velocity_local_m_s'])
            recovered=(np.linalg.norm(tip-self.reference_tip)<=self.recovery_distance_m
                       and np.linalg.norm(velocity)<=self.recovery_speed_m_s)
            self.recovery_count=self.recovery_count+1 if recovered else 0
            if self.recovery_count>=self.recovery_samples and self.action in available:
                self.fired=False
                self.count=self.recovery_count=0
                self.events.append({'event':'rearmed_after_observed_recovery'})
        return super().choose(observation,available)
