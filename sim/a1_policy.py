"""Candidate A1 chooser only; no trainer, checkpoint write or bank admission.

Policy features are current public geometry/proprioception and A1's OWN
execution state. Absolute episode time, opposing skill/phase, test schedules
and snapshot IDs are never read. Feature scaling is not a motor-speed limit.
Busy/unavailable continuation is NOT a sampled actor decision. A ready Guard
choice IS a real, learnable 20ms wait. Duration-aware returns belong to the
collector, not to this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numbers

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


POLICY_VERSION = "a1_candidate_masked_mlp_v1"
FEATURE_VERSION = "a1_public_own_execution_30_v1"
ACTION_NAMES = ("guard", "stab", "slash_left", "parry_left", "parry_right")
FEATURE_ORDER = (
    *(f"own_q_J{i}_over_pi" for i in range(7)),
    *(f"own_qvel_J{i}_over_4rad_s" for i in range(7)),
    *(f"opponent_tip_minus_own_target_local_{axis}_over_1m" for axis in "xyz"),
    *(f"opponent_grip_minus_own_target_local_{axis}_over_1m" for axis in "xyz"),
    *(f"opponent_tip_velocity_local_{axis}_over_1m_s" for axis in "xyz"),
    "own_busy", *(f"own_action_{name}" for name in ACTION_NAMES),
    "own_elapsed_over_own_skill_duration")
FEATURE_SCALES = {
    "own_q_rad": math.pi, "own_qvel_rad_s": 4.,
    "opponent_tip_minus_own_target_local_m": 1.,
    "opponent_grip_minus_own_target_local_m": 1.,
    "opponent_tip_velocity_local_m_s": 1.}
OWN_BUSY_FEATURE_INDEX = FEATURE_ORDER.index("own_busy")


def _positive(value, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real)
            or not np.isfinite(value) or value <= 0):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _vector(value, size, name):
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric vector") from exc
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite vector of length {size}")
    return array


@dataclass(frozen=True)
class A1FeatureEncoder:
    """Versioned fixed-order encoding; durations come from the OWN finite bank."""
    durations_s: tuple[float, ...]

    def __post_init__(self):
        if len(self.durations_s) != 5:
            raise ValueError("The five current A1 skill durations are required")
        object.__setattr__(self, "durations_s", tuple(
            _positive(value, "Own skill duration") for value in self.durations_s))

    @classmethod
    def from_library(cls, library):
        paths = [library.skills[name] for name in ACTION_NAMES]
        if any(path.name != name or isinstance(path.action_id, (bool, np.bool_))
               or not isinstance(path.action_id, numbers.Integral)
               or path.action_id != i for i, (name, path) in enumerate(zip(ACTION_NAMES, paths))):
            raise ValueError("Current A1 names/IDs must match the public five-action schema")
        return cls(tuple(path.duration_s for path in paths))

    def metadata(self):
        return {"feature_version": FEATURE_VERSION, "feature_order": list(FEATURE_ORDER),
            "feature_scales_divisors": dict(FEATURE_SCALES), "feature_dimension": len(FEATURE_ORDER),
            "action_names": list(ACTION_NAMES), "own_skill_durations_s": list(self.durations_s),
            "opponent_hidden_state_or_absolute_time_read": False,
            "own_elapsed_is_own_executor_progress_not_opponent_phase": True,
            "velocity_scaling_is_not_a_speed_cap": True, "numeric_features_clipped": False}

    def encode(self, observation):
        own = observation["own_skill"]
        busy = own["busy"]
        if not isinstance(busy, (bool, np.bool_)):
            raise ValueError("own busy must be boolean")
        action = own["action_id"]
        if (isinstance(action, (bool, np.bool_)) or not isinstance(action, numbers.Integral)
                or not 0 <= action < 5):
            raise ValueError("Own current A1 action ID must be integer0--4")
        elapsed = own["elapsed_s"]
        if (isinstance(elapsed, (bool, np.bool_)) or not isinstance(elapsed, numbers.Real)
                or not np.isfinite(elapsed) or elapsed < 0):
            raise ValueError("Own execution elapsed time must be finite and nonnegative")
        mask = np.asarray(own["action_mask"])
        if mask.shape != (5,) or mask.dtype.kind != "b" or not mask[0]:
            raise ValueError("Five boolean own action-mask entries with Guard available required")
        one_hot = np.zeros(5)
        one_hot[int(action)] = 1.
        features = np.concatenate((
            _vector(observation["own_q_rad"], 7, "Own joints") / math.pi,
            _vector(observation["own_qvel_rad_s"], 7, "Own velocities") / 4.,
            _vector(observation["opponent_tip_minus_own_target_local_m"], 3, "Current public tip"),
            _vector(observation["opponent_grip_minus_own_target_local_m"], 3, "Current public grip"),
            _vector(observation["opponent_tip_velocity_local_m_s"], 3, "Current public tip velocity"),
            [float(busy)], one_hot,
            [float(elapsed)/self.durations_s[int(action)] if action else 0.])).astype(np.float32)
        if features.shape != (30,) or not np.isfinite(features).all():
            raise ValueError("Public features overflowed float32")
        choice = bool(not busy and mask[1:].any())
        effective_mask = mask.astype(bool, copy=True)
        if not choice:
            effective_mask[:] = False
            effective_mask[0] = True
        return features, effective_mask, choice


@dataclass(frozen=True)
class PolicyDecision:
    action: int
    actor_choice: bool
    log_prob: float | None
    entropy: float | None
    value: float
    features: np.ndarray
    action_mask: np.ndarray
    probabilities: np.ndarray


class A1Policy(nn.Module):
    """Small Torch actor/value with an explicit physical-time Guard prior.

    Default h=.25/s gives p(start in a ready20ms tick)=1-exp(-h*.02).
    This is a learnable initialization, not a forced wait, phase oracle or
    action-mask restriction. All four nonGuard skills initially share mass.
    Use stochastic ``act`` for exploration; greedy untrained inference waits.
    """
    def __init__(self, encoder, *, decision_period_s=.02, guard_hazard_per_s=.25,
                 hidden_sizes=(64, 64)):
        super().__init__()
        if not isinstance(encoder, A1FeatureEncoder):
            raise ValueError("An explicit versioned A1FeatureEncoder is required")
        self.encoder = encoder
        self.decision_period_s = _positive(decision_period_s, "Decision period")
        self.guard_hazard_per_s = _positive(guard_hazard_per_s, "Guard departure hazard")
        sizes = tuple(hidden_sizes)
        if not sizes or any(isinstance(n, bool) or not isinstance(n, numbers.Integral) or n <= 0 for n in sizes):
            raise ValueError("Positive integer hidden layer sizes required")
        self.hidden_sizes = tuple(map(int, sizes))
        layers, width = [], len(FEATURE_ORDER)
        for size in self.hidden_sizes:
            layers.extend((nn.Linear(width, size), nn.Tanh()))
            width = size
        self.body = nn.Sequential(*layers)
        self.actor = nn.Linear(width, 5)
        self.critic = nn.Linear(width, 1)
        p_start = -math.expm1(-self.guard_hazard_per_s*self.decision_period_s)
        guard = math.exp(-self.guard_hazard_per_s*self.decision_period_s)
        prior = np.asarray([guard]+[p_start/4.]*4)
        if not np.isfinite(prior).all() or np.any(prior <= 0):
            raise ValueError("Hazard/period underflowed the five positive initial probabilities")
        self.initial_probabilities = tuple(map(float, prior))
        nn.init.zeros_(self.actor.weight)
        with torch.no_grad():
            self.actor.bias.copy_(torch.tensor(np.log(prior), dtype=self.actor.bias.dtype))

    def metadata(self):
        return {"policy_version": POLICY_VERSION, "encoder": self.encoder.metadata(),
            "hidden_sizes": list(self.hidden_sizes), "activation": "Tanh",
            "decision_period_s": self.decision_period_s,
            "initial_guard_departure_hazard_per_s": self.guard_hazard_per_s,
            "initial_unmasked_probabilities": list(self.initial_probabilities),
            "mask_contract": "Renormalize over available own actions; busy/unavailable is forced0, actor_choice=False, log_prob=None",
            "admitted_to_training": False, "module_runs_training_or_writes_checkpoints": False,
            "torch_version": torch.__version__}

    def _features(self, features):
        value = torch.as_tensor(features, dtype=self.actor.weight.dtype, device=self.actor.weight.device)
        if value.ndim not in (1, 2) or value.shape[-1] != len(FEATURE_ORDER) or not torch.isfinite(value).all():
            raise ValueError("Finite fixed-order features with shape(30,) or(batch,30) required")
        return value

    def forward(self, features):
        encoded = self.body(self._features(features))
        logits, value = self.actor(encoded), self.critic(encoded).squeeze(-1)
        if not torch.isfinite(logits).all() or not torch.isfinite(value).all():
            raise ValueError("Nonfinite policy logits/value are not usable rollout data")
        return logits, value

    def evaluate_actions(self, features, actions, action_masks):
        """Differentiable PPO inputs, ONLY for actual actor-choice transitions."""
        x = self._features(features)
        if x.ndim != 2 or not len(x):
            raise ValueError("A nonempty batch of actual decision states is required")
        masks = torch.as_tensor(action_masks, device=x.device)
        ids = torch.as_tensor(actions, device=x.device)
        if masks.dtype != torch.bool or masks.shape != (len(x), 5) or not masks[:, 0].all() \
                or not masks[:, 1:].any(dim=1).all() or (x[:, OWN_BUSY_FEATURE_INDEX] != 0).any():
            raise ValueError("Busy/forced continuation must not enter the actor-choice batch")
        if ids.shape != (len(x),) or ids.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64) \
                or (ids < 0).any() or (ids >= 5).any():
            raise ValueError("One integer current A1 action per decision state required")
        ids = ids.long()
        if not masks.gather(1, ids[:, None]).all():
            raise ValueError("Recorded actor action was unavailable under its saved mask")
        logits, values = self.forward(x)
        distribution = Categorical(logits=logits.masked_fill(~masks, -torch.inf))
        return distribution.log_prob(ids), distribution.entropy(), values

    @torch.no_grad()
    def act(self, observation, *, deterministic=False, generator=None):
        features, mask, choice = self.encoder.encode(observation)
        logits, value = self.forward(features)
        if not choice:
            return PolicyDecision(0, False, None, None, float(value), features.copy(), mask.copy(),
                                  np.array([1., 0., 0., 0., 0.], dtype=np.float32))
        valid = torch.as_tensor(mask, device=logits.device)
        distribution = Categorical(logits=logits.masked_fill(~valid, -torch.inf))
        action = (torch.argmax(distribution.probs) if deterministic else
                  torch.multinomial(distribution.probs, 1, generator=generator).squeeze(0))
        return PolicyDecision(int(action), True, float(distribution.log_prob(action)),
            float(distribution.entropy()), float(value), features.copy(), mask.copy(),
            distribution.probs.cpu().numpy().copy())
