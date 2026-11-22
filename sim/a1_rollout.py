"""Decision-only, physical-time A1 rollout records; no learner or simulator.

A Guard choice is a genuine short wait decision. Busy forced0 frames are not
actor decisions. A nonzero choice stays pending until own readiness or round
end; the opponent can still act at every environment decision tick. Rewards
are impulses at their completed physical times, discounted from choice time.
Whole episodes are transactional: evidence errors cannot become draw samples.
"""
from __future__ import annotations

from dataclasses import dataclass
import numbers

import numpy as np


# Candidate discount e-folding time is120 physical seconds, not120ticks.
DEFAULT_GAMMA_PER_SECOND=float(np.exp(-1./120.))


def _finite_number(value, label):
    if isinstance(value,(bool,str)) or not isinstance(value,numbers.Real) or not np.isfinite(value):
        raise ValueError(f"Finite numeric {label} required")
    return float(value)


def _features(value):
    array=np.asarray(value)
    if array.ndim!=1 or not len(array) or array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError("A finite flat numeric observation vector is required")
    result=np.asarray(array,dtype=float).copy();result.flags.writeable=False
    return result


@dataclass(frozen=True)
class DecisionTransition:
    slot: str
    policy_id: str
    start_s: float
    end_s: float
    observation: np.ndarray
    next_observation: np.ndarray
    action_mask: np.ndarray
    action: int
    log_prob: float
    value: float
    discounted_reward: float
    discount: float
    episode_end: bool

    @property
    def duration_s(self):return self.end_s-self.start_s


class DecisionEpisode:
    """Stage a complete valid self-play episode for one or both learner slots.

    Frozen opponent transitions need not be collected. ``policy_ids`` binds
    each collected slot to its actor snapshot for the entire episode. An env
    timeout_draw is a game end here (zero bootstrap), NOT a continuing-task
    collector cut. Initial implementation collects complete rounds only.
    """
    def __init__(self, policy_ids, *, gamma_per_second=DEFAULT_GAMMA_PER_SECOND, start_s=0.):
        if not isinstance(policy_ids,dict) or not policy_ids or not set(policy_ids)<={"a1","a2"} \
                or any(not isinstance(p,str) or not p.strip() for p in policy_ids.values()):
            raise ValueError("Explicit learner slot to frozen policy identity mapping required")
        gamma=_finite_number(gamma_per_second,"gamma per second")
        if not 0.<gamma<=1.:raise ValueError("Physical-time gamma must be in (0,1]")
        start=_finite_number(start_s,"start time")
        if start<0:raise ValueError("Nonnegative start time required")
        self.policy_ids=dict(policy_ids);self.gamma=gamma;self.time_s=start
        self._pending={slot:None for slot in policy_ids}
        self._rows={slot:[] for slot in policy_ids}
        self._ended=False;self._aborted=None;self._dimension=None

    def start(self,slot,observation,action_mask,action,log_prob,value,*,policy_id,time_s):
        if self._ended or self._aborted is not None:raise RuntimeError("Episode is closed")
        if slot not in self.policy_ids or self._pending[slot] is not None:
            raise ValueError("Only an unoccupied learner decision may start")
        now=_finite_number(time_s,"choice time")
        if abs(now-self.time_s)>1e-9 or policy_id!=self.policy_ids[slot]:
            raise ValueError("Choice must use this episode's current time and frozen actor")
        mask=np.asarray(action_mask)
        if mask.shape!=(5,) or mask.dtype.kind!="b" or not mask[0] or not mask[1:].any():
            raise ValueError("An actual ready-state five-action boolean mask is required")
        if isinstance(action,(bool,np.bool_)) or not isinstance(action,numbers.Integral) \
                or action not in range(5) or not mask[action]:
            raise ValueError("Chosen integer action must be enabled")
        logp=_finite_number(log_prob,"log probability");v=_finite_number(value,"value")
        if logp>1e-7:raise ValueError("A log probability cannot be positive")
        features=_features(observation)
        if self._dimension is not None and len(features)!=self._dimension:
            raise ValueError("Observation dimensions cannot change within the rollout")
        self._dimension=len(features)
        mask=mask.copy();mask.flags.writeable=False
        self._pending[slot]=dict(start_s=now,observation=features,action_mask=mask,
            action=int(action),log_prob=logp,value=v,reward=0.)

    def advance(self,*,time_s,rewards,next_observations,ready_by_slot,
                effective_actions,accepted,episode_end=False):
        if self._ended or self._aborted is not None:raise RuntimeError("Episode is closed")
        end=_finite_number(time_s,"completed time")
        if end<=self.time_s or type(episode_end) is not bool:
            raise ValueError("A later completed time and explicit game-end boolean are required")
        slots=set(self.policy_ids)
        if any(not isinstance(mapping,dict) or not slots<=set(mapping) for mapping in
                (rewards,next_observations,ready_by_slot,effective_actions,accepted)):
            raise ValueError("All learner slots require matching completed-step metadata")
        prepared={}
        for slot in slots:
            pending=self._pending[slot]
            if pending is None:raise ValueError("Every ready learner must record its decision before integration")
            reward=_finite_number(rewards[slot],"reward")
            feature=_features(next_observations[slot])
            if len(feature)!=self._dimension:raise ValueError("Observation dimension changed")
            if type(ready_by_slot[slot]) is not bool or type(accepted[slot]) is not bool:
                raise ValueError("Actual ready and accepted flags must be booleans")
            effective=effective_actions[slot]
            first=abs(pending["start_s"]-self.time_s)<=1e-9
            expected=pending["action"] if first else 0
            if isinstance(effective,(bool,np.bool_)) or not isinstance(effective,numbers.Integral) \
                    or effective!=expected or (first and not accepted[slot]):
                raise ValueError("Executed action differs from recorded decision or forced continuation")
            duration=end-pending["start_s"]
            prospective=pending["reward"]+self.gamma**duration*reward
            if not np.isfinite(prospective):
                raise ValueError("Overflowed accumulated reward is not usable rollout data")
            prepared[slot]=(pending,prospective,feature)
        # No partial buffer mutation if any slot/metadata was invalid.
        for slot,(pending,prospective,feature) in prepared.items():
            duration=end-pending["start_s"]
            pending["reward"]=prospective
            if episode_end or ready_by_slot[slot]:
                self._rows[slot].append(DecisionTransition(slot,self.policy_ids[slot],
                    pending["start_s"],end,pending["observation"],feature,pending["action_mask"],
                    pending["action"],pending["log_prob"],pending["value"],
                    pending["reward"],self.gamma**duration,episode_end))
                self._pending[slot]=None
        self.time_s=end;self._ended=episode_end

    def abort(self,reason):
        if not isinstance(reason,str) or not reason.strip():raise ValueError("Explicit invalid-episode reason required")
        self._aborted=reason;self._pending={slot:None for slot in self.policy_ids}
        self._rows={slot:[] for slot in self.policy_ids}

    def commit(self):
        if self._aborted is not None:raise RuntimeError(f"Invalid episode cannot become training samples: {self._aborted}")
        if not self._ended or any(value is not None for value in self._pending.values()):
            raise RuntimeError("Only a complete valid game round can be committed")
        return {slot:tuple(rows) for slot,rows in self._rows.items()}


def physical_time_gae(transitions,*,lambda_per_second=1.):
    """One complete slot trajectory; default1 gives full Monte Carlo returns.

    Optional lambda is per physical SECOND, never per20ms forced-continue.
    Game timeouts bootstrap zero, as do first-touch/foul terminal states.
    """
    rows=tuple(transitions)
    lam=_finite_number(lambda_per_second,"lambda per second")
    if not 0.<lam<=1. or not rows:
        raise ValueError("A complete nonempty episode and lambda in (0,1] are required")
    for i,row in enumerate(rows):
        if not isinstance(row,DecisionTransition):
            raise ValueError("Positive-duration decision transitions required")
        for value,label in ((row.start_s,"start"),(row.end_s,"end"),(row.value,"value"),
                            (row.discounted_reward,"discounted reward"),(row.discount,"discount")):
            _finite_number(value,label)
        if row.start_s<0 or row.duration_s<=0 or not 0.<row.discount<=1. or type(row.episode_end) is not bool:
            raise ValueError("Finite positive durations, probability discounts and boolean game ends required")
        if row.slot!=rows[0].slot or row.policy_id!=rows[0].policy_id:
            raise ValueError("Do not mix learner slots or actor snapshots in one GAE chain")
        if i and (rows[i-1].episode_end or abs(row.start_s-rows[i-1].end_s)>1e-9):
            raise ValueError("Decision chain must be contiguous and contain only one round")
    if not rows[-1].episode_end:
        raise ValueError("A complete episode with zero final bootstrap is required")
    advantage=np.zeros(len(rows));following=0.
    for i in range(len(rows)-1,-1,-1):
        row=rows[i]
        next_value=0. if row.episode_end else rows[i+1].value
        delta=row.discounted_reward+row.discount*next_value-row.value
        following=delta+(0. if row.episode_end else row.discount*lam**row.duration_s*following)
        if not np.isfinite(following):raise ValueError("Overflowed advantage is not usable learner data")
        advantage[i]=following
    returns=advantage+np.array([row.value for row in rows])
    if not np.isfinite(returns).all():raise ValueError("Overflowed returns are not usable learner data")
    return advantage,returns
