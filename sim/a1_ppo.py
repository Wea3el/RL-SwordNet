"""One explicitly called decision-only PPO optimizer smoke, never a trainer.

Input is a batch of COMMITTED complete learner trajectories. Capture
``ppo_batch_identity`` BEFORE collection, not immediately before updating to
hide stale data. The collector remains responsible for environment/learner
role provenance; compatibility metadata can be checked explicitly here too.
No simulator, rollout loop, opponent pool or checkpoint is started/written.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import numbers

import numpy as np
import torch

from .a1_policy import A1Policy, FEATURE_VERSION, OWN_BUSY_FEATURE_INDEX
from .a1_rollout import DecisionTransition, physical_time_gae


UPDATE_VERSION = "a1_single_epoch_decision_PPO_candidate_v1"
REPLAY_ATOL = 2e-6  # Float32 single-row versus batched MLP roundoff, not stale-policy slack.


def metadata_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                       allow_nan=False).encode()).hexdigest()


def weights_hash(policy):
    """Canonical sorted name+torch dtype+shape+CPU bytes, including critic."""
    digest = hashlib.sha256()
    for name, tensor in sorted(policy.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite policy weights cannot identify usable on-policy data")
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)],
                                 separators=(",", ":")).encode())
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def ppo_batch_identity(policy, policy_id, compatibility_metadata=None):
    if not isinstance(policy_id, str) or not policy_id.strip():
        raise ValueError("Explicit frozen learner policy ID required")
    result = {"policy_id": policy_id, "weights_sha256": weights_hash(policy),
        "policy_metadata_sha256": metadata_hash(policy.metadata()), "feature_version": FEATURE_VERSION}
    if compatibility_metadata is not None:
        result["compatibility_metadata_sha256"] = metadata_hash(compatibility_metadata)
    return result


def _number(value, label, *, positive=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real)
            or not np.isfinite(value) or (positive and value <= 0)):
        raise ValueError(f"Finite {'positive ' if positive else ''}{label} required")
    return float(value)


def _step_count(optimizer):
    steps = []
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                raise ValueError("Nonfinite optimizer state is not usable")
        step = state.get("step", 0)
        step = float(step.item() if isinstance(step, torch.Tensor) else step)
        if not math.isfinite(step) or step < 0 or step != int(step):
            raise ValueError("Optimizer step counter must be a finite nonnegative integer")
        steps.append(int(step))
    return max(steps, default=0)


def _prepare(policy, episodes, expected_policy_id, batch_identity, compatibility):
    if not isinstance(policy, A1Policy) or not all(p.requires_grad for p in policy.parameters()):
        raise ValueError("An unfrozen current A1 learner, not an opponent snapshot, is required")
    expected = ppo_batch_identity(policy, expected_policy_id, compatibility)
    if not isinstance(batch_identity, dict) or any(batch_identity.get(k) != v for k, v in expected.items()):
        raise ValueError("Collected policy/weights/feature/compatibility identity is stale or mixed")
    if compatibility is None and "compatibility_metadata_sha256" in batch_identity:
        raise ValueError("Explicit expected compatibility metadata required for this bound batch")
    trajectories = tuple(tuple(episode) for episode in episodes)
    if not trajectories or any(not episode for episode in trajectories):
        raise ValueError("Nonempty complete learner episodes required")
    rows, advantages, targets, rates = [], [], [], []
    for episode in trajectories:
        if not isinstance(episode[0], DecisionTransition) or abs(episode[0].start_s) > 1e-9:
            raise ValueError("This complete-Guard-reset batch must start its learner decisions at time0")
        for i, row in enumerate(episode):
            if not isinstance(row, DecisionTransition) or row.slot not in ("a1", "a2") \
                    or row.policy_id != expected_policy_id:
                raise ValueError("Only this frozen learner's DecisionTransition rows may update it")
            for features in (row.observation, row.next_observation):
                x = np.asarray(features)
                if x.shape != (30,) or x.dtype.kind not in "fiu" or not np.isfinite(x).all():
                    raise ValueError("Finite versioned30-dimensional policy features required")
            if row.observation[OWN_BUSY_FEATURE_INDEX] != 0:
                raise ValueError("Busy forced continuation is not an actor decision")
            mask = np.asarray(row.action_mask)
            if mask.shape != (5,) or mask.dtype.kind != "b" or not mask[0] or not mask[1:].any():
                raise ValueError("Saved actual ready-state boolean mask required")
            if (isinstance(row.action, (bool, np.bool_)) or not isinstance(row.action, numbers.Integral)
                    or row.action not in range(5) or not mask[row.action]):
                raise ValueError("Recorded actor action must be enabled by its saved mask")
            if _number(row.log_prob, "old log probability") > 1e-7:
                raise ValueError("Old log probability cannot be positive")
            if i and not np.array_equal(episode[i-1].next_observation, row.observation):
                raise ValueError("Decision observations are discontinuous within an episode")
        # This also rejects nonfinite values/rewards, noncontiguous times,
        # mixed slots and partial rounds. Lambda1 is full physical-time MC.
        adv, ret = physical_time_gae(episode, lambda_per_second=1.)
        rates.extend(-math.log(row.discount)/row.duration_s for row in episode)
        rows.extend(episode); advantages.extend(adv); targets.extend(ret)
    if max(rates)-min(rates) > 1e-8:
        raise ValueError("Do not mix different physical-time discounts in one PPO batch")
    dtype, device = next(policy.parameters()).dtype, next(policy.parameters()).device
    features = torch.as_tensor(np.stack([r.observation for r in rows]), dtype=dtype, device=device)
    actions = torch.tensor([r.action for r in rows], dtype=torch.long, device=device)
    masks = torch.as_tensor(np.stack([r.action_mask for r in rows]), dtype=torch.bool, device=device)
    old_logp = torch.tensor([r.log_prob for r in rows], dtype=dtype, device=device)
    old_values = torch.tensor([r.value for r in rows], dtype=dtype, device=device)
    raw_adv = torch.tensor(advantages, dtype=dtype, device=device)
    returns = torch.tensor(targets, dtype=dtype, device=device)
    if not all(torch.isfinite(x).all() for x in (features, old_logp, old_values, raw_adv, returns)):
        raise ValueError("Batch overflowed the policy dtype")
    with torch.no_grad():
        replay_logp, _, replay_values = policy.evaluate_actions(features, actions, masks)
        logp_error = float((replay_logp-old_logp).abs().max())
        value_error = float((replay_values-old_values).abs().max())
    if logp_error > REPLAY_ATOL or value_error > REPLAY_ATOL:
        raise ValueError("Old logprob/value cannot be replayed by the unchanged learner under saved masks")
    return (features, actions, masks, old_logp, raw_adv, returns,
            len(trajectories), logp_error, value_error)


def ppo_update_once(policy, episodes, *, expected_policy_id, batch_identity,
                    expected_compatibility_metadata=None, optimizer=None,
                    learning_rate=3e-4, clip_range=.2, value_coef=.5,
                    entropy_coef=0., max_grad_norm=.5):
    """Exactly one full-batch Adam step after complete on-policy validation.

    Pass a persistent Adam explicitly for later calls; the optional internally
    created Adam is only a standalone one-step smoke. This function is itself
    the explicit authority to mutate learner weights/optimizer, never called
    on import. Opponent/scene compatibility is checked only when supplied.
    """
    lr = _number(learning_rate, "learning rate", positive=True)
    clip = _number(clip_range, "clip range", positive=True)
    vf, ent = _number(value_coef, "value coefficient"), _number(entropy_coef, "entropy coefficient")
    max_norm = _number(max_grad_norm, "gradient norm limit", positive=True)
    if clip >= 1 or min(vf, ent) < 0:
        raise ValueError("Clip must be below1 and loss coefficients nonnegative")
    x, actions, masks, old_logp, raw_adv, returns, n_episodes, logp_error, value_error = _prepare(
        policy, episodes, expected_policy_id, batch_identity, expected_compatibility_metadata)
    parameters = tuple(policy.parameters())
    if optimizer is None:
        optimizer = torch.optim.Adam(parameters, lr=lr)
    if not isinstance(optimizer, torch.optim.Adam):
        raise ValueError("This minimal candidate supports explicit Adam only")
    owned = [p for group in optimizer.param_groups for p in group["params"]]
    if len(owned) != len(parameters) or {id(p) for p in owned} != {id(p) for p in parameters} \
            or any(float(group["lr"]) != lr for group in optimizer.param_groups):
        raise ValueError("Optimizer must own exactly this learner at the explicit learning rate")
    step_before = _step_count(optimizer)
    before = weights_hash(policy)
    # Population std over the ENTIRE real-choice batch. Targets are untouched.
    mean, std = raw_adv.mean(), raw_adv.std(unbiased=False)
    advantages = (raw_adv-mean)/(std+1e-8)
    logp, entropy, values = policy.evaluate_actions(x, actions, masks)
    ratio = torch.exp(logp-old_logp)
    surrogate = torch.minimum(ratio*advantages, ratio.clamp(1.-clip, 1.+clip)*advantages)
    actor_loss, value_loss = -surrogate.mean(), (values-returns).square().mean()
    entropy_mean = entropy.mean()
    loss = actor_loss+vf*value_loss-ent*entropy_mean
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite PPO loss cannot update weights")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
        optimizer.zero_grad(set_to_none=True)
        raise ValueError("Nonfinite gradients: optimizer.step was NOT called")
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    if not torch.isfinite(grad_norm) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
        optimizer.zero_grad(set_to_none=True)
        raise ValueError("Nonfinite clipped gradients: optimizer.step was NOT called")
    saved_weights = copy.deepcopy(policy.state_dict())
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    try:
        optimizer.step()
        after = weights_hash(policy)
        step_after = _step_count(optimizer)
        with torch.no_grad():
            post_logp, _, _ = policy.evaluate_actions(x, actions, masks)
            post_logratio = post_logp-old_logp
            post_ratio = torch.exp(post_logratio)
            if not torch.isfinite(post_ratio).all():
                raise ValueError("Nonfinite post-update policy ratio")
    except Exception:
        policy.load_state_dict(saved_weights)
        optimizer.load_state_dict(saved_optimizer)
        optimizer.zero_grad(set_to_none=True)
        raise
    return {"update_version": UPDATE_VERSION, "status": "one_explicit_optimizer_smoke_completed",
        "policy_id_before": expected_policy_id, "batch_identity": copy.deepcopy(batch_identity),
        "complete_learner_episodes": n_episodes, "actor_choice_rows": len(x),
        "epochs": 1, "optimizer_steps_this_call": 1,
        "optimizer_step_count_before": step_before, "optimizer_step_count_after": step_after,
        "learning_rate": lr, "clip_range": clip, "value_coef": vf, "entropy_coef": ent,
        "max_grad_norm": max_norm, "gradient_norm_before_clip": float(grad_norm),
        "finite_gradients": True, "loss": float(loss.detach()),
        "actor_loss": float(actor_loss.detach()), "value_loss": float(value_loss.detach()),
        "entropy": float(entropy_mean.detach()),
        "raw_advantage_mean": float(mean), "raw_advantage_population_std": float(std),
        "returns_min": float(returns.min()), "returns_max": float(returns.max()),
        "value_targets_normalized": False, "physical_time_lambda_per_second": 1.,
        "old_logprob_replay_max_error": logp_error, "old_value_replay_max_error": value_error,
        "float32_replay_absolute_tolerance": REPLAY_ATOL,
        "ratio_before_min": float(ratio.detach().min()), "ratio_before_max": float(ratio.detach().max()),
        "clip_fraction_before": float(((ratio.detach()-1.).abs()>clip).float().mean()),
        "ratio_after_min": float(post_ratio.min()), "ratio_after_max": float(post_ratio.max()),
        "clip_fraction_after": float(((post_ratio-1.).abs()>clip).float().mean()),
        "approx_kl_after": float((post_ratio-1.-post_logratio).mean()),
        "weights_sha256_before": before, "weights_sha256_after": after,
        "weights_changed": before != after,
        "environment_compatibility_checked": expected_compatibility_metadata is not None,
        "learned_fencing_ability_demonstrated": False, "admitted_to_training": False,
        "checkpoint_written": False, "native_or_training_experiment_run_by_module": False}
