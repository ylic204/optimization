"""Fixed-budget Plackett-Luce policy and external-outcome RL utilities.

The action is an ordered draw of K distinct regions, mapped to a spatially
sorted subset for inference. Its joint log-probability is exact; it is not
mistaken for the probability of an unordered set. Draws do not trigger new
perception or ND updates. This is an offline contextual-bandit policy.
"""

import torch
from torch.distributions import Categorical


def _validate_policy(scores, valid, k, temperature):
    if scores.ndim != 2 or valid.shape != scores.shape or valid.dtype != torch.bool:
        raise ValueError("scores and boolean valid must have shape [B,R]")
    if k <= 0 or bool((valid.sum(-1) < k).any()):
        raise ValueError("k must be positive and not exceed valid regions")
    if temperature <= 0 or not torch.isfinite(torch.tensor(temperature)):
        raise ValueError("policy temperature must be finite and positive")
    if not bool(torch.isfinite(scores[valid]).all()):
        raise ValueError("valid policy scores must be finite")


def ordered_action_statistics(scores, valid, actions, temperature=0.15, reference_scores=None):
    """Joint log-probability, conditional entropy and reference KL [B]."""
    if actions.ndim != 2 or actions.shape[0] != scores.shape[0]:
        raise ValueError("actions must have shape [B,K]")
    k = actions.shape[-1]
    _validate_policy(scores, valid, k, temperature)
    if actions.dtype != torch.long or bool(((actions < 0) | (actions >= scores.shape[1])).any()):
        raise ValueError("actions must contain in-range long indices")
    if reference_scores is not None:
        _validate_policy(reference_scores, valid, k, temperature)
    remaining = valid.clone()
    logp = scores.new_zeros(scores.shape[0])
    entropy = logp.clone()
    kl = logp.clone()
    for step in range(k):
        action = actions[:, step]
        if not bool(remaining.gather(1, action[:, None]).all()):
            raise ValueError("actions must be valid and sampled without replacement")
        distribution = Categorical(logits=(scores / temperature).masked_fill(~remaining, -torch.inf))
        logp = logp + distribution.log_prob(action)
        # Mask normalized log-probabilities BEFORE subtraction/products.
        # Generic categorical KL may form (-inf)-(-inf) at removed actions;
        # its forward can be finite while backward propagates NaNs.
        safe_logp = distribution.logits.masked_fill(~remaining, 0.0)
        entropy = entropy - (distribution.probs * safe_logp).sum(-1)
        if reference_scores is not None:
            reference = Categorical(
                logits=(reference_scores.detach() / temperature).masked_fill(~remaining, -torch.inf)
            )
            safe_reference_logp = reference.logits.masked_fill(~remaining, 0.0)
            kl = kl + (distribution.probs * (safe_logp - safe_reference_logp)).sum(-1)
        remaining = remaining.scatter(1, action[:, None], False)
    return logp, entropy, kl


def sample_region_action(scores, valid, k, temperature=0.15, reference_scores=None):
    _validate_policy(scores, valid, k, temperature)
    remaining = valid.clone()
    actions = []
    # Sampling has no pathwise gradient; log_probability carries REINFORCE.
    with torch.no_grad():
        for _ in range(k):
            distribution = Categorical(
                logits=(scores.detach() / temperature).masked_fill(~remaining, -torch.inf)
            )
            action = distribution.sample()
            actions.append(action)
            remaining = remaining.scatter(1, action[:, None], False)
    actions = torch.stack(actions, dim=-1)
    return (actions, *ordered_action_statistics(
        scores, valid, actions, temperature, reference_scores
    ))


def leave_one_out_advantages(rewards):
    """Independent-sample baseline; no critic or own-reward baseline leakage."""
    if rewards.ndim != 2 or rewards.shape[0] < 2:
        raise ValueError("rewards must have shape [G,B] with G >= 2")
    rewards = rewards.detach()
    baseline = (rewards.sum(0, keepdim=True) - rewards) / (rewards.shape[0] - 1)
    return rewards - baseline


def decision_outcomes(predicted_cost, oracle_cost, valid, metrics, limits, limit_mask, violation_weight=1.0):
    """Choose using predictions; reward using independently constructed labels."""
    if bool((valid.sum(-1) == 0).any()):
        raise ValueError("every scene must contain a valid candidate")
    chosen = predicted_cost.masked_fill(~valid, torch.inf).argmin(-1)
    minimum = oracle_cost.masked_fill(~valid, torch.inf).min(-1).values
    maximum = oracle_cost.masked_fill(~valid, -torch.inf).max(-1).values
    rows = torch.arange(chosen.shape[0], device=chosen.device)
    regret = (oracle_cost[rows, chosen] - minimum) / (maximum - minimum).clamp_min(1e-4)
    violation = ((metrics[rows, chosen] > limits) & limit_mask.bool()).any(-1).float()
    reward = -regret - float(violation_weight) * violation
    return reward.detach(), regret.detach(), violation.detach()


def policy_gradient_loss(log_probabilities, rewards, entropies, reference_kls, kl_weight=0.02, entropy_weight=0.001):
    advantage = leave_one_out_advantages(rewards)
    policy = -(advantage * log_probabilities).mean()
    # KL/entropy averaged per decision, not mislabeled as full set-distribution KL.
    loss = policy + float(kl_weight) * reference_kls.mean() - float(entropy_weight) * entropies.mean()
    return loss, policy, advantage
