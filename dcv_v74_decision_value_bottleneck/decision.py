import torch
import torch.nn.functional as F
from config import CFG


def move_batch(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def gather_edge_features(feat, edge_patch):
    B, E = edge_patch.shape
    D = feat.shape[-1]
    idx = edge_patch[..., None].expand(B, E, D)
    return torch.gather(feat, 1, idx)


def gather_edge_values(x, edge_patch):
    return torch.gather(x, 1, edge_patch)


def legal_mask(batch, w):
    return (batch['patch_graph_feat'][..., 0] > 0.5) & (w < 0.5)


def assert_binary(w):
    ok = ((w == 0) | (w == 1)).all()
    if not bool(ok):
        raise RuntimeError(
            'Causal policy state requires a binary acquisition mask; '
            'soft masks would leak high-resolution information.'
        )


def causal_state_features(preview_feat, high_feat, w):
    assert_binary(w)
    return torch.where(w[..., None].bool(), high_feat, preview_feat)


def state_risks(device):
    return torch.tensor(
        [
            CFG.risk_normal,
            CFG.risk_rough,
            CFG.risk_hazard,
            CFG.blocked_penalty,
        ],
        device=device,
        dtype=torch.float32,
    )


def predicted_edge_cost(perception, state_feat, w, batch):
    """Estimated edge task cost under the CURRENT causal visual state."""
    assert_binary(w)
    ef = gather_edge_features(state_feat, batch['edge_patch'])
    ew = gather_edge_values(w, batch['edge_patch'])

    preview_pen = perception.predict_preview_penalty(ef)
    high_prob = perception.classify_high_features(ef).softmax(-1)
    high_pen = (high_prob * state_risks(high_prob.device)).sum(-1)

    return batch['base_cost'] + torch.where(ew.bool(), high_pen, preview_pen)


def true_path_costs(batch):
    """C_true(P) for every candidate path."""
    return torch.einsum('bpe,be->bp', batch['path_mask'], batch['true_edge_cost'])


def estimated_path_costs(perception, state_feat, w, batch):
    edge_cost = predicted_edge_cost(perception, state_feat, w, batch)
    path_cost = torch.einsum('bpe,be->bp', batch['path_mask'], edge_cost)
    return edge_cost, path_cost


def hard_decision_from_state_feat(perception, state_feat, w, batch):
    """
    HARD evaluation decision.

    1) choose the path minimizing the CURRENT estimated task cost;
    2) evaluate the chosen path with full-information TRUE task cost;
    3) compare against the minimum full-information true task cost.

    Different path IDs are all optimal if their true task costs are equal
    within tolerance.
    """
    _, estimated_pc = estimated_path_costs(perception, state_feat, w, batch)
    chosen_idx = estimated_pc.argmin(-1)

    true_pc = true_path_costs(batch)
    chosen_true_cost = true_pc.gather(1, chosen_idx[:, None]).squeeze(1)
    optimal_true_cost = true_pc.min(dim=-1).values

    regret = (
        (chosen_true_cost - optimal_true_cost)
        / optimal_true_cost.clamp_min(CFG.grad_eps)
    ).clamp_min(0.0)

    abs_gap = chosen_true_cost - optimal_true_cost
    tol = CFG.optimal_cost_atol + CFG.optimal_cost_rtol * optimal_true_cost.abs()
    optimal = abs_gap <= tol

    return {
        'chosen_path_idx': chosen_idx,
        'chosen_true_cost': chosen_true_cost,
        'optimal_true_cost': optimal_true_cost,
        'hard_regret': regret,
        'optimal': optimal,
        'estimated_path_costs': estimated_pc,
        'true_path_costs': true_pc,
    }


def exact_outcome_from_state_feat(perception, state_feat, w, batch):
    """Backward-compatible wrapper used by older V7 scripts."""
    out = hard_decision_from_state_feat(perception, state_feat, w, batch)
    return out['hard_regret'], out['chosen_path_idx']


def soft_decision_regret_from_state_feat(perception, state_feat, w, batch, tau=None):
    """
    DIFFERENTIABLE training loss.

        q(P|w) = softmax(-C_hat(P;w)/tau)
        L_soft = [sum_P q(P|w) C_true(P) - C*] / C*

    This is a soft surrogate of the hard downstream decision regret.
    """
    tau = CFG.teacher_tau if tau is None else tau
    _, estimated_pc = estimated_path_costs(perception, state_feat, w, batch)
    q = torch.softmax(-estimated_pc / tau, dim=-1)

    true_pc = true_path_costs(batch)
    expected_true_cost = (q * true_pc).sum(-1)
    optimal_true_cost = true_pc.min(dim=-1).values

    return (
        (expected_true_cost - optimal_true_cost)
        / optimal_true_cost.clamp_min(CFG.grad_eps)
    ).clamp_min(0.0)


# Backward-compatible alias.
soft_task_regret_from_state_feat = soft_decision_regret_from_state_feat


def semantic_current_edge_cost(w_binary, batch):
    """
    Privileged controlled-world edge cost used to isolate VISUAL SELECTION.
    Unseen edge: coarse normal/abnormal cost.
    Seen edge: true full-information task cost.
    """
    assert_binary(w_binary)
    edge_w = gather_edge_values(w_binary, batch['edge_patch'])
    abnormal = (batch['edge_state'] != 0).float()
    coarse = batch['base_cost'] + abnormal * CFG.preview_abnormal_penalty
    return torch.where(edge_w.bool(), batch['true_edge_cost'], coarse)


def semantic_hard_decision(w_binary, batch):
    ec = semantic_current_edge_cost(w_binary, batch)
    estimated_pc = torch.einsum('bpe,be->bp', batch['path_mask'], ec)
    chosen_idx = estimated_pc.argmin(-1)

    true_pc = true_path_costs(batch)
    chosen_true_cost = true_pc.gather(1, chosen_idx[:, None]).squeeze(1)
    optimal_true_cost = true_pc.min(dim=-1).values

    regret = (
        (chosen_true_cost - optimal_true_cost)
        / optimal_true_cost.clamp_min(CFG.grad_eps)
    ).clamp_min(0.0)

    gap = chosen_true_cost - optimal_true_cost
    tol = CFG.optimal_cost_atol + CFG.optimal_cost_rtol * optimal_true_cost.abs()
    optimal = gap <= tol

    return {
        'chosen_path_idx': chosen_idx,
        'chosen_true_cost': chosen_true_cost,
        'optimal_true_cost': optimal_true_cost,
        'hard_regret': regret,
        'optimal': optimal,
        'estimated_path_costs': estimated_pc,
        'true_path_costs': true_pc,
    }


def semantic_exact_regret(w_binary, batch):
    """Backward-compatible regret-only wrapper."""
    return semantic_hard_decision(w_binary, batch)['hard_regret']


def semantic_soft_decision_regret(w_cont, batch, tau=None):
    """
    Privileged differentiable Teacher decision loss.

    w in [0,1]^M is ONLY an abstract information-fidelity coordinate for
    differentiation. It is never exposed as soft high-resolution visual input.
    """
    tau = CFG.teacher_tau if tau is None else tau
    edge_w = gather_edge_values(w_cont, batch['edge_patch']).clamp(0, 1)
    abnormal = (batch['edge_state'] != 0).float()

    coarse = batch['base_cost'] + abnormal * CFG.preview_abnormal_penalty
    true = batch['true_edge_cost']
    edge_cost = (1.0 - edge_w) * coarse + edge_w * true

    estimated_pc = torch.einsum('bpe,be->bp', batch['path_mask'], edge_cost)
    q = torch.softmax(-estimated_pc / tau, dim=-1)

    true_pc = true_path_costs(batch)
    expected_true_cost = (q * true_pc).sum(-1)
    optimal_true_cost = true_pc.min(dim=-1).values

    return (
        (expected_true_cost - optimal_true_cost)
        / optimal_true_cost.clamp_min(CFG.grad_eps)
    ).clamp_min(0.0)


# Backward-compatible alias.
semantic_soft_regret = semantic_soft_decision_regret


def normalize_direction(g, eligible):
    x = torch.relu(g) * eligible.float()
    n = torch.linalg.vector_norm(x, dim=-1, keepdim=True)

    zero = n.squeeze(-1) <= CFG.grad_eps
    if zero.any():
        xa = torch.abs(g) * eligible.float()
        x[zero] = xa[zero]
        n = torch.linalg.vector_norm(x, dim=-1, keepdim=True)

    return x / n.clamp_min(CFG.grad_eps)


def teacher_gradient_direction(w_binary, batch):
    """
    g^T = Normalize(ReLU(-d L_soft_decision / d w)).
    """
    with torch.enable_grad():
        w = w_binary.detach().clone().requires_grad_(True)
        loss = semantic_soft_decision_regret(w, batch).mean()
        grad = torch.autograd.grad(loss, w, create_graph=False)[0]

    eligible = legal_mask(batch, w_binary)
    return normalize_direction(-grad, eligible).detach()


def exact_counterfactual_gains(perception, preview_feat, high_feat, w, batch):
    """Evaluation-only exact one-step hard-regret improvement."""
    B, M = w.shape
    sf = causal_state_features(preview_feat, high_feat, w)
    r0 = hard_decision_from_state_feat(perception, sf, w, batch)['hard_regret']

    gains = torch.full((B, M), -1e9, device=w.device)
    elig = legal_mask(batch, w)

    for j in range(M):
        rows = elig[:, j]
        if not rows.any():
            continue
        wc = w.clone()
        wc[rows, j] = 1.0
        sfc = causal_state_features(preview_feat, high_feat, wc)
        rc = hard_decision_from_state_feat(perception, sfc, wc, batch)['hard_regret']
        gains[rows, j] = r0[rows] - rc[rows]

    return gains, r0


def masked_cosine(a, b, mask):
    aa = a * mask.float()
    bb = b * mask.float()
    return F.cosine_similarity(aa, bb, dim=-1, eps=1e-8)
