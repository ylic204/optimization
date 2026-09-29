import torch
import torch.nn as nn


class JointDecisionBottleneck(nn.Module):
    """Task-conditioned latent for one-shot, fixed-budget token selection.

    Unlike V7.4, this module does not model a monotone acquisition trajectory.
    It sees all cheap preview tokens jointly and outputs one score per token.  A
    budget projection outside the module converts the scores into a complete
    K-token set.

    The set-value head is training-only.  It forces the latent to retain enough
    information to predict the downstream task loss of an arbitrary token set.
    It is deliberately a task-value head, not an optimization-model decoder.
    """

    def __init__(
        self,
        feat_dim,
        graph_dim=4,
        hidden=256,
        latent_dim=64,
        layers=4,
        heads=8,
    ):
        super().__init__()

        if hidden % heads != 0:
            raise ValueError("hidden must be divisible by heads")

        self.latent_dim = int(latent_dim)

        # preview visual feature + graph/task prior + budget
        self.in_proj = nn.Linear(feat_dim + graph_dim + 1, hidden)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden)
        self.latent_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        self.pool_query = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)
        pool_heads = min(4, latent_dim)
        while latent_dim % pool_heads != 0:
            pool_heads -= 1
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=max(1, pool_heads),
            batch_first=True,
        )

        score_in = latent_dim * 2 + graph_dim + 1
        self.score_head = nn.Sequential(
            nn.LayerNorm(score_in),
            nn.Linear(score_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

        # selected-set context + global scene context + budget
        self.set_value_head = nn.Sequential(
            nn.LayerNorm(latent_dim * 2 + 1),
            nn.Linear(latent_dim * 2 + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    @staticmethod
    def _budget_feature(budget_frac, batch, patches):
        if budget_frac.ndim == 0:
            budget_frac = budget_frac.expand(batch)
        if budget_frac.ndim == 1:
            return budget_frac[:, None, None].expand(batch, patches, 1)
        if budget_frac.ndim == 2:
            return budget_frac[..., None]
        return budget_frac

    @staticmethod
    def _weighted_pool(z, weight):
        weight = weight.float()
        return (z * weight[..., None]).sum(1) / weight.sum(
            1, keepdim=True
        ).clamp_min(1e-6)

    def encode(self, preview_feat, graph_feat, valid, budget_frac):
        batch, patches, _ = preview_feat.shape
        budget_feature = self._budget_feature(budget_frac, batch, patches)
        x = torch.cat([preview_feat, graph_feat, budget_feature], dim=-1)
        hidden = self.norm(
            self.encoder(self.in_proj(x), src_key_padding_mask=~valid)
        )
        return self.latent_head(hidden)

    def forward(self, preview_feat, graph_feat, valid, budget_frac):
        batch, patches, _ = preview_feat.shape
        z = self.encode(preview_feat, graph_feat, valid, budget_frac)

        query = self.pool_query.expand(batch, -1, -1)
        global_context, _ = self.pool_attn(
            query,
            z,
            z,
            key_padding_mask=~valid,
            need_weights=False,
        )
        global_context = global_context.expand(batch, patches, self.latent_dim)
        budget_feature = self._budget_feature(budget_frac, batch, patches)

        score_input = torch.cat(
            [z, global_context, graph_feat, budget_feature], dim=-1
        )
        logits = self.score_head(score_input).squeeze(-1)
        logits = logits.masked_fill(~valid, -1e9)

        return {
            "z": z,
            "global_context": global_context[:, 0],
            "logits": logits,
        }

    def predict_set_value(self, z, mask, valid, budget_frac):
        batch, patches, _ = z.shape
        selected = self._weighted_pool(z, mask * valid.float())
        global_context = self._weighted_pool(z, valid.float())

        if budget_frac.ndim == 0:
            budget_frac = budget_frac.expand(batch)
        elif budget_frac.ndim > 1:
            budget_frac = budget_frac.reshape(batch, -1)[:, 0]

        value_input = torch.cat(
            [selected, global_context, budget_frac[:, None]], dim=-1
        )
        return self.set_value_head(value_input).squeeze(-1)


def budgeted_soft_mask(logits, valid, k, temperature=0.35, iterations=40):
    """Differentiable mask whose valid-token mass is approximately exactly K."""
    valid_count = valid.sum(-1).clamp_min(1)
    if torch.is_tensor(k):
        target = k.to(logits).clamp(min=1.0, max=float(logits.shape[-1]))
        target = torch.minimum(target, valid_count.to(logits))
    else:
        target = torch.full_like(valid_count, float(k), dtype=logits.dtype)
        target = torch.minimum(target, valid_count.to(logits))

    with torch.no_grad():
        safe = logits.masked_fill(~valid, 0.0)
        lo = safe.min(-1).values - 30.0
        hi = safe.max(-1).values + 30.0
        for _ in range(iterations):
            threshold = 0.5 * (lo + hi)
            mass = (
                torch.sigmoid((logits - threshold[:, None]) / temperature)
                * valid.float()
            ).sum(-1)
            too_many = mass > target
            lo = torch.where(too_many, threshold, lo)
            hi = torch.where(too_many, hi, threshold)
        threshold = 0.5 * (lo + hi)

    return (
        torch.sigmoid((logits - threshold[:, None]) / temperature)
        * valid.float()
    )


def hard_topk_mask(logits, valid, k):
    mask = torch.zeros_like(logits)
    for row in range(logits.shape[0]):
        ids = torch.where(valid[row])[0]
        if ids.numel() == 0:
            continue
        keep = min(int(k), int(ids.numel()))
        chosen = ids[torch.topk(logits[row, ids], k=keep).indices]
        mask[row, chosen] = 1.0
    return mask


def random_topk_mask(valid, k, generator=None):
    mask = torch.zeros_like(valid, dtype=torch.float32)
    for row in range(valid.shape[0]):
        ids = torch.where(valid[row])[0]
        if ids.numel() == 0:
            continue
        keep = min(int(k), int(ids.numel()))
        perm = torch.randperm(
            int(ids.numel()), generator=generator, device="cpu"
        )[:keep].to(ids.device)
        mask[row, ids[perm]] = 1.0
    return mask

