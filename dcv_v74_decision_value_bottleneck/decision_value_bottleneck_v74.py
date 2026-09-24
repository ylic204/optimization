import torch
import torch.nn as nn
import torch.nn.functional as F


class DecisionValueBottleneck(nn.Module):
    """
    Causal visual observation -> stochastic decision-value bottleneck
                              -> value of observing every candidate patch.

    Important:
    - No true edge/path cost is used as inference input.
    - No edge-cost reconstruction is required.
    - What survives the bottleneck is defined by downstream decision value.

    q(z_j | o_t) = N(mu_j, diag(sigma_j^2))

    The value head is explicitly GLOBAL:
      score_j = f(z_j, global(z_1...z_M), graph_j, w_j, time, budget)

    This fixes the overly local scoring head used in V7.3.
    """

    def __init__(
        self,
        feat_dim,
        graph_dim=4,
        hidden=256,
        latent_dim=32,
        layers=4,
        heads=8,
    ):
        super().__init__()

        if hidden % heads != 0:
            raise ValueError("hidden must be divisible by heads")

        self.latent_dim = int(latent_dim)

        # visual feature + graph feature + acquired bit + step + budget
        self.in_proj = nn.Linear(
            feat_dim + graph_dim + 3,
            hidden,
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=layers,
        )

        self.norm = nn.LayerNorm(hidden)

        self.mu_head = nn.Linear(
            hidden,
            latent_dim,
        )

        self.logvar_head = nn.Linear(
            hidden,
            latent_dim,
        )

        # Attention pooling for a global decision context.
        self.pool_query = nn.Parameter(
            torch.randn(1, 1, latent_dim) * 0.02
        )

        self.pool_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=max(1, min(4, latent_dim)),
            batch_first=True,
        )

        head_in = (
            latent_dim        # local z_j
            + latent_dim      # global context
            + graph_dim
            + 3               # w_j, step, budget
        )

        self.value_head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

        # Gradient auxiliary head: same context, separate output.
        self.grad_head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    @staticmethod
    def _scalar_to_patch(x, B, M):
        if x.ndim == 1:
            return x[:, None, None].expand(B, M, 1)
        if x.ndim == 2:
            return x[..., None]
        return x

    def encode(
        self,
        state_feat,
        graph_feat,
        w,
        outer_t,
        budget_frac,
        sample=True,
    ):
        B, M, _ = state_feat.shape

        aux = torch.cat(
            [
                self._scalar_to_patch(w, B, M),
                self._scalar_to_patch(outer_t, B, M),
                self._scalar_to_patch(budget_frac, B, M),
            ],
            dim=-1,
        )

        h = torch.cat(
            [
                state_feat,
                graph_feat,
                aux,
            ],
            dim=-1,
        )

        h = self.norm(
            self.encoder(
                self.in_proj(h)
            )
        )

        mu = self.mu_head(h)

        logvar = self.logvar_head(h).clamp(
            min=-8.0,
            max=4.0,
        )

        if sample and self.training:
            eps = torch.randn_like(mu)
            z = mu + torch.exp(0.5 * logvar) * eps
        else:
            z = mu

        return z, mu, logvar

    def forward(
        self,
        state_feat,
        graph_feat,
        w,
        outer_t,
        budget_frac,
        eligible,
        sample=True,
    ):
        B, M, _ = state_feat.shape

        z, mu, logvar = self.encode(
            state_feat,
            graph_feat,
            w,
            outer_t,
            budget_frac,
            sample=sample,
        )

        query = self.pool_query.expand(
            B,
            -1,
            -1,
        )

        # We deliberately pool all graph-valid patches, including already
        # observed patches: they contain useful current-decision context.
        global_ctx, _ = self.pool_attn(
            query,
            z,
            z,
            need_weights=False,
        )

        global_ctx = global_ctx.expand(
            B,
            M,
            self.latent_dim,
        )

        aux = torch.cat(
            [
                self._scalar_to_patch(w, B, M),
                self._scalar_to_patch(outer_t, B, M),
                self._scalar_to_patch(budget_frac, B, M),
            ],
            dim=-1,
        )

        hv = torch.cat(
            [
                z,
                global_ctx,
                graph_feat,
                aux,
            ],
            dim=-1,
        )

        value = self.value_head(
            hv
        ).squeeze(-1)

        grad_logits = self.grad_head(
            hv
        ).squeeze(-1)

        value_logits = value.masked_fill(
            ~eligible,
            -1e9,
        )

        grad_logits_masked = grad_logits.masked_fill(
            ~eligible,
            -1e9,
        )

        return {
            "z": z,
            "mu": mu,
            "logvar": logvar,
            "value": value,
            "value_logits": value_logits,
            "grad_logits": grad_logits,
            "grad_logits_masked": grad_logits_masked,
        }


def vib_kl(mu, logvar, patch_mask=None):
    """
    KL[q(z|o) || N(0,I)].
    This is the compression term.
    Downstream decision-value losses determine what information survives.
    """
    kl = -0.5 * (
        1.0
        + logvar
        - mu.pow(2)
        - logvar.exp()
    )

    if patch_mask is None:
        return kl.mean()

    m = patch_mask.float()[..., None]

    return (
        (kl * m).sum()
        / (
            m.sum()
            * kl.shape[-1]
        ).clamp_min(1.0)
    )


def masked_policy_entropy(logits, eligible):
    p = torch.softmax(
        logits.masked_fill(
            ~eligible,
            -1e9,
        ),
        dim=-1,
    )

    logp = torch.log(
        p.clamp_min(1e-8)
    )

    return -(
        p * logp
    ).sum(
        dim=-1
    )
