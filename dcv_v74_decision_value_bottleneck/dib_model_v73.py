import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CFG
from decision import gather_edge_values


class DecisionInformationBottleneck(nn.Module):
    """
    Causal visual state -> stochastic optimization-aware latent Z_opt
                        -> edge-cost estimate
                        -> decision-gradient distribution

    The latent is learned. No true edge/path costs are given as inference input.

    VIB:
        q(z|o) = N(mu(o), diag(sigma^2(o)))
        L_IB = KL(q(z|o) || N(0,I))

    Each patch has a latent vector z_j. A contextual Transformer lets the latent
    encode global decision relations rather than only local appearance.
    """

    def __init__(
        self,
        feat_dim=None,
        graph_dim=4,
        hidden=256,
        latent_dim=32,
        layers=4,
        heads=8,
        head_temperature=1.0,
    ):
        super().__init__()

        feat_dim = CFG.feat_dim if feat_dim is None else int(feat_dim)

        if hidden % heads != 0:
            raise ValueError(
                f"hidden={hidden} must be divisible by heads={heads}"
            )

        self.latent_dim = int(latent_dim)
        self.head_temperature = float(head_temperature)

        # visual feature + graph feature + acquisition bit + outer step + budget
        self.in_proj = nn.Linear(
            feat_dim + graph_dim + 3,
            hidden,
        )

        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )

        self.context = nn.TransformerEncoder(
            layer,
            num_layers=layers,
        )

        self.context_norm = nn.LayerNorm(
            hidden
        )

        self.mu_head = nn.Linear(
            hidden,
            latent_dim,
        )

        self.logvar_head = nn.Linear(
            hidden,
            latent_dim,
        )

        # Decode optimization state from bottleneck.
        self.cost_decoder = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

        # Gradient/policy score head.
        self.grad_decoder = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    @staticmethod
    def _expand(x, B, M):
        if x.ndim == 1:
            return x[:, None, None].expand(B, M, 1)
        if x.ndim == 2:
            return x[..., None]
        return x

    def encode(
        self,
        state_feat,
        graph_feat,
        w_binary,
        outer_t,
        budget_frac,
        sample=True,
    ):
        B, M, _ = state_feat.shape

        x = torch.cat(
            [
                state_feat,
                graph_feat,
                self._expand(w_binary, B, M),
                self._expand(outer_t, B, M),
                self._expand(budget_frac, B, M),
            ],
            dim=-1,
        )

        h = self.context_norm(
            self.context(
                self.in_proj(x)
            )
        )

        mu = self.mu_head(h)

        # Clamp for numerical stability.
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
        w_binary,
        outer_t,
        budget_frac,
        eligible,
        base_cost,
        edge_patch,
        sample=True,
    ):
        z, mu, logvar = self.encode(
            state_feat,
            graph_feat,
            w_binary,
            outer_t,
            budget_frac,
            sample=sample,
        )

        # Nonnegative visual/risk penalty. Edge base geometry cost stays explicit.
        patch_penalty = F.softplus(
            self.cost_decoder(z).squeeze(-1)
        )

        edge_penalty = gather_edge_values(
            patch_penalty,
            edge_patch,
        )

        pred_edge_cost = (
            base_cost
            + edge_penalty
        )

        grad_logits = self.grad_decoder(
            z
        ).squeeze(-1)

        masked_logits = grad_logits.masked_fill(
            ~eligible,
            -1e9,
        )

        grad_prob = torch.softmax(
            masked_logits / self.head_temperature,
            dim=-1,
        )

        grad_prob = (
            grad_prob
            * eligible.float()
        )

        return {
            "z":
                z,

            "mu":
                mu,

            "logvar":
                logvar,

            "patch_penalty":
                patch_penalty,

            "pred_edge_cost":
                pred_edge_cost,

            "grad_logits":
                grad_logits,

            "grad_prob":
                grad_prob,
        }


def vib_kl(mu, logvar, valid_patch_mask=None):
    """
    KL[N(mu,sigma^2) || N(0,I)] averaged over valid patches and latent dims.
    This is the variational upper-bound style compression penalty on I(Z;X).
    """
    kl = -0.5 * (
        1.0
        + logvar
        - mu.pow(2)
        - logvar.exp()
    )

    if valid_patch_mask is not None:
        m = valid_patch_mask.float()[..., None]

        return (
            (kl * m).sum()
            / (
                m.sum()
                * kl.shape[-1]
            ).clamp_min(1.0)
        )

    return kl.mean()


def categorical_entropy(prob, eps=1e-8):
    """
    H(P) = -sum p log p.
    This is DECISION entropy over candidate paths, not raw image entropy.
    """
    p = prob.clamp_min(eps)

    return -(
        p * p.log()
    ).sum(
        dim=-1
    )
