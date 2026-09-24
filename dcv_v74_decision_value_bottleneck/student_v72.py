import torch
import torch.nn as nn
import torch.nn.functional as F
from config import CFG


class GradientStudentV72(nn.Module):
    """
    Configurable gradient-distillation Student.

    It can be used in:
      1) deployable visual-only mode;
      2) privileged semantic capacity sanity mode.

    head_mode:
      - 'relu_l2': sparse nonnegative L2-normalized direction;
      - 'softmax': probability-like ranking distribution.
    """

    def __init__(
        self,
        feat_dim=None,
        graph_dim=4,
        hidden=128,
        layers=2,
        heads=4,
        privileged_dim=0,
        head_mode='relu_l2',
        head_temperature=1.0,
    ):
        super().__init__()
        feat_dim = CFG.feat_dim if feat_dim is None else int(feat_dim)

        if hidden % heads != 0:
            raise ValueError(f'hidden={hidden} must be divisible by heads={heads}')

        self.privileged_dim = int(privileged_dim)
        self.head_mode = str(head_mode)
        self.head_temperature = float(head_temperature)

        in_dim = feat_dim + graph_dim + 3 + self.privileged_dim
        self.in_proj = nn.Linear(in_dim, hidden)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, 1)

    @staticmethod
    def _expand(x, B, M):
        if x.ndim == 1:
            return x[:, None, None].expand(B, M, 1)
        if x.ndim == 2:
            return x[..., None]
        return x

    def forward(
        self,
        state_feat,
        graph_feat,
        w_binary,
        outer_t,
        budget_frac,
        eligible,
        privileged=None,
    ):
        B, M, _ = state_feat.shape

        pieces = [
            state_feat,
            graph_feat,
            self._expand(w_binary, B, M),
            self._expand(outer_t, B, M),
            self._expand(budget_frac, B, M),
        ]

        if self.privileged_dim > 0:
            if privileged is None:
                raise ValueError('privileged input is required')
            pieces.append(privileged)

        x = torch.cat(pieces, dim=-1)
        h = self.norm(self.encoder(self.in_proj(x)))
        logits = self.head(h).squeeze(-1)

        if self.head_mode == 'relu_l2':
            score = F.relu(logits) * eligible.float()
            norm = torch.linalg.vector_norm(score, dim=-1, keepdim=True)

            # avoid an all-zero dead ReLU head at initialization
            dead = norm.squeeze(-1) <= CFG.grad_eps
            if dead.any():
                fallback = F.softplus(logits) * eligible.float()
                score = torch.where(dead[:, None], fallback, score)
                norm = torch.linalg.vector_norm(score, dim=-1, keepdim=True)

            direction = score / norm.clamp_min(CFG.grad_eps)

        elif self.head_mode == 'softmax':
            masked = logits.masked_fill(~eligible, -1e9)
            direction = torch.softmax(masked / self.head_temperature, dim=-1)
            direction = direction * eligible.float()

        else:
            raise ValueError(f'unknown head_mode={self.head_mode}')

        return direction, logits


def teacher_distribution(target, eligible, eps=1e-8):
    p = torch.relu(target) * eligible.float()
    return p / p.sum(dim=-1, keepdim=True).clamp_min(eps)


def gradient_distillation_loss(
    pred,
    logits,
    target,
    eligible,
    lambda_cos=1.0,
    lambda_kl=0.25,
    lambda_rank=0.10,
    rank_margin=0.02,
):
    """
    Distillation objective aligned with the actual use of the gradient:
      - direction similarity;
      - probability/ranking agreement;
      - pairwise order preservation.
    """
    m = eligible.float()

    pred_l2 = pred * m
    pred_l2 = pred_l2 / torch.linalg.vector_norm(
        pred_l2, dim=-1, keepdim=True
    ).clamp_min(1e-8)

    target_l2 = target * m
    target_l2 = target_l2 / torch.linalg.vector_norm(
        target_l2, dim=-1, keepdim=True
    ).clamp_min(1e-8)

    cosine = F.cosine_similarity(pred_l2, target_l2, dim=-1, eps=1e-8)
    loss_cos = (1.0 - cosine).mean()

    p_t = teacher_distribution(target, eligible)
    masked_logits = logits.masked_fill(~eligible, -1e9)
    log_p_s = F.log_softmax(masked_logits, dim=-1)
    loss_kl = F.kl_div(log_p_s, p_t, reduction='batchmean')

    # Pairwise rank loss across eligible patches where the Teacher has a clear order.
    ti = target[:, :, None]
    tj = target[:, None, :]
    li = logits[:, :, None]
    lj = logits[:, None, :]

    valid = eligible[:, :, None] & eligible[:, None, :]
    teacher_better = (ti - tj) > rank_margin
    pair_mask = valid & teacher_better

    if pair_mask.any():
        pair_loss = F.softplus(-(li - lj))
        loss_rank = pair_loss[pair_mask].mean()
    else:
        loss_rank = logits.sum() * 0.0

    total = (
        lambda_cos * loss_cos
        + lambda_kl * loss_kl
        + lambda_rank * loss_rank
    )

    return {
        'loss': total,
        'cosine': cosine.mean(),
        'loss_cos': loss_cos,
        'loss_kl': loss_kl,
        'loss_rank': loss_rank,
    }
