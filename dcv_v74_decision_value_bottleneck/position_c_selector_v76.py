import torch
import torch.nn as nn


class FrozenLLMTaskEncoder(nn.Module):
    """Frozen Hugging Face language backbone used only to produce task q.

    Modern generative VLMs normally reuse a decoder-only LLM rather than a
    separate trainable language encoder.  We therefore pool the frozen LLM's
    last hidden states and let the selector learn a small task projection.
    """

    def __init__(self, model_name_or_path, local_files_only=False, max_length=96):
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "V7.6 needs transformers: pip install transformers"
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.backbone = AutoModel.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
            trust_remote_code=True,
            torch_dtype="auto",
        )
        self.backbone.eval()
        self.backbone.requires_grad_(False)
        self.max_length = int(max_length)

        config = self.backbone.config
        text_config = getattr(config, "text_config", config)
        self.output_dim = next(
            int(getattr(text_config, name))
            for name in ("hidden_size", "d_model", "n_embd")
            if getattr(text_config, name, None) is not None
        )

    def train(self, mode=True):
        # Keep the language backbone deterministic and frozen even when the
        # enclosing training module switches to train mode.
        super().train(False)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def forward(self, texts, device):
        tokens = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tokens = {name: value.to(device) for name, value in tokens.items()}
        output = self.backbone(**tokens, return_dict=True)
        hidden = output.last_hidden_state.float()
        weight = tokens["attention_mask"].to(hidden.dtype)[..., None]
        return (hidden * weight).sum(1) / weight.sum(1).clamp_min(1.0)


def normalized_patch_positions(batch, patches, device, dtype):
    """Observable 2-D coordinates; no graph or future-map information."""
    side = int(round(patches ** 0.5))
    if side * side != patches:
        raise ValueError("V7.6 currently expects a square patch grid")
    axis = torch.linspace(-1.0, 1.0, side, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    xy = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
    return xy[None].expand(batch, -1, -1)


class PositionCDecisionBottleneck(nn.Module):
    """Task-conditioned selector after the full vision encoder (position C).

    Student-visible inputs are only full vision tokens, 2-D token positions,
    frozen-LLM task embedding q, and the budget.  No graph feature is accepted.
    """

    def __init__(
        self,
        vision_dim,
        task_dim,
        hidden=256,
        latent_dim=64,
        task_hidden=128,
        position_dim=32,
        layers=4,
        heads=8,
    ):
        super().__init__()
        if hidden % heads != 0:
            raise ValueError("hidden must be divisible by heads")
        self.latent_dim = int(latent_dim)

        self.task_proj = nn.Sequential(
            nn.LayerNorm(task_dim),
            nn.Linear(task_dim, task_hidden),
            nn.GELU(),
            nn.LayerNorm(task_hidden),
        )
        self.position_proj = nn.Sequential(
            nn.Linear(2, position_dim),
            nn.GELU(),
            nn.Linear(position_dim, position_dim),
        )
        self.in_proj = nn.Linear(
            vision_dim + position_dim + task_hidden + 1, hidden
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
            latent_dim, max(1, pool_heads), batch_first=True
        )

        score_dim = latent_dim * 2 + position_dim + task_hidden + 1
        self.score_head = nn.Sequential(
            nn.LayerNorm(score_dim),
            nn.Linear(score_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )
        value_dim = latent_dim * 2 + task_hidden + 1
        self.set_value_head = nn.Sequential(
            nn.LayerNorm(value_dim),
            nn.Linear(value_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    @staticmethod
    def _budget_feature(budget, batch, patches):
        if budget.ndim == 0:
            budget = budget.expand(batch)
        return budget.reshape(batch, 1, 1).expand(batch, patches, 1)

    @staticmethod
    def _weighted_pool(z, weight):
        weight = weight.float()
        return (z * weight[..., None]).sum(1) / weight.sum(
            1, keepdim=True
        ).clamp_min(1e-6)

    def forward(self, vision_tokens, position_xy, task_embedding, valid, budget):
        batch, patches, _ = vision_tokens.shape
        q = self.task_proj(task_embedding)
        q_token = q[:, None].expand(batch, patches, -1)
        pos = self.position_proj(position_xy)
        budget_token = self._budget_feature(budget, batch, patches)

        x = torch.cat([vision_tokens, pos, q_token, budget_token], dim=-1)
        h = self.norm(
            self.encoder(self.in_proj(x), src_key_padding_mask=~valid)
        )
        z = self.latent_head(h)

        query = self.pool_query.expand(batch, -1, -1)
        global_context, _ = self.pool_attn(
            query, z, z, key_padding_mask=~valid, need_weights=False
        )
        global_token = global_context.expand(batch, patches, -1)
        score_input = torch.cat(
            [z, global_token, pos, q_token, budget_token], dim=-1
        )
        logits = self.score_head(score_input).squeeze(-1)
        logits = logits.masked_fill(~valid, -1e9)
        return {
            "z": z,
            "task_context": q,
            "global_context": global_context[:, 0],
            "logits": logits,
        }

    def predict_set_value(self, z, task_context, mask, valid, budget):
        batch = z.shape[0]
        selected = self._weighted_pool(z, mask * valid.float())
        global_context = self._weighted_pool(z, valid.float())
        budget = budget.reshape(batch, -1)[:, 0]
        value_input = torch.cat(
            [selected, global_context, task_context, budget[:, None]], dim=-1
        )
        return self.set_value_head(value_input).squeeze(-1)


def gather_selected_tokens(vision_tokens, logits, valid, k):
    """Return the actual [B,K,D] sequence sent to the downstream LLM."""
    if (valid.sum(-1) < int(k)).any():
        raise ValueError("every sample must contain at least k valid tokens")
    masked_logits = logits.masked_fill(~valid, -1e9)
    indices = torch.topk(masked_logits, k=int(k), dim=-1).indices
    gather_index = indices[..., None].expand(-1, -1, vision_tokens.shape[-1])
    selected = torch.gather(vision_tokens, 1, gather_index)
    return selected, indices
