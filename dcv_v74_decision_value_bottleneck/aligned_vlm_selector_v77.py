import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrozenBlipAlignedEncoder(nn.Module):
    """Frozen BLIP vision/text towers and their pretrained projections.

    The selector consumes aligned representations. Raw pooled visual region
    tokens are returned separately for the downstream decision module.
    """

    def __init__(
        self,
        model_name_or_path,
        region_grid=6,
        max_text_length=64,
        local_files_only=False,
    ):
        super().__init__()
        try:
            from transformers import AutoProcessor, BlipModel
        except ImportError as exc:
            raise ImportError(
                "V7.7 requires transformers with BLIP support"
            ) from exc

        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
        )
        self.model = BlipModel.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
            torch_dtype="auto",
        )
        self.model.eval().requires_grad_(False)
        self.region_grid = int(region_grid)
        self.max_text_length = int(max_text_length)
        self.raw_vision_dim = int(self.model.config.vision_config.hidden_size)
        self.aligned_dim = int(self.model.config.projection_dim)
        self.text_hidden_dim = int(self.model.config.text_config.hidden_size)

    def train(self, mode=True):
        super().train(False)
        self.model.eval()
        return self

    def _process_images(self, images, device):
        # Dataset tensors are already float RGB in [0,1]. BLIP's processor
        # performs its pretrained resize/crop/normalization but must not divide
        # them by 255 a second time.
        image_list = [image.detach().cpu() for image in images]
        batch = self.processor.image_processor(
            images=image_list,
            do_rescale=False,
            return_tensors="pt",
        )
        return batch["pixel_values"].to(device)

    def _process_text(self, texts, device):
        tokens = self.processor.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        return {key: value.to(device) for key, value in tokens.items()}

    def _pool_to_regions(self, patch_tokens):
        batch, patches, channels = patch_tokens.shape
        side = int(round(math.sqrt(patches)))
        if side * side != patches:
            raise ValueError(
                f"BLIP patch sequence ({patches}) is not a square grid"
            )
        feature_map = patch_tokens.transpose(1, 2).reshape(
            batch, channels, side, side
        )
        pooled = F.adaptive_avg_pool2d(
            feature_map, (self.region_grid, self.region_grid)
        )
        return pooled.flatten(2).transpose(1, 2)

    @torch.no_grad()
    def forward(self, images, texts):
        device = images.device
        pixel_values = self._process_images(images, device)
        text_inputs = self._process_text(texts, device)

        vision_output = self.model.vision_model(
            pixel_values=pixel_values,
            return_dict=True,
        )
        text_output = self.model.text_model(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
            return_dict=True,
        )

        # BLIP token 0 is the vision class token. The controlled navigation
        # dataset supervises a 6x6 region grid, so the finer BLIP token grid is
        # spatially pooled to the same observable regions.
        raw_regions = self._pool_to_regions(
            vision_output.last_hidden_state[:, 1:]
        )
        aligned_regions = F.normalize(
            self.model.visual_projection(raw_regions).float(), dim=-1
        )
        aligned_text = F.normalize(
            self.model.text_projection(
                text_output.last_hidden_state
            ).float(),
            dim=-1,
        )
        return {
            "raw_visual_tokens": raw_regions.float(),
            "aligned_visual_tokens": aligned_regions,
            "aligned_text_tokens": aligned_text,
            "text_valid": text_inputs["attention_mask"].bool(),
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
        }

    def fuse(self, input_ids, attention_mask, visual_tokens, visual_valid):
        """Run BLIP's pretrained text-vision cross-attention after selection.

        Backbone parameters stay frozen, but gradients are allowed with respect
        to visual_tokens so the soft training mask receives the task signal.
        """
        model_dtype = next(self.model.parameters()).dtype
        output = self.model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=visual_tokens.to(model_dtype),
            encoder_attention_mask=visual_valid.long(),
            return_dict=True,
        )
        return output.last_hidden_state[:, 0].float()


def normalized_region_positions(batch, regions, device, dtype):
    side = int(round(math.sqrt(regions)))
    if side * side != regions:
        raise ValueError("region sequence must be a square grid")
    axis = torch.linspace(-1.0, 1.0, side, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    xy = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
    return xy[None].expand(batch, -1, -1)


class AlignedCrossModalSelector(nn.Module):
    """Decision selector operating in BLIP's pretrained aligned space."""

    def __init__(
        self,
        aligned_dim,
        hidden=256,
        latent_dim=64,
        position_dim=32,
        layers=4,
        heads=8,
    ):
        super().__init__()
        if aligned_dim % heads != 0:
            raise ValueError("aligned_dim must be divisible by heads")
        if hidden % heads != 0:
            raise ValueError("hidden must be divisible by heads")
        self.latent_dim = int(latent_dim)
        self.text_to_visual = nn.MultiheadAttention(
            aligned_dim, heads, batch_first=True
        )
        self.position_proj = nn.Sequential(
            nn.Linear(2, position_dim),
            nn.GELU(),
            nn.Linear(position_dim, position_dim),
        )
        input_dim = aligned_dim * 2 + position_dim + 1
        self.in_proj = nn.Linear(input_dim, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
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
        score_dim = latent_dim * 2 + aligned_dim + position_dim + 1
        self.score_head = nn.Sequential(
            nn.LayerNorm(score_dim),
            nn.Linear(score_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )
        value_dim = latent_dim * 2 + aligned_dim + 1
        self.set_value_head = nn.Sequential(
            nn.LayerNorm(value_dim),
            nn.Linear(value_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    @staticmethod
    def _weighted_pool(tokens, weight):
        weight = weight.float()
        return (tokens * weight[..., None]).sum(1) / weight.sum(
            1, keepdim=True
        ).clamp_min(1e-6)

    @staticmethod
    def _budget_tokens(budget, batch, regions):
        return budget.reshape(batch, 1, 1).expand(batch, regions, 1)

    def forward(
        self,
        aligned_visual,
        aligned_text,
        text_valid,
        position_xy,
        visual_valid,
        budget,
    ):
        batch, regions, _ = aligned_visual.shape
        cross_context, cross_attention = self.text_to_visual(
            query=aligned_visual,
            key=aligned_text,
            value=aligned_text,
            key_padding_mask=~text_valid,
            need_weights=True,
            average_attn_weights=True,
        )
        position = self.position_proj(position_xy)
        budget_tokens = self._budget_tokens(budget, batch, regions)
        selector_input = torch.cat(
            [aligned_visual, cross_context, position, budget_tokens], dim=-1
        )
        hidden = self.norm(
            self.encoder(
                self.in_proj(selector_input),
                src_key_padding_mask=~visual_valid,
            )
        )
        z = self.latent_head(hidden)
        query = self.pool_query.expand(batch, -1, -1)
        global_context, _ = self.pool_attn(
            query,
            z,
            z,
            key_padding_mask=~visual_valid,
            need_weights=False,
        )
        global_tokens = global_context.expand(batch, regions, -1)
        score_input = torch.cat(
            [z, global_tokens, cross_context, position, budget_tokens], dim=-1
        )
        logits = self.score_head(score_input).squeeze(-1)
        logits = logits.masked_fill(~visual_valid, -1e9)
        task_context = self._weighted_pool(aligned_text, text_valid.float())
        return {
            "z": z,
            "logits": logits,
            "task_context": task_context,
            "cross_attention": cross_attention,
        }

    def predict_set_value(
        self, z, task_context, mask, valid, budget
    ):
        selected = self._weighted_pool(z, mask * valid.float())
        global_context = self._weighted_pool(z, valid.float())
        value_input = torch.cat(
            [
                selected,
                global_context,
                task_context,
                budget.reshape(-1, 1),
            ],
            dim=-1,
        )
        return self.set_value_head(value_input).squeeze(-1)


class PathDecisionHead(nn.Module):
    """Predict candidate-path logits from the fused VLM representation."""

    def __init__(self, text_hidden_dim, n_paths, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(text_hidden_dim),
            nn.Linear(text_hidden_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_paths),
        )

    def forward(self, fused_task_state):
        return self.net(fused_task_state)


def gather_selected_visual_tokens(raw_visual_tokens, logits, valid, k):
    if (valid.sum(-1) < int(k)).any():
        raise ValueError("every sample must have at least k valid regions")
    indices = torch.topk(
        logits.masked_fill(~valid, -1e9), k=int(k), dim=-1
    ).indices
    gather_index = indices[..., None].expand(
        -1, -1, raw_visual_tokens.shape[-1]
    )
    selected = torch.gather(raw_visual_tokens, 1, gather_index)
    return selected, indices
