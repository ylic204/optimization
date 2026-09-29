import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalized_region_positions(batch, regions, device, dtype):
    side = int(round(math.sqrt(regions)))
    if side * side != regions:
        raise ValueError("region sequence must be a square grid")
    axis = torch.linspace(-1.0, 1.0, side, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    xy = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
    return xy[None].expand(batch, -1, -1)


def topk_region_indices(logits, valid, k):
    """Return spatially ordered Top-K indices for Qwen's packed ViT stream."""
    if (valid.sum(-1) < int(k)).any():
        raise ValueError("every sample must have at least k valid regions")
    indices = torch.topk(
        logits.masked_fill(~valid, -1e9), k=int(k), dim=-1
    ).indices
    return indices.sort(dim=-1).values


def mask_to_region_indices(mask):
    """Convert a fixed-cardinality binary mask to sorted indices."""
    counts = mask.bool().sum(-1)
    if not torch.equal(counts, counts[:1].expand_as(counts)):
        raise ValueError("all samples must select the same number of regions")
    return torch.stack(
        [torch.where(row.bool())[0] for row in mask], dim=0
    ).sort(dim=-1).values


class FrozenQwen3VLPrunableBackbone(nn.Module):
    """Frozen Qwen3-VL with group-wise pruning inside the vision encoder.

    Qwen3-VL orders each ``spatial_merge_size x spatial_merge_size`` patch
    group contiguously.  The selector scores those merge groups after the
    first ``prune_layer`` vision blocks.  A hard path keeps only K groups for
    the remaining vision blocks, the visual merger, DeepStack and the LLM.
    A soft path keeps the rectangular stream but gates every group, allowing
    the downstream task loss to train the selector.

    This adapter targets the official Transformers Qwen3-VL implementations
    from 4.57+ and 5.x.  It deliberately accesses the public model components
    (``visual`` and ``language_model``) because native arbitrary-token pruning
    is not exposed by ``generate``.
    """

    def __init__(
        self,
        model_name_or_path,
        region_grid=6,
        prune_layer=4,
        max_text_length=64,
        local_files_only=False,
        load_4bit=False,
        attn_implementation="sdpa",
        device=None,
    ):
        super().__init__()
        try:
            from transformers import (
                AutoProcessor,
                BitsAndBytesConfig,
                Qwen3VLForConditionalGeneration,
            )
        except ImportError as exc:
            raise ImportError(
                "V7.9 requires Transformers with Qwen3-VL support "
                "(transformers>=4.57)"
            ) from exc

        device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
        )
        load_kwargs = {
            "local_files_only": local_files_only,
            "attn_implementation": attn_implementation,
        }
        if load_4bit:
            if device.type != "cuda":
                raise ValueError("4-bit Qwen3-VL loading requires CUDA")
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            load_kwargs["device_map"] = {"": device.index or 0}
        else:
            load_kwargs["torch_dtype"] = (
                torch.bfloat16 if device.type == "cuda" else torch.float32
            )

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            **load_kwargs,
        )
        if not load_4bit:
            self.model.to(device)
        self.model.eval().requires_grad_(False)
        self.model.config.use_cache = False

        self.region_grid = int(region_grid)
        self.max_text_length = int(max_text_length)
        self.prune_layer = int(prune_layer)
        self.load_4bit = bool(load_4bit)
        vision_config = self.model.config.vision_config
        text_config = self.model.config.text_config
        self.spatial_merge_size = int(vision_config.spatial_merge_size)
        self.merge_unit = self.spatial_merge_size**2
        self.visual_hidden_dim = int(vision_config.hidden_size)
        self.text_hidden_dim = int(text_config.hidden_size)
        if (self.visual_hidden_dim, self.text_hidden_dim) != (1152, 4096):
            raise ValueError(
                "V7.9 is configured for Qwen3-VL-8B "
                "(vision hidden=1152, text hidden=4096), but the checkpoint "
                f"reports ({self.visual_hidden_dim}, {self.text_hidden_dim})."
            )
        self.target_image_side = (
            self.region_grid
            * self.spatial_merge_size
            * int(vision_config.patch_size)
        )
        self.target_pixels = self.target_image_side**2

        depth = len(self.visual.blocks)
        if not 0 <= self.prune_layer < depth:
            raise ValueError(
                f"prune_layer must be in [0, {depth - 1}], got {prune_layer}"
            )
        deepstack = list(getattr(self.visual, "deepstack_visual_indexes", []))
        if deepstack and self.prune_layer > min(deepstack):
            raise ValueError(
                "prune_layer must not be after the first DeepStack tap; "
                f"got {prune_layer}, first tap={min(deepstack)}"
            )

    @property
    def core(self):
        return self.model.model

    @property
    def visual(self):
        return self.core.visual

    @property
    def language_model(self):
        return self.core.language_model

    @property
    def device(self):
        return next(self.model.parameters()).device

    def train(self, mode=True):
        super().train(False)
        self.model.eval()
        return self

    def _process_images(self, images):
        image_list = [image.detach().cpu() for image in images]
        batch = self.processor.image_processor(
            images=image_list,
            do_rescale=False,
            min_pixels=self.target_pixels,
            max_pixels=self.target_pixels,
            return_tensors="pt",
        )
        pixel_values = batch["pixel_values"].to(self.device)
        image_grid_thw = batch["image_grid_thw"].to(self.device)
        return pixel_values, image_grid_thw

    def _process_text(self, texts):
        tokens = self.processor.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            add_special_tokens=True,
            return_tensors="pt",
        )
        return {key: value.to(self.device) for key, value in tokens.items()}

    def _validate_grid(self, image_grid_thw, batch_size):
        if image_grid_thw.shape != (batch_size, 3):
            raise RuntimeError(
                "V7.9 expects exactly one image per sample; got image_grid_thw "
                f"shape {tuple(image_grid_thw.shape)}"
            )
        expected_hw = self.region_grid * self.spatial_merge_size
        expected = torch.tensor(
            [1, expected_hw, expected_hw],
            device=image_grid_thw.device,
            dtype=image_grid_thw.dtype,
        )
        if not torch.all(image_grid_thw == expected):
            raise RuntimeError(
                "Qwen image preprocessing did not produce the controlled "
                f"{self.region_grid}x{self.region_grid} merge grid. Expected "
                f"[1,{expected_hw},{expected_hw}] per image, got "
                f"{image_grid_thw.tolist()}."
            )

    def _prepare_vision_stream_legacy(self, pixel_values, grid_thw):
        """Transformers 4.57 Qwen3-VL vision stem."""
        hidden = self.visual.patch_embed(pixel_values.type(self.visual.dtype))
        hidden = hidden + self.visual.fast_pos_embed_interpolate(grid_thw)
        rotary = self.visual.rot_pos_emb(grid_thw)
        rotary = rotary.reshape(hidden.shape[0], -1)
        emb = torch.cat((rotary, rotary), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        return hidden.reshape(hidden.shape[0], -1), position_embeddings, cu_seqlens, None

    def _prepare_vision_stream_v5(self, pixel_values, grid_thw):
        """Transformers 5.x Qwen3-VL vision stem."""
        from transformers.vision_utils import (
            get_vision_attention_seqlens,
            get_vision_interpolation_indices_and_weights,
            get_vision_position_ids,
        )

        kwargs = {}
        indices, weights = get_vision_interpolation_indices_and_weights(
            grid_thw,
            num_grid_per_side=self.visual.num_grid_per_side,
            mode=self.visual.interpolation_mode,
            align_corners=self.visual.interpolation_align_corners,
            spatial_merge_size=self.spatial_merge_size,
            kwargs=kwargs,
        )
        position_ids = get_vision_position_ids(
            grid_thw, self.spatial_merge_size, kwargs=kwargs
        )
        cu_seqlens, max_seqlen = get_vision_attention_seqlens(
            grid_thw, self.visual.config, kwargs=kwargs
        )
        hidden = self.visual.patch_embed(pixel_values.type(self.visual.dtype))
        position = (
            self.visual.pos_embed(indices) * weights[:, :, None]
        ).sum(1)
        hidden = hidden + position.to(hidden.dtype)
        position_embeddings = self.visual.rotary_pos_emb(hidden, position_ids)
        return hidden.reshape(hidden.shape[0], -1), position_embeddings, cu_seqlens, max_seqlen

    def _prepare_vision_stream(self, pixel_values, grid_thw):
        if hasattr(self.visual, "fast_pos_embed_interpolate"):
            return self._prepare_vision_stream_legacy(pixel_values, grid_thw)
        return self._prepare_vision_stream_v5(pixel_values, grid_thw)

    @staticmethod
    def _run_block(block, hidden, cu_seqlens, position_embeddings, max_seqlen):
        kwargs = {
            "cu_seqlens": cu_seqlens,
            "position_embeddings": position_embeddings,
        }
        if max_seqlen is not None:
            kwargs["max_seqlen"] = max_seqlen
        return block(hidden, **kwargs)

    @torch.no_grad()
    def encode_inputs(self, images, texts):
        """Run preprocessing and only the pre-pruning frozen layers."""
        batch_size = images.shape[0]
        pixel_values, grid_thw = self._process_images(images)
        self._validate_grid(grid_thw, batch_size)
        text = self._process_text(texts)
        hidden, position_embeddings, cu_seqlens, max_seqlen = (
            self._prepare_vision_stream(pixel_values, grid_thw)
        )
        for block in self.visual.blocks[: self.prune_layer]:
            hidden = self._run_block(
                block, hidden, cu_seqlens, position_embeddings, max_seqlen
            )

        regions = self.region_grid**2
        hidden_groups = hidden.reshape(
            batch_size, regions, self.merge_unit, self.visual_hidden_dim
        )
        grouped_positions = tuple(
            item.reshape(batch_size, regions, self.merge_unit, -1)
            for item in position_embeddings
        )
        text_embeddings = self.language_model.embed_tokens(text["input_ids"])
        return {
            "region_visual_tokens": hidden_groups.mean(2).float(),
            "text_tokens": text_embeddings.float(),
            "text_valid": text["attention_mask"].bool(),
            "input_ids": text["input_ids"],
            "attention_mask": text["attention_mask"],
            "vision_state": {
                "hidden_groups": hidden_groups,
                "position_groups": grouped_positions,
            },
        }

    @staticmethod
    def _gather_groups(groups, indices):
        gather_index = indices[..., None, None].expand(
            -1, -1, groups.shape[2], groups.shape[3]
        )
        return torch.gather(groups, 1, gather_index)

    def continue_vision(
        self,
        vision_state,
        region_mask=None,
        selected_indices=None,
    ):
        """Run the post-pruning ViT blocks and merger.

        Exactly one of ``region_mask`` (soft differentiable path) and
        ``selected_indices`` (hard compute-saving path) must be supplied.
        """
        if (region_mask is None) == (selected_indices is None):
            raise ValueError(
                "provide exactly one of region_mask or selected_indices"
            )
        hidden_groups = vision_state["hidden_groups"]
        position_groups = vision_state["position_groups"]
        batch_size = hidden_groups.shape[0]

        raw_gate = None
        if selected_indices is not None:
            hidden_groups = self._gather_groups(
                hidden_groups, selected_indices
            )
            position_groups = tuple(
                self._gather_groups(item, selected_indices)
                for item in position_groups
            )
            token_indices = selected_indices
        else:
            raw_gate = region_mask.to(hidden_groups.dtype)[..., None, None]
            hidden_groups = hidden_groups * raw_gate
            token_indices = torch.arange(
                hidden_groups.shape[1], device=hidden_groups.device
            )[None].expand(batch_size, -1)

        regions = hidden_groups.shape[1]
        hidden = hidden_groups.reshape(
            batch_size * regions * self.merge_unit, self.visual_hidden_dim
        )
        position_embeddings = tuple(
            item.reshape(batch_size * regions * self.merge_unit, -1)
            for item in position_groups
        )
        per_image = regions * self.merge_unit
        cu_seqlens = torch.arange(
            0,
            (batch_size + 1) * per_image,
            per_image,
            dtype=torch.int32,
            device=hidden.device,
        )
        max_seqlen = per_image if not hasattr(
            self.visual, "fast_pos_embed_interpolate"
        ) else None

        deepstack = []
        for layer_index in range(self.prune_layer, len(self.visual.blocks)):
            hidden = self._run_block(
                self.visual.blocks[layer_index],
                hidden,
                cu_seqlens,
                position_embeddings,
                max_seqlen,
            )
            if raw_gate is not None:
                hidden = hidden.reshape(
                    batch_size, regions, self.merge_unit, -1
                )
                hidden = hidden * raw_gate
                hidden = hidden.reshape(
                    batch_size * regions * self.merge_unit, -1
                )
            if layer_index in self.visual.deepstack_visual_indexes:
                merger_index = self.visual.deepstack_visual_indexes.index(
                    layer_index
                )
                merged = self.visual.deepstack_merger_list[merger_index](hidden)
                deepstack.append(
                    merged.reshape(batch_size, regions, self.text_hidden_dim)
                )

        merged = self.visual.merger(hidden).reshape(
            batch_size, regions, self.text_hidden_dim
        )
        return {
            "visual_tokens": merged,
            "deepstack_tokens": deepstack,
            "region_indices": token_indices,
        }

    def fuse(self, input_ids, attention_mask, visual_output):
        """Fuse selected visual tokens and task text with Qwen3-VL's LLM."""
        visual_tokens = visual_output["visual_tokens"]
        batch_size, visual_length, _ = visual_tokens.shape
        embed_tokens = self.language_model.embed_tokens
        model_dtype = embed_tokens.weight.dtype
        text_embeddings = embed_tokens(input_ids)
        vision_start = embed_tokens(
            torch.full(
                (batch_size, 1),
                int(self.model.config.vision_start_token_id),
                dtype=torch.long,
                device=input_ids.device,
            )
        )
        vision_end = embed_tokens(
            torch.full(
                (batch_size, 1),
                int(self.model.config.vision_end_token_id),
                dtype=torch.long,
                device=input_ids.device,
            )
        )
        inputs_embeds = torch.cat(
            [
                vision_start,
                visual_tokens.to(model_dtype),
                vision_end,
                text_embeddings,
            ],
            dim=1,
        )
        prefix_valid = torch.ones(
            batch_size,
            visual_length + 2,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        fused_attention_mask = torch.cat(
            [prefix_valid, attention_mask], dim=1
        )
        visual_pos_masks = torch.zeros_like(
            fused_attention_mask, dtype=torch.bool
        )
        visual_pos_masks[:, 1 : visual_length + 1] = True
        deepstack = [
            item.reshape(-1, item.shape[-1]).to(model_dtype)
            for item in visual_output["deepstack_tokens"]
        ]
        output = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=fused_attention_mask,
            use_cache=False,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack or None,
            return_dict=True,
        )
        last_text_index = (
            visual_length + 2 + attention_mask.sum(-1) - 1
        ).long()
        batch_index = torch.arange(batch_size, device=input_ids.device)
        return output.last_hidden_state[
            batch_index, last_text_index
        ].float()


class DecisionRelatedQwenSelector(nn.Module):
    """Learn decision-related visual/text alignment without graph features."""

    def __init__(
        self,
        visual_dim,
        text_dim,
        aligned_dim=256,
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
        self.visual_align = nn.Sequential(
            nn.LayerNorm(visual_dim), nn.Linear(visual_dim, aligned_dim)
        )
        self.text_align = nn.Sequential(
            nn.LayerNorm(text_dim), nn.Linear(text_dim, aligned_dim)
        )
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
        visual_tokens,
        text_tokens,
        text_valid,
        position_xy,
        visual_valid,
        budget,
    ):
        aligned_visual = F.normalize(
            self.visual_align(visual_tokens), dim=-1
        )
        aligned_text = F.normalize(self.text_align(text_tokens), dim=-1)
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
            "aligned_visual": aligned_visual,
            "aligned_text": aligned_text,
        }

    def predict_set_value(self, z, task_context, mask, valid, budget):
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

    def pool_decision_latent(self, z, mask, valid):
        """Pool the information available under ``mask`` into one decision code."""
        pooled = self._weighted_pool(z, mask * valid.float())
        return F.normalize(pooled, dim=-1)


class PathDecisionHead(nn.Module):
    """Map Qwen3-VL's fused task state to candidate-path logits."""

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
