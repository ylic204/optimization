"""Qwen3-VL perception model that predicts solver coefficients."""

import torch
import torch.nn as nn

from mapped_vlm_planner_v82 import CandidateTrajectoryEncoder
from optimization_spec_v83 import METRIC_NAMES
from qwen3vl_selector_v79 import (
    DecisionRelatedQwenSelector,
    FrozenQwen3VLPrunableBackbone,
    normalized_region_positions,
    topk_region_indices,
)


class CandidateMetricHead(nn.Module):
    """Predict eight normalized cost coefficients for every candidate path."""

    def __init__(self, vlm_dim=4096, hidden=256):
        super().__init__()
        self.trajectory_encoder = CandidateTrajectoryEncoder(8, hidden)
        self.vlm_encoder = nn.Sequential(
            nn.LayerNorm(vlm_dim), nn.Linear(vlm_dim, hidden), nn.GELU()
        )
        self.state_encoder = nn.Sequential(
            nn.LayerNorm(12), nn.Linear(12, hidden), nn.GELU()
        )
        self.source_embedding = nn.Embedding(2, hidden)
        self.metric_decoder = nn.Sequential(
            nn.LayerNorm(hidden * 4),
            nn.Linear(hidden * 4, hidden),
            nn.GELU(),
            nn.Linear(hidden, len(METRIC_NAMES)),
            nn.Sigmoid(),
        )

    def forward(
        self,
        fused_vlm,
        trajectories,
        trajectory_features,
        goal_state,
        ego_state,
        source_id,
    ):
        # One embedding is produced for each of the eleven map candidates.
        candidate = self.trajectory_encoder(trajectories, trajectory_features)

        # The fused Qwen vector contains image evidence conditioned on task text.
        vlm = self.vlm_encoder(fused_vlm)[:, None].expand_as(candidate)

        # Goal and ego state describe the current local planning problem.
        state = self.state_encoder(torch.cat([goal_state, ego_state], dim=-1))
        state = state[:, None].expand_as(candidate)

        # The domain embedding separates road-driving and indoor-navigation scales.
        source = self.source_embedding(source_id)[:, None].expand_as(candidate)

        decoder_input = torch.cat([candidate, vlm, state, source], dim=-1)
        return self.metric_decoder(decoder_input)


class MappedVLMOptimizerV83(nn.Module):
    """Connect token selection, Qwen fusion and metric prediction."""

    def __init__(self, backbone, selector, metric_head):
        super().__init__()
        self.backbone = backbone
        self.selector = selector
        self.metric_head = metric_head

    def encode_and_select(self, images, texts, budget):
        # Frozen Qwen preprocessing and the early vision blocks produce 81 regions.
        encoded = self.backbone.encode_inputs(images, texts)
        visual = encoded["region_visual_tokens"]
        batch, regions, _ = visual.shape
        valid = torch.ones(batch, regions, dtype=torch.bool, device=visual.device)
        positions = normalized_region_positions(
            batch, regions, visual.device, visual.dtype
        )

        # The selector aligns each visual region with the task instruction.
        selection = self.selector(
            visual_tokens=visual,
            text_tokens=encoded["text_tokens"],
            text_valid=encoded["text_valid"],
            position_xy=positions,
            visual_valid=valid,
            budget=budget,
        )
        return encoded, selection, valid

    def _predict(self, encoded, visual_output, batch):
        # Qwen's LLM fuses the retained visual tokens with the task text.
        fused = self.backbone.fuse(
            encoded["input_ids"], encoded["attention_mask"], visual_output
        )
        return self.metric_head(
            fused,
            batch["candidate_trajectories"],
            batch["candidate_features"],
            batch["goal_state"],
            batch["ego_state"],
            batch["source_id"],
        )

    def predict_with_mask(self, encoded, region_mask, batch):
        # Training keeps a rectangular stream and differentiates through the mask.
        visual_output = self.backbone.continue_vision(
            encoded["vision_state"], region_mask=region_mask
        )
        return self._predict(encoded, visual_output, batch)

    def predict_with_topk(self, encoded, region_logits, valid, k, batch):
        # Inference physically removes unselected regions before later Qwen layers.
        indices = topk_region_indices(region_logits, valid, k)
        visual_output = self.backbone.continue_vision(
            encoded["vision_state"], selected_indices=indices
        )
        return self._predict(encoded, visual_output, batch), indices


def build_model(args, device):
    backbone = FrozenQwen3VLPrunableBackbone(
        args.vlm_model,
        region_grid=args.region_grid,
        prune_layer=args.prune_layer,
        max_text_length=args.max_text_length,
        local_files_only=True,
        load_4bit=args.load_4bit,
        attn_implementation="sdpa",
        device=device,
    )
    selector = DecisionRelatedQwenSelector(
        visual_dim=backbone.visual_hidden_dim,
        text_dim=backbone.text_hidden_dim,
        aligned_dim=args.hidden,
        hidden=args.hidden,
        latent_dim=args.latent_dim,
        position_dim=32,
        layers=4,
        heads=8,
    ).to(device)
    metric_head = CandidateMetricHead(
        vlm_dim=backbone.text_hidden_dim, hidden=args.hidden
    ).to(device)
    return MappedVLMOptimizerV83(backbone, selector, metric_head)
