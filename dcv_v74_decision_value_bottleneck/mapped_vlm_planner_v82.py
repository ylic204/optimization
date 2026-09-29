"""Qwen3-VL token selector plus candidate-conditioned path planner.

There are intentionally no Flow Matching or neurodynamic modules in V8.2.
"""

import torch
import torch.nn as nn

from qwen3vl_selector_v79 import (
    DecisionRelatedQwenSelector,
    FrozenQwen3VLPrunableBackbone,
    normalized_region_positions,
    topk_region_indices,
)


class CandidateTrajectoryEncoder(nn.Module):
    """Encode each candidate from deployable ego-frame geometry."""

    def __init__(self, feature_dim=8, hidden=256):
        super().__init__()
        self.point_net = nn.Sequential(
            nn.Linear(4, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, hidden),
            nn.GELU(),
        )
        self.feature_net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
        )
        self.out = nn.Sequential(
            nn.LayerNorm(hidden * 3),
            nn.Linear(hidden * 3, hidden),
            nn.GELU(),
        )

    def forward(self, trajectories, features):
        if trajectories.ndim != 4 or trajectories.shape[-1] != 3:
            raise ValueError("trajectories must have shape [B,P,H,3]")
        xy = trajectories[..., :2]
        yaw = trajectories[..., 2]
        point_input = torch.cat(
            [xy, torch.sin(yaw)[..., None], torch.cos(yaw)[..., None]],
            dim=-1,
        )
        point = self.point_net(point_input)
        mean_pool = point.mean(dim=2)
        max_pool = point.max(dim=2).values
        geometry = self.feature_net(features)
        return self.out(torch.cat([mean_pool, max_pool, geometry], dim=-1))


class CandidateConditionedPlanningHead(nn.Module):
    """Score a variable path by its geometry and the pruned VLM state.

    Candidate costs and optimal indices are deliberately absent from the
    signature: they are labels used by the training loss, not model inputs.
    """

    def __init__(
        self,
        vlm_dim,
        candidate_feature_dim=8,
        ego_dim=8,
        goal_dim=4,
        hidden=256,
        num_sources=2,
    ):
        super().__init__()
        self.trajectory_encoder = CandidateTrajectoryEncoder(
            candidate_feature_dim, hidden
        )
        self.vlm_proj = nn.Sequential(
            nn.LayerNorm(vlm_dim), nn.Linear(vlm_dim, hidden), nn.GELU()
        )
        self.state_proj = nn.Sequential(
            nn.LayerNorm(ego_dim + goal_dim),
            nn.Linear(ego_dim + goal_dim, hidden),
            nn.GELU(),
        )
        self.source_embedding = nn.Embedding(num_sources, hidden)
        self.context = nn.Sequential(
            nn.LayerNorm(hidden * 3),
            nn.Linear(hidden * 3, hidden),
            nn.GELU(),
        )
        self.score = nn.Sequential(
            nn.LayerNorm(hidden * 3),
            nn.Linear(hidden * 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(
        self,
        fused_task_state,
        candidate_trajectories,
        candidate_features,
        goal_state,
        ego_state,
        source_id,
    ):
        candidate = self.trajectory_encoder(
            candidate_trajectories, candidate_features
        )
        vlm = self.vlm_proj(fused_task_state)
        state = self.state_proj(torch.cat([goal_state, ego_state], dim=-1))
        source = self.source_embedding(source_id)
        context = self.context(torch.cat([vlm, state, source], dim=-1))
        expanded = context[:, None, :].expand_as(candidate)
        score_input = torch.cat(
            [candidate, expanded, candidate * expanded], dim=-1
        )
        return self.score(score_input).squeeze(-1)


class MappedVLMPlannerV82(nn.Module):
    """Thin orchestration wrapper for the V8.2 planner."""

    def __init__(self, backbone, selector, planning_head):
        super().__init__()
        if not isinstance(backbone, FrozenQwen3VLPrunableBackbone):
            raise TypeError("backbone must be FrozenQwen3VLPrunableBackbone")
        self.backbone = backbone
        self.selector = selector
        self.planning_head = planning_head

    def selector_forward(self, images, task_text, budget):
        encoded = self.backbone.encode_inputs(images, task_text)
        visual = encoded["region_visual_tokens"]
        batch, regions, _ = visual.shape
        valid = torch.ones(
            batch, regions, dtype=torch.bool, device=visual.device
        )
        position = normalized_region_positions(
            batch, regions, visual.device, visual.dtype
        )
        selected = self.selector(
            visual_tokens=visual,
            text_tokens=encoded["text_tokens"],
            text_valid=encoded["text_valid"],
            position_xy=position,
            visual_valid=valid,
            budget=budget,
        )
        return encoded, selected, valid

    def logits_from_mask(self, encoded, region_mask, batch):
        visual = self.backbone.continue_vision(
            encoded["vision_state"], region_mask=region_mask
        )
        fused = self.backbone.fuse(
            encoded["input_ids"], encoded["attention_mask"], visual
        )
        return self.planning_head(
            fused,
            batch["candidate_trajectories"],
            batch["candidate_features"],
            batch["goal_state"],
            batch["ego_state"],
            batch["source_id"],
        )

    def logits_from_topk(self, encoded, selector_logits, valid, k, batch):
        indices = topk_region_indices(selector_logits, valid, k)
        visual = self.backbone.continue_vision(
            encoded["vision_state"], selected_indices=indices
        )
        fused = self.backbone.fuse(
            encoded["input_ids"], encoded["attention_mask"], visual
        )
        logits = self.planning_head(
            fused,
            batch["candidate_trajectories"],
            batch["candidate_features"],
            batch["goal_state"],
            batch["ego_state"],
            batch["source_id"],
        )
        return logits, indices


def build_mapped_vlm_planner(
    model_name_or_path,
    device,
    region_grid=9,
    prune_layer=4,
    max_text_length=96,
    load_4bit=False,
    local_files_only=False,
    attn_implementation="sdpa",
    aligned_dim=256,
    hidden=256,
    latent_dim=64,
    position_dim=32,
    selector_layers=4,
    selector_heads=8,
):
    backbone = FrozenQwen3VLPrunableBackbone(
        model_name_or_path,
        region_grid=region_grid,
        prune_layer=prune_layer,
        max_text_length=max_text_length,
        local_files_only=local_files_only,
        load_4bit=load_4bit,
        attn_implementation=attn_implementation,
        device=device,
    )
    selector = DecisionRelatedQwenSelector(
        visual_dim=backbone.visual_hidden_dim,
        text_dim=backbone.text_hidden_dim,
        aligned_dim=aligned_dim,
        hidden=hidden,
        latent_dim=latent_dim,
        position_dim=position_dim,
        layers=selector_layers,
        heads=selector_heads,
    ).to(device)
    head = CandidateConditionedPlanningHead(
        vlm_dim=backbone.text_hidden_dim, hidden=hidden
    ).to(device)
    return MappedVLMPlannerV82(backbone, selector, head)
