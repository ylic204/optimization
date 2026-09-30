"""V8.5: flow-matched, budget-constrained neurodynamic token selection.

Flow Matching and neurodynamics are two views of the same mask dynamics:
the former supervises the local velocity, while the latter executes a fixed
number of projected updates.  No trajectory-Flow -> ND cascade is used.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mapped_vlm_optimizer_v83 import CandidateMetricHead
from qwen3vl_selector_v79 import (
    DecisionRelatedQwenSelector,
    FrozenQwen3VLPrunableBackbone,
    normalized_region_positions,
    topk_region_indices,
)


def _batch_mass(mass, reference):
    mass = torch.as_tensor(mass, device=reference.device, dtype=reference.dtype)
    if mass.ndim == 0:
        mass = mass.expand(reference.shape[0])
    return mass


def project_capped_simplex(values, valid, mass, iterations=40):
    """Project each row onto {0<=w<=1, sum(w)=mass} over valid entries.

    The bisection threshold is treated as a piecewise-constant projection
    parameter.  Gradients still pass through the active coordinates of
    ``values``, matching the straight-through constrained dynamics used here.
    """
    valid_f = valid.to(values.dtype)
    mass = _batch_mass(mass, values)
    valid_count = valid_f.sum(-1)
    if bool((valid_count == 0).any()):
        raise ValueError("each projection row must contain a valid region")
    if bool(((mass < 0.0) | (mass > valid_count)).any()):
        raise ValueError("projection mass must be within the valid region count")

    detached = values.detach().masked_fill(~valid, 0.0)
    with torch.no_grad():
        lower = detached.masked_fill(~valid, torch.inf).min(-1).values - 1.0
        upper = detached.masked_fill(~valid, -torch.inf).max(-1).values
        for _ in range(int(iterations)):
            threshold = 0.5 * (lower + upper)
            projected = (
                (detached - threshold[:, None]).clamp(0.0, 1.0) * valid_f
            )
            current_mass = projected.sum(-1)
            lower = torch.where(current_mass > mass, threshold, lower)
            upper = torch.where(current_mass > mass, upper, threshold)
        threshold = 0.5 * (lower + upper)
    return (values - threshold[:, None]).clamp(0.0, 1.0) * valid_f


def fixed_mass_mask(logits, valid, mass, temperature=0.35):
    """Smooth task-conditioned initialization with an exact relaxed mass."""
    temperature = max(float(temperature), 1e-4)
    scaled = logits / temperature
    return project_capped_simplex(torch.sigmoid(scaled), valid, mass)


def hard_topk_mask(scores, valid, k):
    indices = topk_region_indices(scores, valid, int(k))
    return torch.zeros_like(scores).scatter(1, indices, 1.0)


def endpoint_improvement_loss(initial_values, final_values, margin=0.0):
    """Penalize a final task value that fails to improve on its initial value.

    The initial value is detached so optimization cannot satisfy the hinge by
    deliberately making the baseline worse.
    """
    if float(margin) < 0.0:
        raise ValueError("improvement margin must be non-negative")
    return F.relu(final_values - initial_values.detach() + float(margin))


class MaskGradientFlowND(nn.Module):
    """Learn a projected visual-mask velocity and execute fixed-step ND."""

    def __init__(
        self,
        latent_dim=64,
        task_dim=256,
        hidden=256,
        layers=3,
        heads=8,
        steps=4,
        initial_step_size=0.25,
    ):
        super().__init__()
        if hidden % heads != 0:
            raise ValueError("hidden must be divisible by heads")
        self.steps = int(steps)
        self.region = nn.Sequential(
            nn.LayerNorm(latent_dim), nn.Linear(latent_dim, hidden)
        )
        self.task = nn.Sequential(
            nn.LayerNorm(task_dim), nn.Linear(task_dim, hidden)
        )
        self.position = nn.Sequential(
            nn.Linear(2, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.state = nn.Sequential(
            nn.Linear(2, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
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
        self.velocity_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        inverse_softplus = math.log(math.expm1(float(initial_step_size)))
        self.step_logits = nn.Parameter(
            torch.full((max(1, self.steps),), inverse_softplus)
        )

    def step_size(self, step):
        if self.steps <= 0:
            return self.step_logits.new_tensor(0.0)
        return F.softplus(self.step_logits[int(step)]) + 1e-4

    def velocity(self, mask, time, z, task_context, positions, valid):
        batch, regions = mask.shape
        time_token = time.reshape(batch, 1, 1).expand(batch, regions, 1)
        state_token = torch.cat([mask[..., None], time_token], dim=-1)
        hidden = (
            self.region(z)
            + self.task(task_context)[:, None]
            + self.position(positions)
            + self.state(state_token)
        )
        hidden = self.encoder(hidden, src_key_padding_mask=~valid)
        raw = torch.tanh(self.velocity_head(hidden).squeeze(-1))
        valid_f = valid.to(raw.dtype)
        mean = (raw * valid_f).sum(-1, keepdim=True) / valid_f.sum(
            -1, keepdim=True
        ).clamp_min(1.0)
        return (raw - mean) * valid_f

    def rollout(self, initial_mask, z, task_context, positions, valid, mass):
        state = project_capped_simplex(initial_mask, valid, mass)
        states = [state]
        velocities = []
        for step in range(self.steps):
            time = torch.full(
                (state.shape[0],),
                step / max(1, self.steps),
                device=state.device,
                dtype=state.dtype,
            )
            velocity = self.velocity(
                state, time, z, task_context, positions, valid
            )
            state = project_capped_simplex(
                state + self.step_size(step) * velocity,
                valid,
                mass,
            )
            velocities.append(velocity)
            states.append(state)
        return state, states, velocities


class GradientFlowSelectorV85(nn.Module):
    def __init__(self, backbone, selector, metric_head, mask_flow):
        super().__init__()
        self.backbone = backbone
        self.selector = selector
        self.metric_head = metric_head
        self.mask_flow = mask_flow

    def encode_and_select(self, images, texts, budget):
        encoded = self.backbone.encode_inputs(images, texts)
        visual = encoded["region_visual_tokens"]
        batch, regions, _ = visual.shape
        valid = torch.ones(batch, regions, dtype=torch.bool, device=visual.device)
        positions = normalized_region_positions(
            batch, regions, visual.device, visual.dtype
        )
        selection = self.selector(
            visual_tokens=visual,
            text_tokens=encoded["text_tokens"],
            text_valid=encoded["text_valid"],
            position_xy=positions,
            visual_valid=valid,
            budget=budget,
        )
        return encoded, selection, positions, valid

    def fuse_mask(self, encoded, mask):
        visual = self.backbone.continue_vision(
            encoded["vision_state"], region_mask=mask
        )
        return self.backbone.fuse(
            encoded["input_ids"], encoded["attention_mask"], visual
        )

    def fuse_topk(self, encoded, scores, valid, k):
        indices = topk_region_indices(scores, valid, int(k))
        visual = self.backbone.continue_vision(
            encoded["vision_state"], selected_indices=indices
        )
        fused = self.backbone.fuse(
            encoded["input_ids"], encoded["attention_mask"], visual
        )
        return fused, indices

    def predict_metrics(self, fused, batch):
        return self.metric_head(
            fused,
            batch["candidate_trajectories"],
            batch["candidate_features"],
            batch["goal_state"],
            batch["ego_state"],
            batch["source_id"],
        )


def build_v85_model(args, device):
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
    mask_flow = MaskGradientFlowND(
        latent_dim=args.latent_dim,
        task_dim=args.hidden,
        hidden=args.hidden,
        steps=args.mask_steps,
        initial_step_size=args.mask_step_size,
    ).to(device)
    return GradientFlowSelectorV85(backbone, selector, metric_head, mask_flow)
