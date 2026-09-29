"""Complete V8.4 model: pruned Qwen, Flow Matching and ND refinement."""

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


class ConditionalTrajectoryFlow(nn.Module):
    """Conditional vector field over a fixed-length ego-frame trajectory."""

    def __init__(self, vlm_dim=4096, task_dim=25, hidden=256, layers=4):
        super().__init__()
        self.vlm = nn.Sequential(
            nn.LayerNorm(vlm_dim), nn.Linear(vlm_dim, hidden), nn.GELU()
        )
        self.state = nn.Sequential(
            nn.LayerNorm(12), nn.Linear(12, hidden), nn.GELU()
        )
        self.task = nn.Sequential(
            nn.LayerNorm(task_dim), nn.Linear(task_dim, hidden), nn.GELU()
        )
        # [trajectory_t(x,y), base(x,y), normalized progress, flow time].
        self.point = nn.Linear(6, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=8,
            dim_feedforward=hidden * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            dropout=0.0,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=layers)
        self.velocity = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 2)
        )

    def forward(
        self,
        trajectory_t,
        time,
        base_trajectory,
        fused_vlm,
        goal_state,
        ego_state,
        task_vector,
    ):
        batch, steps, _ = trajectory_t.shape
        progress = torch.linspace(
            0.0, 1.0, steps, device=trajectory_t.device, dtype=trajectory_t.dtype
        )[None, :, None].expand(batch, -1, -1)
        time_token = time[:, None, None].expand(batch, steps, 1)
        point_input = torch.cat(
            [trajectory_t, base_trajectory, progress, time_token], dim=-1
        )
        context = (
            self.vlm(fused_vlm)
            + self.state(torch.cat([goal_state, ego_state], dim=-1))
            + self.task(task_vector)
        )
        hidden = self.point(point_input) + context[:, None]
        return self.velocity(self.temporal(hidden))

    def matching_loss(
        self,
        base_trajectory,
        target_trajectory,
        fused_vlm,
        goal_state,
        ego_state,
        task_vector,
    ):
        """Conditional flow matching along a straight probability path."""
        batch = base_trajectory.shape[0]
        time = torch.rand(batch, device=base_trajectory.device)
        noise = torch.randn_like(base_trajectory) * 0.25
        noise[:, 0] = 0.0
        source = base_trajectory + noise
        trajectory_t = (
            (1.0 - time[:, None, None]) * source
            + time[:, None, None] * target_trajectory
        )
        target_velocity = target_trajectory - source
        predicted_velocity = self(
            trajectory_t,
            time,
            base_trajectory,
            fused_vlm,
            goal_state,
            ego_state,
            task_vector,
        )
        return F.mse_loss(predicted_velocity, target_velocity)

    def integrate(
        self,
        base_trajectory,
        fused_vlm,
        goal_state,
        ego_state,
        task_vector,
        integration_steps=8,
    ):
        """Euler integration from the candidate warm start to a flow solution."""
        if integration_steps <= 0:
            return base_trajectory
        trajectory = base_trajectory
        step_size = 1.0 / integration_steps
        for step in range(integration_steps):
            time = torch.full(
                (trajectory.shape[0],),
                step / integration_steps,
                device=trajectory.device,
                dtype=trajectory.dtype,
            )
            velocity = self(
                trajectory,
                time,
                base_trajectory,
                fused_vlm,
                goal_state,
                ego_state,
                task_vector,
            )
            trajectory = trajectory + step_size * velocity
            trajectory = torch.cat([base_trajectory[:, :1], trajectory[:, 1:]], dim=1)
        return trajectory


class NeurodynamicRefiner(nn.Module):
    """Unrolled gradient-flow dynamics for safety and trajectory quality."""

    def __init__(self, steps=8, step_size=0.15, clearance=0.35):
        super().__init__()
        self.steps = int(steps)
        self.step_size = float(step_size)
        self.clearance = float(clearance)

    @staticmethod
    def sample_sdf(sdf, trajectory, bounds):
        xmin, xmax, ymin, ymax = bounds.unbind(-1)
        x = trajectory[..., 0]
        y = trajectory[..., 1]
        grid_x = 2.0 * (x - xmin[:, None]) / (xmax - xmin)[:, None] - 1.0
        grid_y = 1.0 - 2.0 * (y - ymin[:, None]) / (ymax - ymin)[:, None]
        grid = torch.stack([grid_x, grid_y], dim=-1)[:, :, None]
        sampled = F.grid_sample(
            sdf, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        return sampled[:, 0, :, 0]

    def energy(self, trajectory, flow_reference, goal_state, sdf, bounds, weights):
        clearance = self.sample_sdf(sdf, trajectory, bounds)
        collision = F.relu(self.clearance - clearance).square().mean(-1)

        velocity = trajectory[:, 1:] - trajectory[:, :-1]
        acceleration = velocity[:, 1:] - velocity[:, :-1]
        length = torch.linalg.vector_norm(velocity, dim=-1).sum(-1)
        smoothness = acceleration.square().sum(-1).mean(-1)

        goal_xy = goal_state[:, :2]
        goal_error = (trajectory[:, -1] - goal_xy).square().sum(-1)
        flow_prior = (trajectory - flow_reference).square().sum(-1).mean(-1)

        weight_sum = weights.sum(-1).clamp_min(1.0)
        safety_weight = (weights[:, 0] + weights[:, 1] + weights[:, 2]) / weight_sum
        length_weight = weights[:, 5] / weight_sum
        comfort_weight = weights[:, 6] / weight_sum
        goal_weight = (
            weights[:, 4] + weights[:, 7] + weights[:, 3]
        ) / weight_sum
        return (
            safety_weight * collision
            + 0.02 * length_weight * length
            + comfort_weight * smoothness
            + goal_weight * goal_error
            + 0.05 * flow_prior
        )

    def forward(
        self,
        initial_trajectory,
        goal_state,
        sdf,
        bounds,
        weights,
        differentiable,
    ):
        """Discretize d tau / dt = -grad_tau E(tau)."""
        trajectory = initial_trajectory
        flow_reference = initial_trajectory
        with torch.enable_grad():
            for _ in range(self.steps):
                trajectory = trajectory.requires_grad_(True)
                energy = self.energy(
                    trajectory, flow_reference, goal_state, sdf, bounds, weights
                ).sum()
                gradient = torch.autograd.grad(
                    energy,
                    trajectory,
                    create_graph=differentiable,
                )[0]
                trajectory = trajectory - self.step_size * gradient
                trajectory = torch.cat(
                    [initial_trajectory[:, :1], trajectory[:, 1:]], dim=1
                )
                if not differentiable:
                    trajectory = trajectory.detach()
        return trajectory


class DecisionFlowNDPlannerV84(nn.Module):
    def __init__(self, backbone, selector, metric_head, flow, refiner):
        super().__init__()
        self.backbone = backbone
        self.selector = selector
        self.metric_head = metric_head
        self.flow = flow
        self.refiner = refiner

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
        return encoded, selection, valid

    def fuse_mask(self, encoded, mask):
        visual = self.backbone.continue_vision(
            encoded["vision_state"], region_mask=mask
        )
        return self.backbone.fuse(
            encoded["input_ids"], encoded["attention_mask"], visual
        )

    def fuse_topk(self, encoded, logits, valid, k):
        indices = topk_region_indices(logits, valid, k)
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


def build_v84_model(args, device):
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
    flow = ConditionalTrajectoryFlow(
        vlm_dim=backbone.text_hidden_dim, hidden=args.hidden
    ).to(device)
    refiner = NeurodynamicRefiner(
        steps=args.nd_steps, step_size=args.nd_step_size
    ).to(device)
    return DecisionFlowNDPlannerV84(
        backbone, selector, metric_head, flow, refiner
    )
