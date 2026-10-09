"""CPU regressions for BEV labels, real pruning, and outcome policy gradients."""

from copy import deepcopy
import itertools
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from bev_geometry_v85 import (
    RASTER_LAYOUT, SCHEMA_REVISION, bev_to_legacy_raster,
    legacy_to_bev_raster, sample_bev_grid_nearest,
)
from bev_renderer_v85 import BevConfig, ego_to_pixel
from checkpoint_contract_v85 import validate_checkpoint_contract
from gradient_flow_selector_v85 import (
    GradientFlowSelectorV85, MaskGradientFlowND, project_capped_simplex,
)
from optimization_spec_v83 import METRIC_NAMES, OptimizationTask
from planning_data_bev_v85 import process_bev_raw_record
from planning_data_v84 import resize_binary_grid, sample_grid_nearest
from planning_dataset_bev_v85 import PlanningDatasetBEVV85
from qwen3vl_selector_v79 import DecisionRelatedQwenSelector, FrozenQwen3VLPrunableBackbone
from raw_record_bev_v85 import INPUT_MODE, save_bev_raw_record
from rl_posttraining_v85 import (
    decision_outcomes, leave_one_out_advantages, ordered_action_statistics,
    policy_gradient_loss, sample_region_action,
)
from test_v85_bev import BevGeometryTest
from train_gradient_flow_selector_v85 import train_epoch, train_metric_epoch, train_rl_epoch
from bev_solver_environment_v85 import MODEL_SCHEMA, SELECTOR_TEXT_CONTRACT
import train_gradient_flow_selector_v85 as trainer


def test_bev_legacy_adapter_roundtrip_rectangular():
    grid = np.arange(35).reshape(5, 7)
    np.testing.assert_array_equal(legacy_to_bev_raster(bev_to_legacy_raster(grid)), grid)


def test_actor_center_detected_in_bev_not_legacy_mapping():
    config = BevConfig()
    rendered = BevGeometryTest._render(config)
    xy = np.asarray([[15.0, 1.5]], dtype=np.float32)
    col, row = np.rint(ego_to_pixel(xy, config)[0]).astype(int)
    assert rendered["dynamic_obstacle"][row, col]
    corrected, inside = sample_bev_grid_nearest(rendered["dynamic_obstacle"], xy, config.bounds)
    old, _ = sample_grid_nearest(rendered["dynamic_obstacle"], xy, config.bounds)
    assert corrected[0] and inside[0]
    assert not old[0]


def _processed_sample(tmp_path):
    config = BevConfig()
    raw_path = tmp_path / "raw.npz"
    save_bev_raw_record(
        raw_path, reference_path_ego=np.asarray([[0, 0, 0], [20, 0, 0], [40, 0, 0]]),
        map_bounds=config.bounds, goal_xy=np.asarray([40, 0]), ego_state=np.zeros(8),
        task_text="safe progress", sample_id="log/scenario/0",
        expert_trajectory_ego=np.asarray([[0, 0, 0], [20, 0, 0]]),
        bev_config_json=config.to_json(), **BevGeometryTest._render(config),
    )
    with np.load(raw_path, allow_pickle=False) as data:
        raw = {key: data[key] for key in data.files}
    return raw, process_bev_raw_record(raw)


def test_processed_labels_use_bev_geometry_and_dataset_rejects_stale(tmp_path):
    raw, sample = _processed_sample(tmp_path)
    obstacle = legacy_to_bev_raster(resize_binary_grid(bev_to_legacy_raster(raw["dynamic_obstacle"])))
    collision, _ = sample_bev_grid_nearest(
        obstacle, sample["candidate_trajectories"][..., :2], raw["map_bounds"]
    )
    valid = sample["candidate_valid"]
    np.testing.assert_allclose(
        sample["candidate_metrics"][valid, METRIC_NAMES.index("collision")], collision[valid].mean(-1)
    )
    root = tmp_path / "processed"
    root.mkdir()
    np.savez_compressed(root / "sample.npz", **sample)
    dataset = PlanningDatasetBEVV85(root)
    assert dataset[0]["image"].shape == (3, 288, 288)
    assert str(sample["metric_raster_layout"].item()) == RASTER_LAYOUT
    assert int(sample["schema_revision"]) == SCHEMA_REVISION
    stale = dict(sample, schema_revision=np.int64(1))
    np.savez_compressed(root / "sample.npz", **stale)
    with pytest.raises(ValueError, match="reprocess"):
        PlanningDatasetBEVV85(root)


def test_projection_jacobian_is_tangent_and_passes_gradcheck():
    values = torch.tensor([[0.1, 0.3, 0.7, 1.5]], dtype=torch.double, requires_grad=True)
    valid = torch.tensor([[True, True, True, False]])
    assert torch.autograd.gradcheck(lambda x: project_capped_simplex(x, valid, 1.4, iterations=60), (values,))
    projected = project_capped_simplex(values, valid, 1.4, iterations=60)
    projected.sum().backward()
    assert torch.allclose(values.grad, torch.zeros_like(values), atol=1e-10)


def test_projection_saturated_coordinates_have_zero_gradient():
    values = torch.tensor([[-10.0, 0.1, 0.3, 10.0]], requires_grad=True)
    result = project_capped_simplex(values, torch.ones_like(values, dtype=torch.bool), 2.0)
    (result * torch.tensor([[1.0, 2.0, 4.0, 8.0]])).sum().backward()
    assert values.grad[0, 0] == values.grad[0, 3] == 0
    assert torch.allclose(values.grad[0, 1:3], torch.tensor([-1.0, 1.0]))


def test_policy_keeps_gradient_when_projected_mask_is_binary():
    flow = MaskGradientFlowND(latent_dim=8, task_dim=16, hidden=16, layers=1, heads=4, steps=1)
    initial = torch.tensor([[0.5, 0.5]])
    velocity = torch.tensor([[-10.0, 10.0]], requires_grad=True)
    proposal = initial + flow.step_size(0) * velocity
    final = project_capped_simplex(proposal, torch.ones_like(initial, dtype=torch.bool), 1.0)
    assert torch.equal(final, torch.tensor([[0.0, 1.0]]))
    scores = flow.decision_scores([initial, final], [velocity])
    logp, _, _ = ordered_action_statistics(scores, torch.ones_like(initial, dtype=torch.bool), torch.tensor([[0]]), 1.0)
    (-logp.mean()).backward()
    assert velocity.grad.abs().sum() > 0


def test_ordered_policy_probability_normalizes_and_rejects_duplicates():
    scores = torch.tensor([[0.1, 0.7, 0.4]], dtype=torch.double, requires_grad=True)
    valid = torch.ones_like(scores, dtype=torch.bool)
    logps = [ordered_action_statistics(scores, valid, torch.tensor([action]), 1.0)[0]
            for action in itertools.permutations(range(3), 2)]
    assert torch.allclose(torch.stack(logps).exp().sum(), torch.tensor(1.0, dtype=torch.double))
    action = torch.tensor([[1, 2]])
    logp, _, kl = ordered_action_statistics(scores, valid, action, 1.0, scores.detach())
    expected = torch.log_softmax(scores, -1)[0, 1] + torch.log_softmax(scores[:, [0, 2]], -1)[0, 1]
    assert torch.allclose(logp[0], expected)
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-10)
    with pytest.raises(ValueError, match="without replacement"):
        ordered_action_statistics(scores, valid, torch.tensor([[1, 1]]))


def test_sampling_budget_validity_and_score_gradients():
    torch.manual_seed(10)
    scores = torch.randn(2, 7, requires_grad=True)
    valid = torch.ones_like(scores, dtype=torch.bool)
    valid[0, -2:] = False
    action, logp, entropy, kl = sample_region_action(scores, valid, 3, 1.0, scores.detach() + 0.1)
    assert action.shape == (2, 3)
    assert valid.gather(1, action).all()
    assert all(row.unique().numel() == 3 for row in action)
    assert torch.isfinite(logp).all() and torch.isfinite(entropy).all() and torch.isfinite(kl).all()
    (-logp.mean()).backward()
    assert torch.isfinite(scores.grad).all() and scores.grad.abs().sum() > 0
    assert (scores.grad[~valid] == 0).all()


def test_leave_one_out_and_external_reward_are_not_self_scoring():
    rewards = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=True)
    advantage = leave_one_out_advantages(rewards)
    assert not advantage.requires_grad
    assert torch.allclose(advantage, torch.tensor([[-1.5], [0.0], [1.5]]))
    assert torch.equal(leave_one_out_advantages(torch.ones(3, 2)), torch.zeros(3, 2))
    predicted = torch.tensor([[9.0, -100.0, -999.0]])
    oracle = torch.tensor([[0.0, 4.0, 100.0]])
    valid = torch.tensor([[True, True, False]])
    metrics = torch.zeros(1, 3, 8)
    metrics[0, 1, 0] = 0.5
    limits = torch.zeros(1, 8)
    limit_mask = torch.zeros(1, 8)
    limit_mask[:, 0] = 1.0
    reward, regret, violation = decision_outcomes(predicted, oracle, valid, metrics, limits, limit_mask)
    assert reward.item() == -2 and regret.item() == violation.item() == 1


def test_reinforce_moves_toward_rewarded_region():
    torch.manual_seed(11)
    scores = nn.Parameter(torch.zeros(1, 3))
    optimizer = torch.optim.SGD([scores], lr=0.2)
    valid = torch.ones_like(scores, dtype=torch.bool)
    logps, rewards = [], []
    for _ in range(64):
        action, logp, _, _ = sample_region_action(scores, valid, 1, 1.0)
        logps.append(logp)
        rewards.append((action[:, 0] == 0).float())
    logps, rewards = torch.stack(logps), torch.stack(rewards)
    zeros = torch.zeros_like(logps)
    loss, _, _ = policy_gradient_loss(logps, rewards, zeros, zeros, 0, 0)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert scores[0, 0] > scores[0, 1] and scores[0, 0] > scores[0, 2]


class TinyPrunableBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        visual = torch.randn(9, 8)
        visual[:, 0] = torch.linspace(-1.0, 1.0, 9)
        self.register_buffer("visual", visual)
        self.calls = []

    def encode_inputs(self, images, texts):
        batch = images.shape[0]
        tokens = self.visual[None].expand(batch, -1, -1)
        return {
            "region_visual_tokens": tokens, "text_tokens": torch.ones(batch, 3, 8),
            "text_valid": torch.ones(batch, 3, dtype=torch.bool),
            "vision_state": {"tokens": tokens}, "input_ids": torch.ones(batch, 3, dtype=torch.long),
            "attention_mask": torch.ones(batch, 3, dtype=torch.long),
        }

    def continue_vision(self, state, region_mask=None, selected_indices=None, selected_region_gates=None):
        tokens = state["tokens"]
        if selected_indices is not None:
            tokens = tokens.gather(1, selected_indices[..., None].expand(-1, -1, 8))
            self.calls.append(("hard", tokens.shape[1], torch.is_grad_enabled()))
            if selected_region_gates is not None:
                tokens = tokens * selected_region_gates[..., None]
        else:
            self.calls.append(("soft", tokens.shape[1], torch.is_grad_enabled()))
            tokens = tokens * region_mask[..., None]
        return {"visual_tokens": tokens}

    def fuse(self, input_ids, attention_mask, visual):
        return visual["visual_tokens"].mean(1)

    def generate_selected_json(self, state, indices, prompt, max_new_tokens):
        # Deterministic frozen visual stub; the real SciPy solver still runs.
        visual = self.continue_vision(state, selected_indices=indices)["visual_tokens"]
        speed = 1.0 + float(visual[..., 0].mean()) * 0.4
        times = np.arange(1, 17) * 0.5
        return json.dumps({"schema": MODEL_SCHEMA, "reference_xy": [[speed * t, 0.] for t in times],
                           "corridor": [[-16., 64., -40., 40.]] * 16,
                           "weights": {"tracking": 1., "smoothness": 0., "goal": 0.}, "halfspaces": []})


class TinyMetricHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))

    def forward(self, fused, trajectories, features, goal, ego, source):
        collision = torch.stack([torch.sigmoid(-self.scale * fused[:, 0]), torch.sigmoid(self.scale * fused[:, 0])], -1)
        return torch.cat([collision[..., None], torch.zeros(*collision.shape, 7)], -1)


def _tiny_setup():
    torch.manual_seed(15)
    model = GradientFlowSelectorV85(
        TinyPrunableBackbone(),
        DecisionRelatedQwenSelector(visual_dim=8, text_dim=8, aligned_dim=16,
                                   hidden=16, latent_dim=8, position_dim=8, layers=1, heads=4),
        TinyMetricHead(),
        MaskGradientFlowND(latent_dim=8, task_dim=16, hidden=16, layers=1, heads=4, steps=2),
    )
    bounds = torch.tensor([[float(i // 3), float(i // 3 + 1), float(i % 3), float(i % 3 + 1)] for i in range(9)])[None]
    batch = {
        "image": torch.zeros(1, 3, 96, 96), "task_text": ["safe progress"], "source_id": torch.zeros(1, dtype=torch.long),
        "region_world_bounds": bounds, "candidate_trajectories": torch.zeros(1, 2, 16, 3),
        "candidate_features": torch.zeros(1, 2, 8), "goal_state": torch.zeros(1, 4), "ego_state": torch.zeros(1, 8),
        "candidate_valid": torch.ones(1, 2, dtype=torch.bool), "candidate_metrics": torch.zeros(1, 2, 8),
        "planning_time_s": torch.arange(1, 17)[None].float() * 0.5,
        "trajectory_gt_available": torch.ones(1, dtype=torch.bool), "expert_is_fallback": torch.zeros(1, dtype=torch.bool),
        "expert_trajectory": torch.stack([torch.arange(1, 17).float() * 0.5, torch.zeros(16)], -1)[None],
        "map_bounds": torch.tensor([[-16., 64., -40., 40.]]),
        "traversable": torch.ones(1, 96, 96, dtype=torch.bool), "dynamic_obstacle": torch.zeros(1, 96, 96, dtype=torch.bool),
        "sample_id": ["tiny/scene/0"],
    }
    batch["candidate_metrics"][0, 1, 0] = 1.0
    args = SimpleNamespace(
        budget=1 / 3, mask_temperature=0.7, mask_steps=2, decision_temperature=0.25,
        teacher_step_size=0.25, gradient_eps=1e-6, gfm_magnitude_weight=0.1, gfm_step_weight=0.5,
        lambda_metric=1.0, lambda_gfm=1.0, lambda_latent=0.25, lambda_value=0.25,
        lambda_improve=0.1, improve_margin=0.0, grad_clip=5.0, nuplan_text="", pointnav_text="",
        rl_group_size=8, rl_temperature=1.0, rl_kl_weight=0.02, rl_entropy_weight=0.001, rl_violation_weight=1.0,
        rl_backend="candidate_proxy", modeling_max_new_tokens=2048, solver_max_speed=15., solver_max_acceleration=4.,
        solver_max_iterations=300, rl_ade_weight=1., rl_fde_weight=0.25, rl_failure_cost=100., seed=1234,
    )
    weights = {key: float(key == "collision") for key in METRIC_NAMES}
    tasks = [OptimizationTask("test", "nuplan", weights, {}, 1.0, "test")] * 2
    return model, batch, args, tasks


def test_unit_forward_gates_equal_hard_pruning_and_reach_scores():
    model, batch, args, _ = _tiny_setup()
    encoded = model.backbone.encode_inputs(batch["image"], batch["task_text"])
    scores = torch.linspace(0.1, 0.9, 9)[None].requires_grad_(True)
    valid = torch.ones_like(scores, dtype=torch.bool)
    hard, indices = model.fuse_topk(encoded, scores, valid, 3)
    surrogate, trained_indices = model.fuse_topk(encoded, scores, valid, 3, straight_through=True)
    assert torch.equal(indices, trained_indices) and torch.equal(hard, surrogate)
    assert model.backbone.calls[-1][1] == 3
    surrogate.sum().backward()
    assert scores.grad.abs().sum() > 0


def test_real_vision_adapter_unit_gates_preserve_shortened_stream():
    backbone = FrozenQwen3VLPrunableBackbone.__new__(FrozenQwen3VLPrunableBackbone)
    nn.Module.__init__(backbone)
    backbone.visual_hidden_dim = backbone.text_hidden_dim = 2
    backbone.merge_unit = 1
    backbone.prune_layer = 0
    visual = SimpleNamespace(blocks=[None, None], deepstack_visual_indexes=[], merger=nn.Identity())
    backbone.model = SimpleNamespace(model=SimpleNamespace(visual=visual))
    backbone._run_block = lambda block, hidden, *args: hidden + hidden.mean(0, keepdim=True)
    state = {"hidden_groups": torch.arange(10.0).reshape(1, 5, 1, 2),
             "position_groups": (torch.ones(1, 5, 1, 2), torch.zeros(1, 5, 1, 2))}
    indices = torch.tensor([[1, 4]])
    gates = torch.ones(1, 2, requires_grad=True)
    hard = backbone.continue_vision(state, selected_indices=indices)
    training = backbone.continue_vision(state, selected_indices=indices, selected_region_gates=gates)
    assert hard["visual_tokens"].shape == (1, 2, 2)
    assert torch.equal(training["visual_tokens"], hard["visual_tokens"])
    training["visual_tokens"].sum().backward()
    assert gates.grad.abs().sum() > 0


def test_all_three_training_stages_run_without_qwen_weights():
    model, batch, args, tasks = _tiny_setup()
    device = torch.device("cpu")
    head_before = model.metric_head.scale.detach().clone()
    optimizer = torch.optim.AdamW(model.metric_head.parameters(), lr=0.01)
    metric = train_metric_epoch(model, [batch], optimizer, args, device, tasks)
    assert np.isfinite(metric["loss"]) and not torch.equal(model.metric_head.scale, head_before)
    model.metric_head.eval().requires_grad_(False)
    fixed_head = model.metric_head.scale.detach().clone()
    trainable = list(model.selector.parameters()) + list(model.mask_flow.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=0.01)
    distill = train_epoch(model, [batch], optimizer, args, device, tasks)
    assert np.isfinite(distill["loss"]) and distill["teacher_direction_norm"] > 0
    assert torch.equal(model.metric_head.scale, fixed_head)
    reference = GradientFlowSelectorV85(model.backbone, deepcopy(model.selector), model.metric_head, deepcopy(model.mask_flow))
    reference.selector.eval().requires_grad_(False)
    reference.mask_flow.eval().requires_grad_(False)
    model.backbone.calls.clear()
    before = [parameter.detach().clone() for parameter in trainable]
    rl = train_rl_epoch(model, reference, [batch], optimizer, args, device, tasks)
    assert all(np.isfinite(value) for value in rl.values())
    assert rl["advantage_abs"] > 0
    assert any(not torch.equal(old, new) for old, new in zip(before, trainable))
    assert all(kind == "hard" and count == 3 and not grad for kind, count, grad in model.backbone.calls)
    assert torch.equal(model.metric_head.scale, fixed_head)
    assert all(parameter.grad is None for parameter in reference.selector.parameters())


def test_checkpoint_geometry_revision_and_stage_contract():
    args = SimpleNamespace(region_grid=9, prune_layer=4, hidden=256, latent_dim=64,
                           mask_steps=4, max_text_length=96, allow_cross_view_resume=False)
    dataset = SimpleNamespace(bev_config_json=BevConfig().to_json())
    checkpoint = dict(input_mode=INPUT_MODE, bev_config_json=dataset.bev_config_json,
                      data_schema_revision=SCHEMA_REVISION, training_stage="distill", args=vars(args),
                      selector_text_contract=SELECTOR_TEXT_CONTRACT)
    validate_checkpoint_contract(checkpoint, args, dataset, "rl")
    for changes, message in [({"bev_config_json": "{}"}, "geometry"),
                             ({"data_schema_revision": 1}, "stale"),
                             ({"training_stage": "metric"}, "distilled")]:
        with pytest.raises(ValueError, match=message):
            validate_checkpoint_contract(dict(checkpoint, **changes), args, dataset, "rl")


def test_cli_checkpoint_chain_metric_distill_rl(tmp_path, monkeypatch):
    """Exercise parsing, freezing, saving/loading and evaluation with tiny modules."""
    _, batch, _, tasks = _tiny_setup()

    class TinyDataset:
        bev_config_json = BevConfig(image_size=96, region_grid=3).to_json()

        def __init__(self, *args, **kwargs):
            pass

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return {key: value[0] for key, value in batch.items()}

    task_path = tmp_path / "task.json"
    tasks[0].save(task_path)
    monkeypatch.setattr(trainer, "PlanningDatasetBEVV85", TinyDataset)
    monkeypatch.setattr(trainer, "build_v85_model", lambda args, device: _tiny_setup()[0])
    previous = None
    checkpoints = []
    common = ["--data", "fake-train", "--val", "fake-val", "--vlm-model", "tiny",
              "--nuplan-task", str(task_path), "--pointnav-task", str(task_path),
              "--device", "cpu", "--workers", "0", "--epochs", "1",
              "--region-grid", "3", "--hidden", "16", "--latent-dim", "8", "--mask-steps", "2",
              "--budget", str(1 / 3), "--rl-group-size", "8", "--rl-temperature", "1.0"]
    for stage in ("metric", "distill", "rl"):
        output = tmp_path / f"{stage}.pt"
        metrics = tmp_path / f"{stage}.json"
        command = ["train", *common, "--stage", stage, "--output", str(output), "--metrics", str(metrics)]
        if previous is not None:
            command.extend(["--resume", str(previous)])
        monkeypatch.setattr(sys, "argv", command)
        trainer.main()
        checkpoint = torch.load(output, map_location="cpu")
        assert checkpoint["training_stage"] == stage
        assert checkpoint["data_schema_revision"] == SCHEMA_REVISION
        assert ("ade_m" if stage == "rl" else "constraint_violation_proxy") in checkpoint["validation"]["learned"]
        assert json.loads(metrics.read_text())[0]["epoch"] == 1
        checkpoints.append(checkpoint)
        previous = output
    assert torch.equal(checkpoints[0]["metric_head"]["scale"], checkpoints[1]["metric_head"]["scale"])
    assert torch.equal(checkpoints[1]["metric_head"]["scale"], checkpoints[2]["metric_head"]["scale"])
    assert checkpoints[-1]["policy_kind"] == "plackett_luce_ordered_subset_v1"
