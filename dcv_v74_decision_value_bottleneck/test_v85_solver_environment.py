"""Regression checks for actual solver feedback and target/input separation."""

from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from bev_solver_environment_v85 import (
    FrozenBEVSolverEnvironment, MODEL_SCHEMA, SolverConfig, build_model_prompt,
    solve_trajectory_qp, validate_model, summarize_outcomes,
)
from gradient_flow_selector_v85 import GradientFlowSelectorV85
from planning_data_bev_v85 import process_bev_raw_record
from planning_dataset_bev_v85 import PlanningDatasetBEVV85
from qwen3vl_selector_v79 import FrozenQwen3VLPrunableBackbone
from test_v85_posttraining import _processed_sample, _tiny_setup
from train_gradient_flow_selector_v85 import train_rl_epoch, validate_solver
from trajectory_gt_v85 import aligned_expert, displacement_errors
from export_nuplan_bev_v85 import _future_expert, _reference_path_ego
from bev_renderer_v85 import BevConfig


def model_payload(n=16):
    return {"schema": MODEL_SCHEMA, "reference_xy": [[i * 0.5, 0.] for i in range(1, n + 1)],
            "corridor": [[-16., 64., -40., 40.]] * n,
            "weights": {"tracking": 1., "smoothness": 0., "goal": 0.}, "halfspaces": []}


def test_time_alignment_preserves_stop_and_nonuniform_motion():
    raw = {"expert_trajectory_ego": np.array([[1, 0, 0], [1, 0, 0], [4, 0, 0]]),
           "expert_time_s": np.array([1., 2., 4.]), "planning_time_s": np.array([1., 2., 3., 4.])}
    poses, times, available = aligned_expert(raw, steps=4)
    assert available
    np.testing.assert_allclose(poses[:, 0], [1., 1., 2.5, 4.])
    np.testing.assert_allclose(times, [1, 2, 3, 4])
    raw["expert_trajectory_ego"] *= 0
    assert not aligned_expert(raw, steps=4)[0].any()
    raw["planning_time_s"][-1] = 5
    with pytest.raises(ValueError, match="covered"):
        aligned_expert(raw, steps=4)


def test_timestamp_missing_gt_is_not_route_fallback(tmp_path):
    raw, sample = _processed_sample(tmp_path)
    assert not sample["trajectory_gt_available"]
    root = tmp_path / "processed"
    root.mkdir()
    np.savez_compressed(root / "sample.npz", **sample)
    with pytest.raises(ValueError, match="re-export"):
        PlanningDatasetBEVV85(root, require_trajectory_gt=True)
    raw.update(expert_trajectory_ego=np.zeros((16, 3)), expert_time_s=np.arange(1, 17) * .5,
               planning_time_s=np.arange(1, 17) * .5)
    stationary = process_bev_raw_record(raw)
    np.savez_compressed(root / "sample.npz", **stationary)
    dataset = PlanningDatasetBEVV85(root, require_trajectory_gt=True)
    assert not dataset[0]["expert_trajectory"].any()
    assert dataset[0]["trajectory_gt_available"]


def test_continuous_solver_matches_reference_and_honors_halfspace():
    times, bounds = np.arange(1, 17) * .5, [-16, 64, -40, 40]
    ego = [1., 0.]
    payload = model_payload()
    solved = solve_trajectory_qp(payload, times, bounds, ego, [8, 0], SolverConfig())
    assert solved["success"] and solved["max_violation"] <= 1e-4
    np.testing.assert_allclose(np.asarray(solved["trajectory"])[:, :2], payload["reference_xy"], atol=1e-5)
    # A new coordinate is optimized, rather than picking a predefined candidate.
    payload["halfspaces"] = [{"step": 16, "normal": [1, 0], "upper": 3.17}]
    solved = solve_trajectory_qp(payload, times, bounds, ego, [8, 0], SolverConfig())
    assert solved["success"]
    assert abs(solved["trajectory"][-1][0] - 3.17) < 1e-4


def test_infeasible_model_reports_failure_without_trajectory():
    payload = model_payload()
    payload["corridor"][0] = [40., 41., -1., 1.]
    result = solve_trajectory_qp(payload, np.arange(1, 17) * .5, [-16, 64, -40, 40], [0, 0], [8, 0], SolverConfig())
    assert not result["success"] and "trajectory" not in result


@pytest.mark.parametrize("change", [
    {"weights": {"tracking": -1, "smoothness": 0, "goal": 0}},
    {"reference_xy": [[float("nan"), 0]] * 16},
    {"corridor": [[-100, 64, -40, 40]] * 16},
    {"halfspaces": [{"step": 17, "normal": [1, 0], "upper": 0}]},
    {"python": "not a supported key"},
])
def test_generated_problem_validation_rejects_invalid_models(change):
    with pytest.raises((ValueError, TypeError)):
        validate_model(json.dumps(dict(model_payload(), **change)), np.arange(1, 17) * .5, [-16, 64, -40, 40])


def test_gt_is_only_reward_and_not_modeling_input():
    model, batch, args, _ = _tiny_setup()
    args.rl_backend = "llm_solver"
    environment = FrozenBEVSolverEnvironment(model.backbone, args)
    encoded = model.backbone.encode_inputs(batch["image"], batch["task_text"])
    actions = torch.tensor([[0, 1, 2]])
    a = environment.evaluate(encoded, actions, batch, batch["task_text"])[0]
    changed = dict(batch, expert_trajectory=batch["expert_trajectory"] + 20,
                   candidate_metrics=torch.full_like(batch["candidate_metrics"], 900))
    b = environment.evaluate(encoded, actions, changed, batch["task_text"])[0]
    assert a["status"] == b["status"] == "success"
    assert a["model_answer"] == b["model_answer"] and a["trajectory"] == b["trajectory"]
    assert a["loss"] < b["loss"] and a["ade_m"] < b["ade_m"]
    assert "expert" not in build_model_prompt("safe", [.5, 1], [-16, 64, -40, 40], [0, 0], [8, 0])


def test_model_failure_is_penalized_and_not_reported_as_zero_ade():
    model, batch, args, _ = _tiny_setup()
    environment = FrozenBEVSolverEnvironment(model.backbone, args)
    model.backbone.generate_selected_json = lambda *args: "not JSON"
    encoded = model.backbone.encode_inputs(batch["image"], batch["task_text"])
    result = environment.evaluate(encoded, torch.tensor([[0, 1, 2]]), batch, batch["task_text"])
    assert result[0]["status"] == "model_failure" and result[0]["loss"] == args.rl_failure_cost
    summary = summarize_outcomes(result)
    assert summary["ade_m"] is None and summary["model_failure_rate"] == 1.


def test_actual_solver_rl_updates_selector_and_freezes_head_and_reference(tmp_path):
    model, batch, args, tasks = _tiny_setup()
    args.rl_backend = "llm_solver"
    model.backbone.requires_grad_(False)
    model.metric_head.requires_grad_(False)
    head = deepcopy(model.metric_head.state_dict())
    reference = GradientFlowSelectorV85(model.backbone, deepcopy(model.selector), model.metric_head, deepcopy(model.mask_flow))
    reference.selector.eval().requires_grad_(False)
    reference.mask_flow.eval().requires_grad_(False)
    trainable = list(model.selector.parameters()) + list(model.mask_flow.parameters())
    before = [p.detach().clone() for p in trainable]
    optimizer = torch.optim.AdamW(trainable, lr=.01)
    metrics = train_rl_epoch(model, reference, [batch], optimizer, args, torch.device("cpu"), tasks)
    assert metrics["advantage_abs"] > 0 and metrics["model_failure_rate"] == metrics["solver_failure_rate"] == 0
    assert any(not torch.equal(a, b) for a, b in zip(before, trainable))
    assert all(torch.equal(head[k], v) for k, v in model.metric_head.state_dict().items())
    assert all(p.grad is None for p in reference.selector.parameters())
    records_path = tmp_path / "outcomes.jsonl"
    metrics = validate_solver(model, [batch], args, torch.device("cpu"), FrozenBEVSolverEnvironment(model.backbone, args), records_path)
    assert set(metrics) == {"full", "random", "initial", "learned"}
    assert all(metrics[m]["tokens"] == 3 for m in ("random", "initial", "learned"))
    records = [json.loads(x) for x in records_path.read_text().splitlines()]
    assert len(records) == 4 and all("gt_trajectory" in x and "optimization_model" in x for x in records)


def test_ade_fde_use_time_aligned_coordinates():
    ade, fde = displacement_errors([[0, 0], [3, 4]], [[0, 0], [0, 0]])
    assert ade == 2.5 and fde == 5


def test_export_records_actual_times_and_missing_future_contract():
    anchor = 10_000_000
    states = [SimpleNamespace(time_point=SimpleNamespace(time_us=anchor + int(t * 1e6)),
                              rear_axle=SimpleNamespace(x=t, y=0., heading=0.)) for t in (.5, 1.1)]
    scenario = SimpleNamespace(get_time_point=lambda i: SimpleNamespace(time_us=anchor),
                               get_ego_future_trajectory=lambda *args, **kwargs: iter(states))
    args = SimpleNamespace(future_horizon=1., future_samples=2)
    poses, times = _future_expert(scenario, 0, np.zeros(3), args)
    np.testing.assert_allclose(times, [.5, 1.1])
    np.testing.assert_allclose(poses[:, 0], times)
    states.clear()
    poses, times = _future_expert(scenario, 0, np.zeros(3), args)
    assert poses.shape == (0, 3) and times.shape == (0,)
    assert _reference_path_ego([], np.zeros(3), BevConfig()).shape == (0, 3)


@pytest.mark.parametrize("prune_layer", [0, 1])
@pytest.mark.parametrize("system_prompt", [None, "Top-down navigation environment"])
def test_real_qwen_text_model_sparse_prefill_and_cached_decode(prune_layer, system_prompt):
    torch.manual_seed(7)
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen3VLConfig(
        text_config=dict(vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=2, head_dim=16,
                         rope_scaling={"rope_type": "default", "mrope_section": [2, 3, 3], "mrope_interleaved": True}),
        vision_config=dict(depth=2, hidden_size=16, intermediate_size=32, num_heads=2,
                           out_hidden_size=32, deepstack_visual_indexes=[1]),
        image_token_id=5, vision_start_token_id=4, vision_end_token_id=6,
    )
    backbone = FrozenQwen3VLPrunableBackbone.__new__(FrozenQwen3VLPrunableBackbone)
    nn.Module.__init__(backbone)
    backbone.model = transformers.Qwen3VLForConditionalGeneration(config).eval().requires_grad_(False)
    backbone.model.generation_config.eos_token_id = []
    backbone.region_grid = 3
    backbone.visual_hidden_dim, backbone.text_hidden_dim = 16, 32
    backbone.merge_unit, backbone.prune_layer = 4, prune_layer
    backbone.spatial_merge_size = 2
    backbone.target_pixels = 96**2
    backbone.model.generation_config.pad_token_id = 0

    class Tokenizer:
        def __call__(self, *args, **kwargs):
            return {"input_ids": torch.tensor([([9,10,11] if system_prompt else []) + [1,4,5,6,7,8]])}

        def decode(self, ids, **kwargs):
            return json.dumps(ids)

    def chat_template(messages, **kwargs):
        if system_prompt:
            assert messages[0] == {"role":"system", "content":system_prompt}
        else:
            assert len(messages)==1 and messages[0]["role"]=="user"
        return "chat"
    backbone.processor = SimpleNamespace(tokenizer=Tokenizer(), apply_chat_template=chat_template)
    pixels, grid = torch.randn(36, 3 * 2 * 16 * 16), torch.tensor([[1, 6, 6]])
    backbone.processor.image_processor = lambda **kwargs: {"pixel_values": pixels, "image_grid_thw": grid}
    hidden, pairs, cu_seqlens, max_seqlen = backbone._prepare_vision_stream(pixels, grid)
    for block in backbone.visual.blocks[:prune_layer]:
        hidden = backbone._run_block(block, hidden, cu_seqlens, pairs, max_seqlen)
    state = {"hidden_groups": hidden.reshape(1, 9, 4, 16),
             "position_groups": tuple(x.reshape(1, 9, 4, -1) for x in pairs)}
    visual = backbone.continue_vision(state, selected_indices=torch.tensor([[0, 8]]))
    assert visual["visual_tokens"].shape == (1, 2, 32) and len(visual["deepstack_tokens"]) == 1
    embeddings, attention, mask, positions = backbone._selected_chat_inputs(visual, "prompt", system_prompt)
    offset = 3 if system_prompt else 0
    assert embeddings.shape == (1, 7+offset, 32) and mask.sum() == 2
    assert positions[2, 0, 2+offset:4+offset].tolist() == [2+offset, 4+offset]
    assert positions[3, 0, 2+offset:4+offset].tolist() == [2+offset, 4+offset]
    answer = backbone.generate_selected_json(state, torch.tensor([[8, 0]]), "prompt", max_new_tokens=3, system_prompt=system_prompt)
    assert len(json.loads(answer)) == 3
    assert all(p.grad is None for p in backbone.model.parameters())

    image = torch.zeros(1, 3, 96, 96)
    parity = backbone.compare_full_prefill(image, state, "prompt", system_prompt)
    assert parity["first_token_equal"]
    assert parity["max_abs_logit_difference"] < 1e-5
    native = backbone.generate_native_report(image, "prompt", 3, system_prompt)
    custom = backbone.generate_selected_report(state, torch.arange(9)[None], "prompt", 3, system_prompt)
    assert native["text"] == custom["text"]
    assert native["generated_tokens"] == custom["generated_tokens"] == 3
    assert native["finish_reason"] == custom["finish_reason"] == "length"
