import torch

from joint_decision_bottleneck_v75 import budgeted_soft_mask
from qwen3vl_selector_v78 import (
    DecisionRelatedQwenSelector,
    PathDecisionHead,
    mask_to_region_indices,
    normalized_region_positions,
    topk_region_indices,
)


def main():
    torch.manual_seed(0)
    batch, regions, text_length = 3, 36, 12
    visual_dim, text_dim, k = 48, 64, 5
    selector = DecisionRelatedQwenSelector(
        visual_dim=visual_dim,
        text_dim=text_dim,
        aligned_dim=32,
        hidden=64,
        latent_dim=16,
        position_dim=8,
        layers=2,
        heads=4,
    )
    path_head = PathDecisionHead(
        text_hidden_dim=96, n_paths=81, hidden=32
    )
    visual = torch.randn(batch, regions, visual_dim)
    text = torch.randn(batch, text_length, text_dim)
    text_valid = torch.ones(batch, text_length, dtype=torch.bool)
    visual_valid = torch.ones(batch, regions, dtype=torch.bool)
    budget = torch.full((batch,), k / regions)
    positions = normalized_region_positions(
        batch, regions, visual.device, visual.dtype
    )
    output = selector(
        visual,
        text,
        text_valid,
        positions,
        visual_valid,
        budget,
    )
    soft_mask = budgeted_soft_mask(output["logits"], visual_valid, k)
    indices = topk_region_indices(output["logits"], visual_valid, k)
    hard_mask = torch.zeros_like(output["logits"]).scatter(1, indices, 1.0)
    recovered_indices = mask_to_region_indices(hard_mask)
    value = selector.predict_set_value(
        output["z"],
        output["task_context"],
        soft_mask,
        visual_valid,
        budget,
    )
    fused = torch.randn(batch, 96)
    path_logits = path_head(fused)
    total = (
        value.square().mean()
        + path_logits.square().mean()
        + soft_mask.square().mean()
    )
    total.backward()

    assert output["logits"].shape == (batch, regions)
    assert output["cross_attention"].shape == (
        batch,
        regions,
        text_length,
    )
    assert indices.shape == (batch, k)
    assert torch.equal(indices, recovered_indices)
    assert path_logits.shape == (batch, 81)
    assert torch.allclose(
        soft_mask.sum(-1),
        torch.full((batch,), float(k)),
        atol=1e-3,
    )
    print("V7.8 Qwen3-VL decision selector smoke test: PASS")


if __name__ == "__main__":
    main()
