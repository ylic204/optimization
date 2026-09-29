import torch

from aligned_vlm_selector_v77 import (
    AlignedCrossModalSelector,
    PathDecisionHead,
    gather_selected_visual_tokens,
    normalized_region_positions,
)
from joint_decision_bottleneck_v75 import budgeted_soft_mask


def main():
    torch.manual_seed(0)
    batch, regions, text_length = 3, 36, 12
    aligned_dim, raw_dim, k = 32, 48, 5
    selector = AlignedCrossModalSelector(
        aligned_dim=aligned_dim,
        hidden=64,
        latent_dim=16,
        position_dim=8,
        layers=2,
        heads=4,
    )
    path_head = PathDecisionHead(text_hidden_dim=24, n_paths=81, hidden=32)
    aligned_visual = torch.randn(batch, regions, aligned_dim)
    aligned_text = torch.randn(batch, text_length, aligned_dim)
    raw_visual = torch.randn(batch, regions, raw_dim)
    text_valid = torch.ones(batch, text_length, dtype=torch.bool)
    visual_valid = torch.ones(batch, regions, dtype=torch.bool)
    budget = torch.full((batch,), k / regions)
    positions = normalized_region_positions(
        batch, regions, aligned_visual.device, aligned_visual.dtype
    )
    output = selector(
        aligned_visual,
        aligned_text,
        text_valid,
        positions,
        visual_valid,
        budget,
    )
    soft_mask = budgeted_soft_mask(output["logits"], visual_valid, k)
    selected, indices = gather_selected_visual_tokens(
        raw_visual, output["logits"], visual_valid, k
    )
    value = selector.predict_set_value(
        output["z"],
        output["task_context"],
        soft_mask,
        visual_valid,
        budget,
    )
    fused = torch.randn(batch, 24)
    path_logits = path_head(fused)
    (value.square().mean() + path_logits.square().mean() + soft_mask.square().mean()).backward()
    assert output["logits"].shape == (batch, regions)
    assert output["cross_attention"].shape == (batch, regions, text_length)
    assert selected.shape == (batch, k, raw_dim)
    assert indices.shape == (batch, k)
    assert path_logits.shape == (batch, 81)
    assert torch.allclose(
        soft_mask.sum(-1), torch.full((batch,), float(k)), atol=1e-3
    )
    print("V7.7 aligned-VLM selector smoke test: PASS")


if __name__ == "__main__":
    main()
