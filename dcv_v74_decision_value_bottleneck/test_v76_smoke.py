import torch

from joint_decision_bottleneck_v75 import budgeted_soft_mask
from position_c_selector_v76 import (
    PositionCDecisionBottleneck,
    gather_selected_tokens,
    normalized_patch_positions,
)


def main():
    torch.manual_seed(0)
    batch, patches, vision_dim, task_dim, k = 3, 36, 64, 96, 5
    model = PositionCDecisionBottleneck(
        vision_dim=vision_dim,
        task_dim=task_dim,
        hidden=64,
        latent_dim=16,
        task_hidden=24,
        position_dim=8,
        layers=2,
        heads=4,
    )
    vision_tokens = torch.randn(batch, patches, vision_dim)
    task_embedding = torch.randn(batch, task_dim)
    valid = torch.ones(batch, patches, dtype=torch.bool)
    budget = torch.full((batch,), k / patches)
    positions = normalized_patch_positions(
        batch, patches, vision_tokens.device, vision_tokens.dtype
    )

    output = model(
        vision_tokens, positions, task_embedding, valid, budget
    )
    soft_mask = budgeted_soft_mask(output["logits"], valid, k)
    selected, indices = gather_selected_tokens(
        vision_tokens, output["logits"], valid, k
    )
    value = model.predict_set_value(
        output["z"],
        output["task_context"],
        soft_mask,
        valid,
        budget,
    )
    (value.square().mean() + soft_mask.square().mean()).backward()

    assert output["z"].shape == (batch, patches, 16)
    assert output["logits"].shape == (batch, patches)
    assert selected.shape == (batch, k, vision_dim)
    assert indices.shape == (batch, k)
    assert torch.allclose(
        soft_mask.sum(-1), torch.full((batch,), float(k)), atol=1e-3
    )
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    print("V7.6 position-C smoke test: PASS")


if __name__ == "__main__":
    main()
