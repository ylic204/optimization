import torch

from joint_decision_bottleneck_v75 import (
    JointDecisionBottleneck,
    budgeted_soft_mask,
    hard_topk_mask,
)


def main():
    torch.manual_seed(0)
    batch, patches, feat_dim = 3, 36, 64
    model = JointDecisionBottleneck(
        feat_dim=feat_dim,
        graph_dim=4,
        hidden=64,
        latent_dim=16,
        layers=2,
        heads=4,
    )
    preview = torch.randn(batch, patches, feat_dim)
    graph = torch.randn(batch, patches, 4)
    valid = torch.ones(batch, patches, dtype=torch.bool)
    budget = torch.full((batch,), 5.0 / patches)

    output = model(preview, graph, valid, budget)
    soft = budgeted_soft_mask(output["logits"], valid, k=5, temperature=0.35)
    hard = hard_topk_mask(output["logits"], valid, k=5)
    value = model.predict_set_value(output["z"], soft, valid, budget)
    loss = value.square().mean() + soft.square().mean()
    loss.backward()

    assert output["z"].shape == (batch, patches, 16)
    assert output["logits"].shape == (batch, patches)
    assert torch.allclose(soft.sum(-1), torch.full((batch,), 5.0), atol=1e-3)
    assert torch.equal(hard.sum(-1), torch.full((batch,), 5.0))
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    print("V7.5 smoke test: PASS")


if __name__ == "__main__":
    main()

