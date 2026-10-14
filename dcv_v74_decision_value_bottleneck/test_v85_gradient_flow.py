import torch

from gradient_flow_selector_v85 import (
    MaskGradientFlowND,
    endpoint_improvement_loss,
    normalized_bev_region_positions,
    project_capped_simplex,
)


def test_capped_simplex_preserves_budget_and_bounds():
    torch.manual_seed(3)
    values = torch.randn(3, 11, requires_grad=True)
    valid = torch.ones_like(values, dtype=torch.bool)
    valid[1, -2:] = False
    mass = torch.tensor([4.0, 3.0, 5.0])
    projected = project_capped_simplex(values, valid, mass)
    assert torch.allclose(projected.sum(-1), mass, atol=1e-5)
    assert bool((projected >= 0.0).all())
    assert bool((projected <= 1.0).all())
    assert bool((projected[1, -2:] == 0.0).all())
    projected.square().sum().backward()
    assert values.grad is not None


def test_mask_nd_rollout_stays_feasible():
    torch.manual_seed(4)
    batch, regions = 2, 9
    valid = torch.ones(batch, regions, dtype=torch.bool)
    initial = torch.full((batch, regions), 3.0 / regions)
    z = torch.randn(batch, regions, 8)
    task = torch.randn(batch, 16)
    positions = torch.randn(batch, regions, 2)
    model = MaskGradientFlowND(
        latent_dim=8,
        task_dim=16,
        hidden=32,
        layers=1,
        heads=4,
        steps=3,
        initial_step_size=0.2,
    )
    final, states, velocities = model.rollout(
        initial, z, task, positions, valid, mass=3.0
    )
    assert len(states) == 4
    assert len(velocities) == 3
    assert torch.allclose(final.sum(-1), torch.full((batch,), 3.0), atol=1e-5)
    assert bool((final >= 0.0).all())
    assert bool((final <= 1.0).all())


def test_endpoint_improvement_detaches_baseline():
    initial = torch.tensor([0.5, 0.3], requires_grad=True)
    final = torch.tensor([0.4, 0.4], requires_grad=True)
    loss = endpoint_improvement_loss(initial, final, margin=0.0).mean()
    loss.backward()
    assert initial.grad is None
    assert torch.allclose(final.grad, torch.tensor([0.0, 0.5]))


def test_invalid_tokens_do_not_set_projection_bounds():
    values = torch.tensor([[0.2, 0.4, 1e6]])
    valid = torch.tensor([[True, True, False]])
    projected = project_capped_simplex(values, valid, mass=1.0)
    assert torch.allclose(projected.sum(-1), torch.tensor([1.0]), atol=1e-5)
    assert projected[0, 2] == 0.0


def test_bev_positions_are_x_forward_y_left():
    bounds = torch.tensor(
        [
            [
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 1.0, -1.0, 0.0],
                [-1.0, 0.0, 0.0, 1.0],
                [-1.0, 0.0, -1.0, 0.0],
            ]
        ]
    )
    visual = torch.zeros(1, 4, 8)
    positions = normalized_bev_region_positions(bounds, visual)
    expected = torch.tensor(
        [[[0.5, 0.5], [0.5, -0.5], [-0.5, 0.5], [-0.5, -0.5]]]
    )
    assert torch.allclose(positions, expected)
