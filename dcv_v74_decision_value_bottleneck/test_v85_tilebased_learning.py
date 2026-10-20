"""Actual training pipeline with a tiny frozen generator and real solver."""
import copy
import json
import shutil
import sys
import pytest
import torch
from test_v85_tilebased import setup_tiny, fixture_export
from run_tilebased_v85 import encode
from tilebased_segments_v85 import SEGMENT_SCHEMA
from tilebased_learning_v85 import paired_learning_report, LearningProgress
from tilebased_solver_environment_v85 import FrozenTilebasedSolverEnvironment


def row(sample, loss, success=True, selected=(0, 1)):
    return dict(sample_id=sample, loss=loss, status='success' if success else 'model_failure',
                relative_regret=loss if success else None, system_prompt='shared',
                model_representation='corridor', retained_tokens=len(selected), total_tokens=4,
                selected_indices=list(selected))


def test_learning_comparison_matches_tasks_not_masks_and_keeps_failures():
    before = [row('a', 10, False), row('b', 3)]
    after = [row('b', 4, selected=(1, 2)), row('a', 2, selected=(2, 3))]
    report = paired_learning_report(before, after)
    assert report['mean_task_loss_improvement'] == 3.5
    assert report['mean_regret_improvement_both_success'] == -1
    assert report['success_rate_change'] == .5
    assert report['mean_selection_jaccard'] == pytest.approx(1/6)
    with pytest.raises(ValueError, match='same token budget'):
        paired_learning_report([row('a', 1)], [row('a', 0, selected=(0,))])
    with pytest.raises(ValueError, match='duplicate'):
        paired_learning_report(before+before, after)
    with pytest.raises(ValueError, match='identical'):
        paired_learning_report(before, after[:1])


def test_formal_attempt_counters_include_model_failures_and_separate_diagnostics():
    model, public, labels, args = setup_tiny()
    encoded, *_ = encode(model, public, args.budget)
    env = FrozenTilebasedSolverEnvironment(model.backbone, 2)
    progress = LearningProgress()
    with progress.measure(env, 'training'):
        env.evaluate(encoded, torch.tensor([[0, 1]]), public, labels)
    model.backbone.answer = 'broken JSON'
    with progress.measure(env, 'evaluation'):
        env.evaluate(encoded, torch.tensor([[0, 1]]), public, labels)
    assert progress.values['training_modeling_calls'] == 1
    assert progress.values['training_formal_solver_calls'] == 1
    assert progress.values['training_diagnostic_solver_calls'] == 1
    assert progress.values['evaluation_modeling_calls'] == 1
    assert progress.values['evaluation_formal_solver_calls'] == 0


def test_experiment_trains_freezes_and_compares_same_budget_without_test_split(tmp_path, monkeypatch):
    import run_tilebased_v85 as runner
    data = tmp_path/'data'; data.mkdir(); fixture_export(data)
    shutil.copytree(data/'train', data/'val')
    # Deliberately no test split: the learning experiment must not read test.
    builds = []
    frozen = {}
    def build(args, device):
        model, _, _, _ = setup_tiny()
        model.backbone.features = torch.randn(1, 100, 8)
        model.backbone.target_image_side = 320
        model.backbone.text_hidden_dim = 8
        model.backbone.answer = json.dumps(dict(schema=SEGMENT_SCHEMA, source_rc=[0,0],
            target_rc=[19,19], source_cost=1,
            segments=[dict(from_rc=[0,0],to_rc=[19,19],mean_cost=1)]))
        frozen.update(copy.deepcopy(model.backbone.state_dict()))
        builds.append(model)
        return model
    monkeypatch.setattr(runner, 'build_v85_model', build)
    out = tmp_path/'experiment'
    monkeypatch.setattr(sys, 'argv', ['run_tilebased_v85.py', '--mode', 'experiment',
        '--data', str(data), '--vlm-model', 'test-double', '--device', 'cpu', '--output', str(out),
        '--metric-epochs', '1', '--distill-epochs', '1', '--rl-epochs', '1', '--rl-limit', '1',
        '--hidden', '16', '--latent-dim', '8', '--mask-steps', '2', '--rl-group-size', '2'])
    runner.main()
    assert len(builds) == 1  # Qwen loaded once for all stages and references.
    model = builds[0]
    assert all(torch.equal(v, model.backbone.state_dict()[k]) for k,v in frozen.items())
    assert all(p.grad is None for p in model.backbone.parameters())
    initial = torch.load(out/'untrained.pt', weights_only=True)
    metric = torch.load(out/'metric/best.pt', weights_only=True)
    distill = torch.load(out/'distill/best.pt', weights_only=True)
    last = torch.load(out/'rl/last.pt', weights_only=True)
    best = torch.load(out/'rl/best.pt', weights_only=True)
    assert any(not torch.equal(v, distill['selector'][k]) for k,v in initial['selector'].items())
    assert any(not torch.equal(v, distill['mask_flow'][k]) for k,v in initial['mask_flow'].items())
    assert all(torch.equal(v, metric['selector'][k]) for k,v in initial['selector'].items())
    assert all(torch.equal(v, distill['metric_head'][k]) and torch.equal(v, last['metric_head'][k])
               for k,v in metric['metric_head'].items())
    # Every generated graph has optimal cost on this fixture. A tie must retain
    # the distilled initialization, not misreport an RL gain or overwrite it.
    assert best['stage'] == 'distill' and best['epoch'] == 0
    assert all(torch.equal(v, best['selector'][k]) for k,v in distill['selector'].items())
    assert last['stage'] == 'rl'
    report = json.loads((out/'summary.json').read_text())
    assert report['evaluation_split'] == 'val'
    assert set(report['before']) == {'full', 'untrained'}
    assert report['before']['untrained']['retained_tokens'] == report['after']['learned']['retained_tokens'] == 15
    assert report['comparisons']['untrained_to_learned']['mean_task_loss_improvement'] == 0
    assert report['visual_token_reduction'] == .85
    counts = report['progress']
    assert counts['optimizer_steps'] == 3
    assert counts['selector_optimizer_steps'] == 2
    assert counts['metric_optimizer_steps'] == 1
    assert counts['training_modeling_calls'] == counts['training_formal_solver_calls'] == 3
    assert counts['training_diagnostic_solver_calls'] == 6
    assert counts['evaluation_modeling_calls'] == counts['evaluation_formal_solver_calls'] == 5
    assert counts['evaluation_diagnostic_solver_calls'] == 10
    curve = [json.loads(line) for line in (out/'learning_curve.jsonl').read_text().splitlines()]
    assert [entry['stage'] for entry in curve] == ['untrained', 'distilled', 'rl', 'selected_best']
    assert curve[-1]['progress']['optimizer_steps'] == 3
    updates = [json.loads(line) for line in (out/'rl/updates.jsonl').read_text().splitlines()]
    assert updates[0]['metrics']['reward_std'] == updates[0]['metrics']['advantage_abs'] == 0
    assert updates[0]['metrics']['informative_reward_group_rate'] == 0
    assert updates[0]['metrics']['all_samples_failed_group_rate'] == 0
    assert updates[0]['metrics']['sampled_success_rate'] == 1
    assert json.loads((out/'args.json').read_text())['model_representation'] == 'segments'
