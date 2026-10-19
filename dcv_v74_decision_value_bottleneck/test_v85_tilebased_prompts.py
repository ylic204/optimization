"""Prompt integration, paired-budget fairness and failure-aware reporting."""
import json
import sys
import numpy as np
import pytest
import torch
from tilebased_core_v85 import build_model_prompt, PKMN_LEGEND
from tilebased_prompts_v85 import modeling_user_prompt, modeling_system_prompt
from tilebased_prompt_comparison_v85 import paired_prompt_report
from tilebased_solver_environment_v85 import FrozenTilebasedSolverEnvironment
from test_v85_tilebased import setup_tiny, fixture_export
from run_tilebased_v85 import encode


def test_legacy_unchanged_navigation_uses_system_and_keeps_labels_private():
    model,public,labels,args=setup_tiny()
    encoded,*_=encode(model,public,args.budget)
    indices=torch.tensor([[0,2]])
    calls=[]
    def generate(state,chosen,prompt,max_new_tokens,system_prompt=None):
        calls.append((prompt,system_prompt,chosen.tolist()))
        return model.backbone.answer
    model.backbone.generate_selected_json=generate
    for variant in ('legacy','navigation'):
        env=FrozenTilebasedSolverEnvironment(model.backbone,2,prompt_variant=variant)
        a=env.evaluate(encoded,indices,public,labels)[0]
        other={'true_costs':torch.ones_like(labels['true_costs']),'optimal_cost':torch.tensor([20.])}
        b=env.evaluate(encoded,indices,public,other)[0]
        assert calls[-1]==calls[-2]
        assert a['status']==b['status']=='success' and a['loss']>b['loss']
        assert a['modeling_prompt_variant']==variant
    expected=build_model_prompt(public['task_text'][0],'pkmn',[0,0],[19,19],[0,2],2,PKMN_LEGEND)
    assert calls[0]==(expected,None,[[0,2]])
    assert calls[-1][1]==modeling_system_prompt('navigation','pkmn')
    assert 'Output only the requested JSON' in calls[-1][0]
    assert 'true_costs' not in calls[-1][0]


def test_paired_report_counts_failures_and_rejects_unpaired_budgets():
    def row(sid,success,loss):
        return dict(sample_id=sid,status='success' if success else 'model_failure',loss=loss,
                    relative_regret=loss if success else None,selected_indices=[0,2])
    old=[row('a',False,10),row('b',True,2),row('c',True,3)]
    new=[row('a',True,1),row('b',False,10),row('c',True,1)]
    report=paired_prompt_report(old,new)
    assert report['mean_task_loss_improvement']==1.
    assert report['success_rate_change']==0.
    assert report['mean_regret_improvement_both_success']==2.
    assert report['task_loss_win_rate']==pytest.approx(2/3)
    new[0]['selected_indices']=[0,3]
    with pytest.raises(ValueError,match='identical visual'):paired_prompt_report(old,new)
    with pytest.raises(ValueError,match='duplicate'):paired_prompt_report(old+old,new)


def test_compare_cli_runs_same_images_queries_regions_and_records_prompts(tmp_path,monkeypatch):
    import run_tilebased_v85 as runner
    data=tmp_path/'data';data.mkdir();fixture_export(data)
    calls=[]
    def build(args,device):
        model,_,_,_=setup_tiny()
        model.backbone.features=torch.randn(1,100,8)
        model.backbone.target_image_side=320;model.backbone.text_hidden_dim=8
        def generate(state,indices,prompt,max_new_tokens,system_prompt=None):
            calls.append((state['hidden_groups'].clone(),indices.clone(),prompt,system_prompt))
            return model.backbone.answer
        model.backbone.generate_selected_json=generate
        return model
    monkeypatch.setattr(runner,'build_v85_model',build)
    out=tmp_path/'comparison'
    monkeypatch.setattr(sys,'argv',['run','--mode','evaluate','--compare-prompts','--data',str(data),
        '--split','train','--vlm-model','test-double','--device','cpu','--output',str(out),
        '--modes','full','random','--budget','.15'])
    runner.main()
    for a,b in zip(calls[::2],calls[1::2]):
        assert torch.equal(a[0],b[0]) and torch.equal(a[1],b[1])
        assert a[3] is None and b[3] and a[2]!=b[2]
    assert calls[0][1].shape[1]==100 and calls[2][1].shape[1]==15
    summary=json.loads((out/'summary.json').read_text())
    assert set(summary)=={'legacy','navigation','paired'}
    assert summary['paired']['full']['mean_task_loss_improvement']==0
    rows=[json.loads(line) for line in (out/'outcomes.jsonl').read_text().splitlines()]
    assert len(rows)==4 and all('prompt_sha256' in row for row in rows)
    assert json.loads((out/'prompt_config.json').read_text())['navigation']['system_prompt']
