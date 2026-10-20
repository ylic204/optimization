"""Sparse route modeling: geometry, exact graph solve, GT isolation and feedback."""
import json
import sys
import copy
import numpy as np
import pytest
import torch
from tilebased_core_v85 import shortest_path, path_cost, validate_path, task_text, PKMN_LEGEND
from tilebased_corridor_v85 import (CORRIDOR_SCHEMA, raster_segment, validate_corridor,
    compile_corridor, solve_corridor, corridor_evaluation)
from tilebased_prompts_v85 import modeling_user_prompt
from tilebased_solver_environment_v85 import FrozenTilebasedSolverEnvironment
from test_v85_tilebased import setup_tiny, fixture_export
from run_tilebased_v85 import encode, rl_loss
from gradient_flow_selector_v85 import GradientFlowSelectorV85


def graph_payload(direct=10.,detour=1.):
    return dict(schema=CORRIDOR_SCHEMA,source_rc=[1,0],target_rc=[1,2],source_cost=1.,
                nodes_rc=[[1,0],[1,2],[0,1]],edges=[[0,1,direct],[0,2,detour],[2,1,detour]])


def compiled(obj):
    return compile_corridor(validate_corridor(json.dumps(obj),'pkmn',obj['source_rc'],obj['target_rc']))


def test_geometry_reversible_neighbor_steps_and_bounds():
    for a in [(0,0),(1,7),(19,0),(19,19)]:
        for r in range(20):
            for c in range(20):
                b=(r,c);line=raster_segment(a,b)
                assert line==raster_segment(b,a)[::-1]
                assert len(line)==max(abs(a[0]-r),abs(a[1]-c))+1
                validate_path(np.array(line),a,b,(20,20))


def test_solver_compares_connections_and_regret_decomposition():
    truth=np.ones((20,20));truth[1,1]=25
    optimum=path_cost(truth,shortest_path(truth,[1,0],[1,2]))
    assert optimum==3
    graph=compiled(graph_payload())
    route,cost=solve_corridor(graph)
    assert route.tolist()==[[1,0],[0,1],[1,2]] and cost==3
    # Correct topology, wrong estimates lead to a costly crossing.
    wrong=compiled(graph_payload(direct=1,detour=10))
    route,estimate=solve_corridor(wrong)
    score=corridor_evaluation(wrong,route,estimate,truth,optimum)
    assert score['corridor_restriction_regret']==0
    assert score['corridor_cost_model_regret']==8
    # Omitting the detour creates representation regret even with true costs.
    obj=graph_payload();obj['edges']=[[0,1,10.]]
    restricted=compiled(obj);route,estimate=solve_corridor(restricted)
    score=corridor_evaluation(restricted,route,estimate,truth,optimum)
    assert score['corridor_restriction_regret']==8 and score['corridor_cost_model_regret']==0


def test_crossings_connect_only_at_modeled_cells_and_source_counted_once():
    obj=graph_payload();obj['nodes_rc']=[[1,0],[0,1],[1,2],[2,1]]
    obj['target_rc']=[0,1];obj['edges']=[[0,2,2],[3,1,3]]
    graph=compiled(obj);route,cost=solve_corridor(graph)
    assert route.tolist()==[[1,0],[1,1],[0,1]] and cost==6
    obj['edges']=[[0,2,2]]
    assert solve_corridor(compiled(obj))==(None,None)
    obj=dict(schema=CORRIDOR_SCHEMA,source_rc=[2,2],target_rc=[2,2],source_cost=1.5,nodes_rc=[[2,2]],edges=[])
    route,cost=solve_corridor(compiled(obj))
    assert route.tolist()==[[2,2]] and cost==1.5


@pytest.mark.parametrize('change',[
    {'node_costs':[[1.]]}, {'source_cost':True}, {'source_cost':float('nan')},
    {'nodes_rc':[[1,0],[1,2],[20,0]]}, {'nodes_rc':[[1,0],[1,2],[1,2]]},
    {'edges':[[0,8,1]]}, {'edges':[[0,1,-1]]}, {'edges':[[0,1,30]]},
    {'edges':[[True,1,1]]}, {'edges':[[0,1,1],[0,1,2]]},
])
def test_reject_invalid_model(change):
    obj=dict(graph_payload(),**change)
    with pytest.raises((ValueError,TypeError)):
        compiled(obj)


def test_environment_does_not_use_gt_to_route_or_fill_missing_connections():
    model,public,labels,args=setup_tiny()
    public['source_rc']=torch.tensor([[1,0]]);public['target_rc']=torch.tensor([[1,2]])
    public['task_text']=[task_text('pkmn',[1,0],[1,2])]
    truth=torch.ones(1,20,20);truth[0,1,1]=25
    labels={'true_costs':truth,'optimal_cost':torch.tensor([3.])}
    model.backbone.answer=json.dumps(graph_payload(direct=1,detour=10))
    encoded,*_=encode(model,public,args.budget)
    env=FrozenTilebasedSolverEnvironment(model.backbone,2,prompt_variant='navigation',model_representation='corridor')
    first=env.evaluate(encoded,torch.tensor([[0,2]]),public,labels)[0]
    second=env.evaluate(encoded,torch.tensor([[0,2]]),public,dict(true_costs=torch.ones(1,20,20),optimal_cost=torch.tensor([3.])))[0]
    assert first['status']==second['status']=='success'
    assert first['loss']==8 and second['loss']==0
    assert first['path_rc']==second['path_rc']
    assert first['modeling_prompt']==second['modeling_prompt']
    assert 'matrix_diagnostics' not in first and 'cost_cell_accuracy' not in first
    disconnected=graph_payload();disconnected['edges']=[[0,2,1]]
    model.backbone.answer=json.dumps(disconnected)
    failure=env.evaluate(encoded,torch.tensor([[0,2]]),public,labels)[0]
    assert failure['status']=='solver_failure' and failure['loss']==10
    assert failure['path_rc'] is None and not failure['graph_connected']


def test_corridor_rl_backpropagates_only_into_selector_and_flow():
    model,public,labels,args=setup_tiny()
    public['source_rc']=torch.tensor([[1,0]]);public['target_rc']=torch.tensor([[1,2]])
    public['task_text']=[task_text('pkmn',[1,0],[1,2])]
    truth=torch.ones(1,20,20);truth[0,1,1]=25
    labels={'true_costs':truth,'optimal_cost':torch.tensor([3.])}
    def generate(state,indices,prompt,max_new_tokens,system_prompt=None):
        return json.dumps(graph_payload(direct=10,detour=1) if 0 in indices[0].tolist()
                          else graph_payload(direct=1,detour=10))
    model.backbone.generate_selected_json=generate
    reference=GradientFlowSelectorV85(model.backbone,copy.deepcopy(model.selector),model.metric_head,copy.deepcopy(model.mask_flow))
    reference.selector.requires_grad_(False);reference.mask_flow.requires_grad_(False)
    env=FrozenTilebasedSolverEnvironment(model.backbone,2,prompt_variant='navigation',model_representation='corridor')
    encoded,selection,positions,valid=encode(model,public,args.budget)
    loss,metrics=rl_loss(model,reference,env,encoded,selection,positions,valid,public,labels,args)
    loss.backward()
    assert torch.isfinite(loss) and metrics['advantage_abs']>0
    for module in (model.selector,model.mask_flow):
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters())
    assert all(p.grad is None for p in model.metric_head.parameters())


def test_default_cli_is_corridor_and_full_random_share_prompt_rules(tmp_path,monkeypatch):
    import run_tilebased_v85 as runner
    data=tmp_path/'data';data.mkdir();fixture_export(data)
    def build(args,device):
        model,_,_,_=setup_tiny();model.backbone.features=torch.randn(1,100,8)
        model.backbone.target_image_side=320;model.backbone.text_hidden_dim=8
        model.backbone.answer=json.dumps(dict(schema=CORRIDOR_SCHEMA,source_rc=[0,0],target_rc=[19,19],
            source_cost=1.,nodes_rc=[[0,0],[19,19]],edges=[[0,1,1.]]))
        return model
    monkeypatch.setattr(runner,'build_v85_model',build)
    out=tmp_path/'result'
    monkeypatch.setattr(sys,'argv',['run','--mode','evaluate','--data',str(data),'--split','train',
        '--vlm-model','test-double','--device','cpu','--output',str(out),'--modes','full','random'])
    runner.main()
    rows=[json.loads(s) for s in (out/'outcomes.jsonl').read_text().splitlines()]
    assert len(rows)==2 and rows[0]['system_prompt']==rows[1]['system_prompt']
    assert all(r['model_representation']=='corridor' and r['true_cost']==20 for r in rows)
    assert 'cost_cell_accuracy_successful' not in json.loads((out/'summary.json').read_text())['full']
    assert json.loads((out/'args.json').read_text())['model_representation']=='corridor'
