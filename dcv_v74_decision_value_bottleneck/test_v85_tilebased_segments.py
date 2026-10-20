"""Coordinate graph semantics and bounded generation, including failure cases."""
import copy
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tilebased_segments_v85 import SEGMENT_SCHEMA, validate_segments, SegmentOutputGuard
from tilebased_corridor_v85 import compile_corridor, solve_corridor, MAX_EDGES
from tilebased_prompts_v85 import modeling_user_prompt
from tilebased_core_v85 import PKMN_LEGEND, task_text
from tilebased_solver_environment_v85 import FrozenTilebasedSolverEnvironment, summarize_outcomes
from qwen3vl_selector_v79 import FrozenQwen3VLPrunableBackbone
from test_v85_tilebased import setup_tiny
from run_tilebased_v85 import encode, rl_loss
from gradient_flow_selector_v85 import GradientFlowSelectorV85


def segment(a, b, cost=1):
    return dict(from_rc=a, to_rc=b, mean_cost=cost)


def payload():
    return dict(schema=SEGMENT_SCHEMA,source_rc=[1,0],target_rc=[1,2],source_cost=1,
                segments=[segment([1,0],[1,2],10),segment([1,0],[0,1]),segment([0,1],[1,2])])


def compiled(obj):
    return compile_corridor(validate_segments(json.dumps(obj),'pkmn',obj['source_rc'],obj['target_rc']))


def test_direct_coordinates_order_independent_and_solver_chooses_cost():
    obj=payload(); route,cost=solve_corridor(compiled(obj))
    assert route.tolist()==[[1,0],[0,1],[1,2]] and cost==3
    obj['segments'].reverse()
    second,value=solve_corridor(compiled(obj))
    assert np.array_equal(route,second) and value==cost
    for s in obj['segments']:s['mean_cost']=10 if s['to_rc']==[0,1] else 1
    assert solve_corridor(compiled(obj))[0].tolist()==[[1,0],[1,1],[1,2]]


def test_missing_connections_are_not_completed_and_stationary_query():
    obj=payload();obj['segments']=[segment([1,0],[0,1])]
    assert solve_corridor(compiled(obj))==(None,None)
    obj['segments']=[]
    assert solve_corridor(compiled(obj))==(None,None)
    obj['target_rc']=obj['source_rc']
    assert solve_corridor(compiled(obj))[0].tolist()==[[1,0]]


@pytest.mark.parametrize('bad',[
    segment([20,0],[1,2]), segment([1.,0],[1,2]), segment([True,0],[1,2]),
    segment([1,0],[1,0]), segment([1,0],[1,2],-1), segment([1,0],[1,2],float('nan')),
    segment([1,0],[1,2],True), [0,1,1], dict(from_node=0,to_node=1,mean_cost=1),
])
def test_invalid_coordinates_and_costs_rejected(bad):
    obj=payload();obj['segments']=[bad]
    with pytest.raises((ValueError,TypeError)):compiled(obj)


def test_guard_handles_partial_tokens_repeats_limits_and_complete_json():
    guard=SegmentOutputGuard('pkmn')
    valid=json.dumps(payload())
    for end in range(1,len(valid)):
        assert guard(valid[:end]) is None
    assert guard(valid)=='json_complete'
    one=json.dumps(segment([1,0],[1,2]))
    assert guard('{"segments":['+one+','+one)=='repeated_segment'
    assert guard('{"segments":['+json.dumps(segment([20,0],[1,2])))=='invalid_segment'
    assert guard('{"segments":['+one+' X')=='invalid_segment_separator'
    connections=[segment([i//19,i%19],[i//19,i%19+1]) for i in range(MAX_EDGES)]
    prefix='{"segments":['+','.join(map(json.dumps,connections))
    assert guard(prefix) is None
    assert guard(prefix+', {')=='segment_limit'
    wrapped='```json\n'+valid+'\n```'
    assert guard(wrapped)=='json_complete'
    obj=payload();obj['segments']*=2
    with pytest.raises(ValueError,match='duplicate'):compiled(obj)


def test_environment_keeps_gt_out_of_generation_and_counts_empty_failures():
    model,public,labels,args=setup_tiny()
    public['source_rc']=torch.tensor([[1,0]]);public['target_rc']=torch.tensor([[1,2]])
    public['task_text']=[task_text('pkmn',[1,0],[1,2])]
    truth=torch.ones(1,20,20);truth[0,0,1]=25
    labels=dict(true_costs=truth,optimal_cost=torch.tensor([3.]))
    calls=[]
    def generate(state,indices,prompt,max_new_tokens,system_prompt=None,output_guard=None):
        text=model.backbone.answer
        calls.append((prompt,system_prompt))
        return dict(text=text,finish_reason=output_guard(text),generated_tokens=100)
    model.backbone.generate_selected_report=generate
    model.backbone.answer=json.dumps(payload())
    encoded,*_=encode(model,public,args.budget)
    env=FrozenTilebasedSolverEnvironment(model.backbone,2,prompt_variant='navigation',model_representation='segments')
    first=env.evaluate(encoded,torch.tensor([[0,2]]),public,labels)[0]
    second=env.evaluate(encoded,torch.tensor([[0,2]]),public,dict(true_costs=torch.ones_like(truth),optimal_cost=torch.tensor([3.])))[0]
    assert first['status']==second['status']=='success'
    assert first['loss']==8 and second['loss']==0 and calls[0]==calls[1]
    assert first['streaming_guard_enabled'] and first['path_rc']==second['path_rc']
    obj=payload();obj['segments']=[];model.backbone.answer=json.dumps(obj)
    failed=env.evaluate(encoded,torch.tensor([[0,2]]),public,labels)[0]
    assert failed['failure_category']=='empty_connections' and failed['status']=='solver_failure'
    model.backbone.answer='{"segments":['+json.dumps(segment([30,0],[1,2]))
    stopped=env.evaluate(encoded,torch.tensor([[0,2]]),public,labels)[0]
    assert stopped['failure_category']=='invalid_segment' and stopped['status']=='model_failure'
    assert env.calls['modeling_calls']==4 and env.calls['formal_solver_calls']==3
    summary=summarize_outcomes([first,failed,stopped])
    assert summary['failure_categories']=={'empty_connections':1,'invalid_segment':1}


def test_coordinate_solver_reward_reaches_selector_and_flow():
    model,public,labels,args=setup_tiny()
    public['source_rc']=torch.tensor([[1,0]]);public['target_rc']=torch.tensor([[1,2]])
    public['task_text']=[task_text('pkmn',[1,0],[1,2])]
    truth=torch.ones(1,20,20);truth[0,1,1]=25
    labels=dict(true_costs=truth,optimal_cost=torch.tensor([3.]))
    def generate(state,indices,prompt,max_new_tokens,system_prompt=None):
        obj=payload()
        if 0 not in indices[0].tolist():obj['segments']=[segment([1,0],[1,2])]
        return json.dumps(obj)
    model.backbone.generate_selected_json=generate
    reference=GradientFlowSelectorV85(model.backbone,copy.deepcopy(model.selector),model.metric_head,copy.deepcopy(model.mask_flow))
    reference.selector.requires_grad_(False);reference.mask_flow.requires_grad_(False)
    env=FrozenTilebasedSolverEnvironment(model.backbone,2,prompt_variant='navigation',model_representation='segments')
    encoded,selection,positions,valid=encode(model,public,args.budget)
    loss,metrics=rl_loss(model,reference,env,encoded,selection,positions,valid,public,labels,args)
    loss.backward()
    assert metrics['informative_reward_group_rate']>0 and metrics['sampled_success_rate']==1
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.selector.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.mask_flow.parameters())
    assert all(p.grad is None for p in model.metric_head.parameters())


def test_cached_decoder_stops_when_guard_reports_completion():
    chunks=['{','"segments":','[]','}', 'unwanted continuation']
    forwarded=[]
    def language(**kwargs):
        forwarded.append(kwargs)
        logits=torch.full((1,1,7),-100.)
        logits[0,0,len(forwarded)-1]=1.
        return SimpleNamespace(last_hidden_state=logits,past_key_values='cache')
    obj=SimpleNamespace(
        model=SimpleNamespace(eval=lambda:None,generation_config=SimpleNamespace(eos_token_id=6),lm_head=lambda h:h),
        language_model=language,
        processor=SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda ids,**kw:''.join(chunks[i] for i in ids))),
        continue_vision=lambda *a,**kw:dict(deepstack_tokens=[]),
        _selected_chat_inputs=lambda *a:(torch.zeros(1,2,7),torch.ones(1,2,dtype=torch.long),
                                        torch.zeros(1,2,dtype=torch.bool),torch.zeros(4,1,2,dtype=torch.long)))
    result=FrozenQwen3VLPrunableBackbone.generate_selected_report(obj,{},torch.tensor([[0]]),'prompt',
        max_new_tokens=5,output_guard=SegmentOutputGuard('pkmn'))
    assert result['text']=='{"segments":[]}' and result['finish_reason']=='json_complete'
    assert result['generated_tokens']==4 and len(forwarded)==4
    assert all(x['past_key_values']=='cache' for x in forwarded[1:])


def test_public_prompt_has_no_node_numbering_or_full_cost_grid():
    prompt=modeling_user_prompt('navigation',task_text('pkmn',[1,0],[1,2]),'pkmn',[1,0],[1,2],
        list(range(100)),10,PKMN_LEGEND,'segments')
    assert 'node 1 is G' not in prompt and 'nodes_rc' not in prompt
    assert 'All visual regions are retained.' in prompt and 'segments' in prompt
    assert 'lake=6' in prompt and 'ocean=10' in prompt
