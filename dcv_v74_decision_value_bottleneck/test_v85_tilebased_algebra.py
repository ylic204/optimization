import copy
import json
from types import SimpleNamespace
import numpy as np
import pytest
from tilebased_algebra_v85 import (SCHEMA,ModelError,compile_model,solve_model,
                                  validate_output,decode_route,algebra_prompt)


def example():
    return dict(schema=SCHEMA,sets={'N':[0,1,2,3],'E':[0,1,2,3]},
        parameters={'tail':[0,1,0,2],'head':[1,3,2,3],'cost':[1,1,3,3],'start_cost':1},
        variables={'x':dict(index='E',type='binary',lb='0',ub='1')},
        objective=dict(sense='min',expression='start_cost + sum(cost[e]*x[e] for e in E)'),
        constraints=[dict(name='balance',forall={'n':'N'},
            lhs='sum(x[e] for e in E if tail[e] == n) - sum(x[e] for e in E if head[e] == n)',
            sense='==',rhs='1 if n == 0 else (-1 if n == 3 else 0)')],
        output=dict(variable='x',arcs=[
            dict(index=0,polyline_rc=[[0,0],[0,1]]),dict(index=1,polyline_rc=[[0,1],[0,2]]),
            dict(index=2,polyline_rc=[[0,0],[1,1]]),dict(index=3,polyline_rc=[[1,1],[0,2]])]))


def run(obj):
    m=compile_model(json.dumps(obj));binding=validate_output(m,(20,20));solution=solve_model(m)
    return m,solution,binding


def test_generated_objective_changes_actual_solver_choice():
    obj=example();m,s,b=run(obj)
    assert s['objective_value']==3 and s['constraint_count']==4
    assert decode_route(m,s,b,[0,0],[0,2]).tolist()==[[0,0],[0,1],[0,2]]
    obj['objective']['sense']='max';m,s,b=run(obj)
    assert s['objective_value']==7
    assert decode_route(m,s,b,[0,0],[0,2]).tolist()==[[0,0],[1,1],[0,2]]


def test_generated_constraint_changes_solution_and_is_not_repaired():
    obj=example();obj['constraints'].append(dict(name='close',forall={},lhs='x[0]',sense='==',rhs='0'))
    m,s,b=run(obj);assert s['objective_value']==7
    obj['constraints'].append(dict(name='close_other',forall={},lhs='x[2]',sense='==',rhs='0'))
    m,s,b=run(obj);assert s['status']=='infeasible'
    obj=example();obj['constraints']=[]
    m,s,b=run(obj)
    assert s['constraint_count']==0 and s['values']==[0.,0.,0.,0.]
    with pytest.raises(ModelError,match='missing'):decode_route(m,s,b,[0,0],[0,2])


def test_generic_integer_model_not_hardcoded_to_navigation():
    obj=example();obj['sets']={};obj['parameters']={}
    obj['variables']={'a':dict(index=None,type='integer',lb='0',ub='5')}
    obj['objective']={'sense':'max','expression':'2*a + 7'}
    obj['constraints']=[dict(name='capacity',forall={},lhs='3*a',sense='<=',rhs='8')]
    m=compile_model(json.dumps(obj));s=solve_model(m)
    assert s['values']==[2.] and s['objective_value']==11
    obj['variables']['a']['type']='continuous'
    assert solve_model(compile_model(json.dumps(obj)))['values'][0]==pytest.approx(8/3)


@pytest.mark.parametrize('expression',[
    'x[0]*x[1]','x[0]/x[1]','1 if x[0]>0 else 2','__import__("os")',
    'cost.__class__','cost[-1]','x[9]','missing+1','1/0','sum(x[e] for e in E for j in E)',
])
def test_invalid_expressions_rejected(expression):
    obj=example();obj['objective']['expression']=expression
    with pytest.raises(ModelError):compile_model(json.dumps(obj))


def test_format_distinct_from_model_errors():
    for text in ('{','{"a":1,"a":2}','{"a":NaN}'):
        with pytest.raises(ModelError) as e:compile_model(text)
        assert e.value.category=='format_error'
    with pytest.raises(ModelError) as e:compile_model('{}')
    assert e.value.category=='model_error'


def test_invalid_selected_branches_not_optimized_again():
    m,s,b=run(example());s['values']=[1,1,1,1]
    with pytest.raises(ModelError,match='branching'):decode_route(m,s,b,[0,0],[0,2])
    s['values']=[.5,.5,.5,.5]
    with pytest.raises(ModelError,match='0/1'):decode_route(m,s,b,[0,0],[0,2])


def test_no_hidden_terrain_or_dijkstra_in_formal_model(monkeypatch):
    import torch
    import tilebased_solver_environment_v85 as environment
    from tilebased_core_v85 import LAND_PROTOCOL,task_text
    from test_v85_tilebased import setup_tiny
    from run_tilebased_v85 import encode
    model,public,labels,args=setup_tiny()
    public.update(source_rc=torch.tensor([[0,0]]),target_rc=torch.tensor([[0,2]]),
        navigation_protocol=[LAND_PROTOCOL],task_text=[task_text('pkmn',[0,0],[0,2],True)])
    from test_v85_symbolic import example as symbolic_example
    model.backbone.answer=json.dumps(symbolic_example((20,20)));encoded,*_=encode(model,public,args.budget)
    env=environment.FrozenTilebasedSolverEnvironment(model.backbone,2,prompt_variant='navigation',model_representation='algebra')
    labels=dict(true_costs=torch.ones((1,20,20)),allowed_mask=torch.ones((1,20,20),dtype=torch.bool),
                optimal_cost=torch.tensor([3.]),shortest_path_length=torch.tensor([2]))
    # Existing shortest_path is permitted only for the independent uniform diagnostic,
    # never formal solving; replacing it changes diagnostic only.
    monkeypatch.setattr(environment,'shortest_path',lambda *a,**k:np.array([[19,19]]))
    indices=torch.tensor([[0,1,2,3]])
    good=env.evaluate(encoded,indices,public,labels)[0]
    assert good['status']=='success' and good['solver_result']['objective_value']==3
    assert good['compiled_model']['constraints'][0]['lower']==1
    r,c=good['path_rc'][1]
    labels['allowed_mask'][0,r,c]=False;labels['true_costs'][0,r,c]=6
    bad=env.evaluate(encoded,indices,public,labels)[0]
    assert bad['status']=='constraint_failure' and bad['path_rc']==good['path_rc']
    assert bad['generated_model']==good['generated_model'] and bad['solver_result']==good['solver_result']
    obj=symbolic_example((20,20));obj['model']['constraints']=[];model.backbone.answer=json.dumps(obj)
    missing=env.evaluate(encoded,indices,public,labels)[0]
    assert missing['failure_category']=='invalid_solution' and missing['solver_result']['constraint_count']==0
    model.backbone.answer='{}'
    invalid=env.evaluate(encoded,indices,public,labels)[0]
    assert invalid['failure_category']=='model_error' and 'solver_result' not in invalid


def test_prompt_is_shared_and_retrieves_path_template():
    from tilebased_prompts_v85 import modeling_system_prompt,modeling_user_prompt
    p=modeling_user_prompt('navigation','Find a minimum total terrain-cost LAND path','pkmn',[0,0],[0,2],[0,1],2,{},'algebra')
    assert 'variables' in p and 'constraints' in p and 'objective' in p
    assert 'tail[e]' in p and 'head[e]' in p and 'tilebased_symbolic_v1' in p
    assert 'polyline_rc' not in p
    assert 'sky' in modeling_system_prompt('navigation','pkmn','algebra')
    with pytest.raises(ValueError):modeling_system_prompt('legacy','pkmn','algebra')


def test_full_gate_pass_fail_and_configuration(tmp_path):
    from validate_tilebased_full_v85 import audit,require_full_validation,code_signature
    args=dict(mode='evaluate',split='val',model_representation='algebra',modeling_prompt='navigation',
              algebra_code_signature=code_signature(),dataset_signature='same',data=str(tmp_path/'data'),
              vlm_model='qwen',load_4bit=True,prune_layer=4,modeling_max_new_tokens=8192)
    (tmp_path/'args.json').write_text(json.dumps(args))
    rows=[dict(sample_id=str(i),mode='full',model_representation='algebra',retained_tokens=100,total_tokens=100,
               solver_result={'status':'optimal'},status='success') for i in range(32)]
    log=tmp_path/'outcomes.jsonl'
    log.write_text('\n'.join(json.dumps(r) for r in rows))
    report=audit(tmp_path);assert report['passed']
    path=tmp_path/'full_validation.json';path.write_text(json.dumps(report))
    require_full_validation(path,SimpleNamespace(**args),SimpleNamespace(signature='same'))
    args['vlm_model']='another'
    with pytest.raises(ValueError,match='configuration'):require_full_validation(path,SimpleNamespace(**args),SimpleNamespace(signature='same'))
    rows[0]['status']='constraint_failure';log.write_text('\n'.join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError,match='stale'):require_full_validation(path,SimpleNamespace(**args),SimpleNamespace(signature='same'))
    log.write_text(json.dumps(rows[1]));assert not audit(tmp_path)['passed']
