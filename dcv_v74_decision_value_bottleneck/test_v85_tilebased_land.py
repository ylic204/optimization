"""Land routing, reference consistency, constraint scoring and no GT-assisted repair."""
import copy
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from tilebased_core_v85 import (LAND_PROTOCOL, land_mask, shortest_path, path_cost,
                               task_text, validate_path, evaluate_path)
from tilebased_regions_v85 import REGION_SCHEMA, validate_regions, compile_regions, RegionOutputGuard
from tilebased_corridor_v85 import solve_corridor
from tilebased_solver_environment_v85 import FrozenTilebasedSolverEnvironment, summarize_outcomes
from planning_dataset_tilebased_v85 import TilebasedDatasetV85
from prepare_tilebased_land_v85 import prepare_land
from test_v85_tilebased import setup_tiny, fixture_export
from run_tilebased_v85 import encode, proxy_paths


def model_answer(blocked=None):
    return dict(schema=REGION_SCHEMA,source_rc=[2,0],target_rc=[2,4],
        regions=[dict(bounds_rc=[0,0,4,4],cost=1)],blocked_regions=blocked or [])


def compiled(obj):
    return compile_regions(validate_regions(json.dumps(obj),'pkmn',obj['source_rc'],obj['target_rc']))


def test_region_solver_can_turn_inside_wide_area_and_choose_cheaper_detour():
    obj=model_answer([[1,2,3,2]])
    obj['regions'].append(dict(bounds_rc=[0,0,0,4],cost=10))
    graph=compiled(obj);route,value=solve_corridor(graph)
    assert (route[:,0]==4).any() and not (route[:,0]==0).any()
    validate_path(route,[2,0],[2,4],graph['shape'],graph['allowed'],True)
    assert value==path_cost(graph['costs'],shortest_path(np.maximum(graph['costs'],1),[2,0],[2,4],graph['allowed'],True))
    obj['regions'].reverse();other=compiled(obj)
    assert graph['arcs']==other['arcs']  # overlapping costs are not order-dependent
    assert graph['modeled_arc_count']>len(route)-1  # actual routing freedom


def test_no_corner_cutting_no_endpoint_insertion_and_no_implicit_connections():
    allowed=np.array([[True,False],[False,True]])
    assert shortest_path(np.ones((2,2)),[0,0],[1,1],allowed,True) is None
    with pytest.raises(ValueError,match='corner'):validate_path([[0,0],[1,1]],[0,0],[1,1],(2,2),allowed,True)
    obj=model_answer();obj['regions']=[dict(bounds_rc=[2,0,2,0],cost=1),dict(bounds_rc=[2,4,2,4],cost=1)]
    assert solve_corridor(compiled(obj))==(None,None)
    obj['regions']=[]
    assert not compiled(obj)['endpoints_covered']
    obj=model_answer([[2,0,2,0]])
    assert not compiled(obj)['endpoints_covered']


@pytest.mark.parametrize('box',[[False,0,2,3],[0.,0,2,3],[0,0,20,4],[3,0,1,4],[0,0,1]])
def test_region_bounds_are_strict(box):
    obj=model_answer();obj['regions'][0]['bounds_rc']=box
    with pytest.raises(ValueError):compiled(obj)


def test_closed_json_guard_and_duplicate_regions():
    obj=model_answer();answer=json.dumps(obj);guard=RegionOutputGuard()
    assert guard(answer[:-1]) is None and guard(answer)=='json_complete'
    obj['regions']*=2
    with pytest.raises(ValueError,match='duplicate'):compiled(obj)


def test_hidden_water_does_not_change_solver_path_and_is_counted_as_failure():
    model,public,labels,args=setup_tiny()
    public.update(source_rc=torch.tensor([[2,0]]),target_rc=torch.tensor([[2,4]]),
                  navigation_protocol=[LAND_PROTOCOL],task_text=[task_text('pkmn',[2,0],[2,4],True)])
    model.backbone.answer=json.dumps(model_answer())
    encoded,*_=encode(model,public,args.budget)
    env=FrozenTilebasedSolverEnvironment(model.backbone,2,prompt_variant='navigation',model_representation='regions')
    truth=torch.ones((1,20,20))
    labels=dict(true_costs=truth,allowed_mask=torch.ones_like(truth,dtype=torch.bool),
                optimal_cost=torch.tensor([5.]),shortest_path_length=torch.tensor([4]))
    safe=env.evaluate(encoded,torch.tensor([[0,1]]),public,labels)[0]
    assert safe['status']=='success' and safe['spl']==1
    wr,wc=safe['path_rc'][2]
    labels['true_costs'][0,wr,wc]=6;labels['allowed_mask'][0,wr,wc]=False
    fail=env.evaluate(encoded,torch.tensor([[0,1]]),public,labels)[0]
    assert fail['status']=='constraint_failure' and fail['loss']==10
    assert fail['path_rc']==safe['path_rc'] and fail['modeling_prompt']==safe['modeling_prompt']
    assert fail['spl']==fail['cost_efficiency']==0
    assert summarize_outcomes([safe,fail])['constraint_failure_rate']==.5
    assert 'FORBIDDEN' in safe['system_prompt']
    assert 'All finite-cost cells are traversable' not in safe['system_prompt']
    # Teacher routes are training-only but must obey the same land protocol.
    candidates,_,valid=proxy_paths(public,labels)
    assert candidates[0,:,wr,wc].sum()==0 and valid.any()


def fixture_land(root):
    raw=root/'raw';raw.mkdir();fixture_export(raw)
    folder=raw/'train'
    with np.load(folder/'000000.npz') as a:data={k:a[k] for k in a.files}
    # Water stripe leaves a bottom passage; original diagonal GT now crosses water.
    data['true_costs'][:18,10]=6
    np.savez(folder/'000000.npz',**data)
    record=json.loads((folder/'tasks.jsonl').read_text())
    record['map_sha256']=hashlib.sha256(data['image'].tobytes()).hexdigest()
    records=[record]
    for name,point in [('water-start',[0,0]),('water-goal',[19,19]),('disconnected',None)]:
        changed=copy.deepcopy(data)
        if point is None:changed['true_costs'][:,10]=6
        else:changed['true_costs'][tuple(point)]=10
        filename=name+'.npz';np.savez(folder/filename,**changed)
        records.append(dict(record,sample_id=name,map_file=filename))
    (folder/'tasks.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    for split in ('val','test'):shutil.copytree(folder,raw/split)
    out=root/'land';report=prepare_land(raw,out)
    return raw,out,report


def test_rebuilt_references_endpoint_filter_spl_and_public_label_separation(tmp_path):
    raw,out,report=fixture_land(tmp_path)
    assert report['splits']['train']==dict(original_tasks=4,retained_tasks=1,water_start=1,water_goal=1,land_disconnected=1,retained_maps=1)
    sample=TilebasedDatasetV85(out)[0];public,labels=sample['public_inputs'],sample['supervision']
    assert 'allowed_mask' not in public and public['navigation_protocol']==LAND_PROTOCOL
    costs=labels['true_costs'].numpy();allowed=labels['allowed_mask'].numpy()
    length=int(labels['gt_path_length']);route=labels['gt_path_rc'][:length].numpy()
    validate_path(route,public['source_rc'],public['target_rc'],costs.shape,allowed,True)
    assert int(labels['shortest_path_length'])>19  # Chebyshev would be wrong here.
    with np.load(out/'train/000000.npz') as data:
        assert np.array_equal(data['original_gt_path_rc'][0,0,:20],np.arange(20)[:,None].repeat(2,axis=1))
    score=evaluate_path(route,costs,public['source_rc'],public['target_rc'],float(labels['optimal_cost']),allowed,True,int(labels['shortest_path_length']))
    assert score['spl']==1 and score['cost_efficiency']==1
    assert TilebasedDatasetV85(out).signature!=TilebasedDatasetV85(raw).signature
    with pytest.raises(ValueError):prepare_land(out,tmp_path/'again')


@pytest.mark.parametrize("polygon", [False, True])
def test_visualizer_preserves_invalid_actual_path_and_labels_land_reference(tmp_path, polygon):
    _,out,_=fixture_land(tmp_path)
    record=json.loads((out/'val/tasks.jsonl').read_text())
    with np.load(out/'val/000000.npz') as data:gt=data['gt_path_rc'][0,0,:int(data['gt_path_length'][0,0])].tolist()
    path=[[i,i] for i in range(20)]
    row=dict(sample_id=record['sample_id'],mode='full',status='constraint_failure',model_representation='regions',
             path_rc=path,gt_path_rc=gt,true_cost=None,compiled_graph=dict(regions=[dict(bounds_rc=[0,0,19,19],cost=1)],blocked_regions=[],arcs=[]))
    if polygon:
        row['compiled_graph']['regions']=[dict(polygon_rc=[[0,0],[0,20],[20,20],[20,0]],cost=1)]
        row['compiled_graph']['blocked_regions']=[dict(polygon_rc=[[1,1],[1,2],[2,2],[2,1]])]
    log=tmp_path/'outcomes.jsonl';log.write_text(json.dumps(row)+'\n')
    images=tmp_path/'images'
    subprocess.run([sys.executable,str(Path(__file__).with_name('visualize_tilebased_v85.py')),
        '--data',str(out),'--outcomes',str(log),'--output',str(images)],check=True,capture_output=True)
    saved=json.loads((images/'0000_details.json').read_text())
    assert saved['path_rc']==path and saved['measured']['spl']==0
    assert 'forbidden water' in saved['measured']['constraint_violation']
    assert (images/'0000_paths.png').exists()


def test_land_vqa_uses_forbidden_water_rules():
    from vqa_tilebased_v85 import navigation_question
    prompt=navigation_question(dict(source_rc=[2,0],target_rc=[2,4]),True,'lake=forbidden; ocean=forbidden; sand=1')
    assert 'Lake and ocean are forbidden' in prompt
    assert 'All cells in this benchmark are traversable' not in prompt
    assert 'lake=6' not in prompt


def test_concave_polygon_preserves_missing_corner_and_winding():
    from tilebased_regions_v85 import geometry, geometry_mask
    points=[[0,0],[0,5],[2,5],[2,2],[5,2],[5,0]]
    mask=geometry_mask(geometry({'polygon_rc':points},(20,20)),(20,20))
    assert mask.sum()==16 and mask[0,4] and mask[4,0] and not mask[4,4]
    assert np.array_equal(mask,geometry_mask(geometry({'polygon_rc':points[::-1]},(20,20)),(20,20)))


def test_multiple_polygons_and_irregular_holes_solver():
    obj=model_answer()
    obj['regions']=[{'polygon_rc':[[0,0],[0,5],[5,5],[5,0]],'cost':2},
                    {'polygon_rc':[[0,0],[0,5],[1,5],[1,0]],'cost':1}]
    obj['blocked_regions']=[{'polygon_rc':[[1,2],[1,3],[2,3],[2,4],[3,4],[3,2]]},
                            {'polygon_rc':[[4,0],[4,1],[5,1],[5,0]]}]
    graph=compiled(obj)
    route,cost=solve_corridor(graph)
    assert route is not None and cost>0
    assert not graph['allowed'][1,2] and not graph['allowed'][2,3] and not graph['allowed'][4,0]
    assert all(graph['allowed'][tuple(p)] for p in route)
    assert graph['costs'][0,0]==2  # max overlap, not order-dependent override
    obj['regions'].reverse()
    assert np.array_equal(graph['allowed'],compiled(obj)['allowed'])


@pytest.mark.parametrize('points',[
    [[0,0],[4,4],[0,4],[4,0]],
    [[0,0],[0,4],[0,2],[4,2],[4,0]],
    [[0,0],[0,4],[4,4],[0,0]],
    [[0,0],[0,21],[4,4]],
    [[0,0],[0,1],[0,2]],
])
def test_reject_invalid_polygons(points):
    obj=model_answer();obj['regions']=[{'polygon_rc':points,'cost':1}]
    with pytest.raises(ValueError):compiled(obj)


def test_polygon_prompt_and_old_schema_compatibility():
    from tilebased_regions_v85 import regions_user_prompt, LEGACY_REGION_SCHEMA
    text=regions_user_prompt('go','pkmn',[2,0],[2,4],range(100),10,{})
    assert 'polygon_rc' in text and 'GRID CORNERS' in text and 'bounds_rc' not in text
    obj=model_answer();obj['schema']=LEGACY_REGION_SCHEMA
    assert compiled(obj)['endpoints_covered']
