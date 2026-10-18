"""CPU contract, ground-truth, model isolation and training-gradient tests."""
import copy
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader
from tilebased_core_v85 import (CONTRACT, DATASETS, MODEL_SCHEMA, PKMN_LEGEND, coordinate,
    shortest_path, path_cost, verify_gt, validate_model, evaluate_path, task_text)
from tilebased_solver_environment_v85 import FrozenTilebasedSolverEnvironment, summarize_outcomes
from planning_dataset_tilebased_v85 import TilebasedDatasetV85, region_bounds
from gradient_flow_selector_v85 import GradientFlowSelectorV85, MaskGradientFlowND
from qwen3vl_selector_v79 import DecisionRelatedQwenSelector
from run_tilebased_v85 import TileCostHead, encode, proxy_paths, distill_loss, rl_loss, validate_checkpoint
from download_tilebased_v85 import download


def payload(costs=None):
    return dict(schema=MODEL_SCHEMA, source_rc=[0,0], target_rc=[19,19],
                node_costs=(np.ones((20,20)) if costs is None else costs).tolist())


def test_diagonal_node_cost_includes_both_endpoints():
    costs = np.array([[1.,25.],[25.,2.]])
    path = shortest_path(costs,[0,0],[1,1])
    assert path.tolist() == [[0,0],[1,1]]
    assert path_cost(costs,path) == 3.
    # A high-cost tile remains a valid node, even if visually called a wall.
    assert path_cost(costs,shortest_path(costs,[0,0],[0,1])) == 26.
    with pytest.raises(ValueError): coordinate([.0,1.],costs.shape)


def test_gt_mask_and_multiple_optima():
    costs = np.ones((3,3))
    mask = np.zeros((3,3)); mask[[1,0,1],[0,1,2]] = 1
    ordered,value = verify_gt(costs,mask,[1,0],[1,2])
    assert value == 3.
    result = evaluate_path([[1,0],[1,1],[1,2]],costs,[1,0],[1,2],value)
    assert result['optimal'] and result['relative_regret'] == 0
    mask[2,2] = 1
    with pytest.raises(ValueError): verify_gt(costs,mask,[1,0],[1,2])


@pytest.mark.parametrize('change',[
    {'source_rc':[1,0]}, {'node_costs':[[1.]]}, {'node_costs':[[float('nan')]*20]*20},
    {'node_costs':[[-1.]*20]*20}, {'node_costs':[[1.3]*20]*20}, {'python':'execute me'},
    {'target_rc':[19.0,19.0]},
])
def test_rejects_wrong_or_nonfinite_generated_models(change):
    with pytest.raises((ValueError,TypeError)):
        validate_model(json.dumps(dict(payload(),**change)),'pkmn',[0,0],[19,19])


def fixture_export(root):
    spec = DATASETS['pkmn']
    config = dict(contract=CONTRACT,complete=True,dataset='pkmn',upstream_revision='fixture',
                  grid_shape=[20,20],image_side=320,region_grid=10,coordinate_order='row_column_zero_based',
                  graph='fixture',cost_convention='including_source',legend=PKMN_LEGEND)
    (root/'dataset.json').write_text(json.dumps(config))
    folder = root/'train';folder.mkdir()
    costs = np.ones((20,20),np.float32)
    source,target = [0,0],[19,19]
    path = shortest_path(costs,source,target)
    mask = np.zeros((1,1,20,20),bool);mask[0,0,path[:,0],path[:,1]]=1
    ordered = np.full((1,1,400,2),-1,np.int16); ordered[0,0,:len(path)] = path
    np.savez(folder/'000000.npz',image=np.zeros((320,320,3),np.uint8),true_costs=costs,
             sources_rc=np.array([[source]]),targets_rc=np.array([target]),gt_path_mask=mask,
             gt_path_rc=ordered,gt_path_length=np.array([[len(path)]]),optimal_cost=np.array([[20.]]))
    record = dict(sample_id='fixture',map_file='000000.npz',target_index=0,source_index=0,
                  source_rc=source,target_rc=target,task_text=task_text('pkmn',source,target))
    (folder/'tasks.jsonl').write_text(json.dumps(record)+'\n')
    return root


def test_loader_keeps_labels_out_of_public_inputs_and_batches(tmp_path):
    dataset = TilebasedDatasetV85(fixture_export(tmp_path))
    batch = next(iter(DataLoader(dataset,batch_size=1)))
    assert batch['public_inputs']['image'].shape == (1,3,320,320)
    assert batch['public_inputs']['region_world_bounds'].shape == (1,100,4)
    assert batch['supervision']['gt_path_rc'].shape == (1,400,2)
    assert not set(batch['public_inputs']) & set(batch['supervision'])
    assert batch['supervision']['optimal_cost'].item() == 20.
    config = json.loads((tmp_path/'dataset.json').read_text());config['complete']=False
    (tmp_path/'dataset.json').write_text(json.dumps(config))
    with pytest.raises(ValueError,match='incomplete'):TilebasedDatasetV85(tmp_path)


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('features',torch.randn(1,4,8))
        self.prompts=[]
        self.answer=json.dumps(payload())
    def encode_inputs(self,images,texts):
        b=len(images)
        return {'region_visual_tokens':self.features.expand(b,-1,-1),
                'text_tokens':torch.ones(b,3,8),'text_valid':torch.ones(b,3,dtype=torch.bool),
                'input_ids':torch.ones(b,3,dtype=torch.long),'attention_mask':torch.ones(b,3),
                'vision_state':{'hidden_groups':self.features.expand(b,-1,-1)}}
    def continue_vision(self,state,region_mask=None,selected_indices=None,**kwargs):
        x=state['hidden_groups']
        if region_mask is not None:return (x*region_mask[...,None]).mean(1)
        return x.gather(1,selected_indices[...,None].expand(-1,-1,8)).mean(1)
    def fuse(self,ids,mask,visual):return visual
    def generate_selected_json(self,state,indices,prompt,max_new_tokens):
        self.prompts.append(prompt)
        return self.answer


def setup_tiny():
    torch.manual_seed(17)
    backbone=TinyBackbone()
    selector=DecisionRelatedQwenSelector(visual_dim=8,text_dim=8,aligned_dim=16,hidden=16,
                                         latent_dim=8,position_dim=8,layers=1,heads=4)
    model=GradientFlowSelectorV85(backbone,selector,TileCostHead(8,16,20),
        MaskGradientFlowND(latent_dim=8,task_dim=16,hidden=16,layers=1,heads=4,steps=2))
    model.metric_head.requires_grad_(False)
    public={'image':torch.zeros(1,3,320,320),'task_text':[task_text('pkmn',[0,0],[19,19])],
            'source_rc':torch.tensor([[0,0]]),'target_rc':torch.tensor([[19,19]]),
            'region_world_bounds':torch.tensor(region_bounds(2,(20,20)))[None],
            'dataset':['pkmn'],'legend':[PKMN_LEGEND],'sample_id':['fixture']}
    costs=torch.ones(1,20,20)
    costs[0,range(1,19),range(1,19)]=10.
    optimal=path_cost(costs[0].numpy(),shortest_path(costs[0].numpy(),[0,0],[19,19]))
    labels={'true_costs':costs,'optimal_cost':torch.tensor([optimal])}
    args=SimpleNamespace(budget=.5,mask_temperature=.35,teacher_step_size=.25,decision_temperature=.1,
                         lambda_gfm=1.,lambda_improve=.1,lambda_metric=.1,improve_margin=0.,
                         rl_group_size=8,rl_temperature=1.,rl_kl_weight=.02,rl_entropy_weight=.001)
    return model,public,labels,args


def test_real_solver_reward_uses_true_costs_but_prompt_never_uses_labels():
    model,public,labels,args=setup_tiny()
    encoded,*_=encode(model,public,args.budget)
    env=FrozenTilebasedSolverEnvironment(model.backbone,2)
    indices=torch.tensor([[0,2]])
    first=env.evaluate(encoded,indices,public,labels)[0]
    labels2={'true_costs':torch.ones_like(labels['true_costs']), 'optimal_cost':torch.tensor([20.])}
    second=env.evaluate(encoded,indices,public,labels2)[0]
    assert model.backbone.prompts[0]==model.backbone.prompts[1]
    assert first['status']=='success' and first['relative_regret']>0
    assert second['relative_regret']==0
    assert first['path_rc']==second['path_rc']
    model.backbone.answer='invalid JSON'
    failed=env.evaluate(encoded,indices,public,labels)[0]
    assert failed['status']=='model_failure' and failed['relative_regret'] is None
    assert summarize_outcomes([failed])['relative_regret_successful'] is None


def test_distillation_backpropagates_through_mask_with_frozen_proxy():
    model,public,labels,args=setup_tiny()
    encoded,selection,positions,valid=encode(model,public,args.budget)
    loss,metrics=distill_loss(model,encoded,selection,positions,valid,labels,proxy_paths(public,labels),args)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.selector.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.mask_flow.parameters())
    assert all(p.grad is None for p in model.metric_head.parameters())
    assert all(p.grad is None for p in model.backbone.parameters())


def test_rl_uses_solver_outcomes_and_updates_only_selector_flow():
    model,public,labels,args=setup_tiny()
    # This stub only substitutes frozen generation; parsing/solving/reward are real.
    def generate(state,indices,prompt,max_new_tokens):
        costs=labels['true_costs'][0].numpy() if 0 in indices[0].tolist() else np.ones((20,20))
        return json.dumps(payload(costs))
    model.backbone.generate_selected_json=generate
    reference=GradientFlowSelectorV85(model.backbone,copy.deepcopy(model.selector),model.metric_head,copy.deepcopy(model.mask_flow))
    reference.selector.requires_grad_(False);reference.mask_flow.requires_grad_(False)
    encoded,selection,positions,valid=encode(model,public,args.budget)
    env=FrozenTilebasedSolverEnvironment(model.backbone,2)
    loss,metrics=rl_loss(model,reference,env,encoded,selection,positions,valid,public,labels,args)
    loss.backward()
    assert torch.isfinite(loss) and metrics['advantage_abs']>0
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.selector.parameters())
    assert all(p.grad is None for p in reference.selector.parameters())
    assert all(p.grad is None for p in model.metric_head.parameters())


def test_offline_download_rejects_corrupt_or_missing_archives(tmp_path):
    with pytest.raises(FileNotFoundError):download(tmp_path,['pkmn'],['val'],offline=True)
    (tmp_path/'pkmn').mkdir();(tmp_path/'pkmn/val.npz').write_text('<html>error</html>')
    with pytest.raises(ValueError,match='checksum'):download(tmp_path,['pkmn'],['val'],offline=True)


def test_legacy_checkpoint_rejected_before_model_load(tmp_path):
    dataset=TilebasedDatasetV85(fixture_export(tmp_path))
    with pytest.raises(ValueError,match='contract'):validate_checkpoint({'input_mode':'bev'},dataset,SimpleNamespace())


def test_all_stage_cli_chain_with_tiny_frozen_model(tmp_path, monkeypatch):
    import shutil
    import sys
    import run_tilebased_v85 as runner
    data=tmp_path/'data';data.mkdir();fixture_export(data)
    for split in ('val','test'):shutil.copytree(data/'train',data/split)
    def build(args,device):
        model,_,_,_=setup_tiny()
        model.backbone.features=torch.randn(1,100,8)
        model.backbone.target_image_side=320
        model.backbone.text_hidden_dim=8
        return model
    monkeypatch.setattr(runner,'build_v85_model',build)
    previous=None
    for mode in ('metric','distill','rl','evaluate'):
        out=tmp_path/mode
        argv=['run_tilebased_v85.py','--mode',mode,'--data',str(data),'--vlm-model','test-double',
              '--device','cpu','--output',str(out),'--epochs','1','--hidden','16','--latent-dim','8',
              '--mask-steps','2','--rl-group-size','2']
        if previous:argv += ['--checkpoint',str(previous)]
        monkeypatch.setattr(sys,'argv',argv)
        runner.main()
        if mode!='evaluate':
            previous=out/'best.pt'
            assert previous.exists()
            saved=torch.load(previous,weights_only=True)
            assert saved['stage']==mode
            assert all(torch.isfinite(v).all() for v in saved['selector'].values())
        else:
            report=json.loads((out/'summary.json').read_text())
            assert set(report)=={'full','random','initial','learned'}
            assert report['full']['optimal_rate_all']==1.
