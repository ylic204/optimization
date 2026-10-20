"""Fixed subset selection and navigation metrics independent of exact GT masks."""
import json
import shutil
import numpy as np
import pytest
from tilebased_core_v85 import evaluate_path
from tilebased_solver_environment_v85 import summarize_outcomes
from filter_tilebased_v85 import filter_dataset
from planning_dataset_tilebased_v85 import TilebasedDatasetV85
from test_v85_tilebased import fixture_export


def test_spl_length_and_cost_are_distinct_and_gt_matching_unnecessary():
    costs=np.ones((3,3));costs[1,1]=10
    # Direct is distance-optimal but expensive; detour minimizes terrain cost.
    direct=evaluate_path([[1,0],[1,1],[1,2]],costs,[1,0],[1,2],3)
    detour=evaluate_path([[1,0],[0,0],[0,1],[0,2],[1,2]],costs,[1,0],[1,2],3)
    assert direct['spl']==1 and direct['cost_efficiency']==.25
    assert detour['spl']==.5 and detour['cost_efficiency']==.6
    alt=evaluate_path([[1,0],[2,1],[1,2]],costs,[1,0],[1,2],3)
    assert alt['spl']==1 and alt['cost_efficiency']==1 and alt['optimal']
    stationary=evaluate_path([[0,0]],costs,[0,0],[0,0],1)
    assert stationary['spl']==1 and stationary['path_length']==0


def test_failures_contribute_zero_to_both_efficiencies():
    success=dict(status='success',loss=0,optimal=True,relative_regret=0,true_cost=3,
                 retained_tokens=15,total_tokens=100,environment_seconds=0,spl=.5,cost_efficiency=1,path_length=4)
    failure=dict(success,status='model_failure',loss=10,optimal=None,spl=1,cost_efficiency=1)
    result=summarize_outcomes([success,failure])
    assert result['sr']==.5 and result['spl']==.25
    assert result['success_weighted_cost_efficiency']==.5
    failed=summarize_outcomes([failure])
    assert failed['spl']==0 and failed['success_weighted_cost_efficiency']==0


def test_fixed_water_goal_filter_preserves_ids_labels_and_splits(tmp_path):
    raw=tmp_path/'original';raw.mkdir();fixture_export(raw)
    original=(raw/'train'/'tasks.jsonl').read_text()
    original_record=json.loads(original)
    with np.load(raw/'train'/'000000.npz') as data: arrays={k:data[k] for k in data.files}
    arrays['true_costs'][19,19]=6
    np.savez(raw/'train'/'000001.npz',**arrays)
    excluded=dict(original_record,sample_id='water_goal',map_file='000001.npz')
    (raw/'train'/'tasks.jsonl').write_text(original+json.dumps(excluded)+'\n')
    for split in ('val','test'):shutil.copytree(raw/'train',raw/split)
    out=tmp_path/'filtered';report=filter_dataset(raw,out)
    for split in ('train','val','test'):
        assert report['splits'][split]['retained_tasks']==1
        assert report['splits'][split]['excluded_water_goal']==1
        assert (out/split/'tasks.jsonl').read_text()==original
        assert (out/split/'000000.npz').read_bytes()==(raw/split/'000000.npz').read_bytes()
        assert not (out/split/'000001.npz').exists()
        assert json.loads((out/split/'excluded_tasks.jsonl').read_text())['exclusion_reason']=='water_goal'
    assert TilebasedDatasetV85(out).signature!=TilebasedDatasetV85(raw).signature
    assert len(TilebasedDatasetV85(out))==1
    with pytest.raises(ValueError,match='already filtered'):filter_dataset(out,tmp_path/'again')
    with pytest.raises(ValueError,match='new or empty'):filter_dataset(raw,out)
