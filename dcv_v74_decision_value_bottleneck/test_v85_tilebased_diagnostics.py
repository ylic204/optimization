"""Diagnostics must expose failures without repairing models or using GT as input."""
import json
import sys
import numpy as np
import pytest
from tilebased_core_v85 import matrix_diagnostics, validate_model, PKMN_LEGEND
from diagnose_tilebased_v85 import query_cells, build_cells_prompt, score_cells
from test_v85_tilebased import payload, fixture_export


@pytest.mark.parametrize('shape', [(21,20),(20,10),(17,20)])
def test_matrix_shape_failures_are_logged_not_repaired(shape):
    answer = json.dumps(payload(np.ones(shape)))
    report = matrix_diagnostics(answer)
    assert report['matrix_shape'] == list(shape)
    assert report['constant_matrix']
    with pytest.raises(ValueError, match='shape mismatch'):
        validate_model(answer,'pkmn',[0,0],[19,19])


def test_malformed_and_constant_matrix_diagnostics():
    assert not matrix_diagnostics('{')['json_valid']
    assert not matrix_diagnostics('{"a":1,"a":2}')['json_valid']
    ragged = matrix_diagnostics('{"node_costs":[[1,2],[1]]}')
    assert ragged['matrix_row_lengths'] == [2,1]
    assert 'matrix_shape' not in ragged
    mixed = np.ones((20,20)); mixed[1,1] = 25
    report = matrix_diagnostics(json.dumps(payload(mixed)))
    assert not report['constant_matrix']
    assert report['predicted_cost_histogram'] == {'1.0':399,'25.0':1}


def test_cell_query_uses_public_geometry_and_scores_unknown_as_incorrect():
    cells = query_cells([2,2],[19,19],20)
    assert len(cells) == len(set(cells)) and cells[0] == (2,2)
    prompt = build_cells_prompt('pkmn',cells,PKMN_LEGEND)
    assert 'Requested cells' in prompt and 'zero-based' in prompt
    items = [dict(row=r,column=c,terrain='unknown',cost=None) for r,c in cells]
    result = score_cells(json.dumps({'cells':items}),cells,np.ones((20,20)),'pkmn')
    assert result['cell_accuracy'] == result['known_fraction'] == 0
    items[0].update(terrain='sand',cost=1)
    result = score_cells(json.dumps({'cells':items}),cells,np.ones((20,20)),'pkmn')
    assert result['cell_accuracy'] == pytest.approx(1/len(cells))
    items[0]['column'] = 0
    with pytest.raises(ValueError,match='coordinates'):
        score_cells(json.dumps({'cells':items}),cells,np.ones((20,20)),'pkmn')


def test_diagnostic_cli_pairs_identical_inputs_and_records_both_probes(tmp_path,monkeypatch):
    import diagnose_tilebased_v85 as diagnostic
    pytest.importorskip('transformers')
    data = tmp_path/'data'; data.mkdir(); fixture_export(data)
    calls = []
    class FakeBackbone:
        region_grid, target_image_side = 10,320
        def __init__(self,*args,**kwargs): pass
        def encode_inputs(self,image,text):
            return {'vision_state':{'image':image.clone()}}
        def compare_full_prefill(self,image,state,prompt):
            assert image.equal(state['image'])
            return {'first_token_equal':True,'max_abs_logit_difference':0}
        def report(self,image,prompt,budget,backend):
            calls.append((backend,image.clone(),prompt,budget))
            if 'Requested cells:' in prompt:
                cells=query_cells([0,0],[19,19],20)
                answer={'cells':[dict(row=r,column=c,terrain='sand',cost=1) for r,c in cells]}
            else: answer=payload()
            return dict(text=json.dumps(answer),finish_reason='eos',generated_tokens=100,
                        max_new_tokens=budget,backend=backend)
        def generate_native_report(self,image,prompt,budget):
            return self.report(image,prompt,budget,'native')
        def generate_selected_report(self,state,indices,prompt,budget):
            assert indices.tolist()==[list(range(100))]
            return self.report(state['image'],prompt,budget,'custom')
    monkeypatch.setattr(diagnostic,'FrozenQwen3VLPrunableBackbone',FakeBackbone)
    output=tmp_path/'results';output.mkdir()  # Existing EMPTY directory is permitted.
    monkeypatch.setattr(sys,'argv',['diagnose','--data',str(data),'--vlm-model','fixture',
        '--device','cpu','--split','train','--map-indices','0','--blank-control','--output',str(output)])
    diagnostic.main()
    records=[json.loads(x) for x in (output/'outcomes.jsonl').read_text().splitlines()]
    assert len(records)==8 and all(r['status']=='success' for r in records)
    assert len((output/'prefill.jsonl').read_text().splitlines())==4
    for native,custom in zip(calls[::2],calls[1::2]):
        assert native[1].equal(custom[1]) and native[2:]==custom[2:]
    summary=json.loads((output/'summary.json').read_text())
    assert summary['paired']=={'pairs':4,'identical_answers':4}
