"""Whole-image VQA input isolation and official generation plumbing."""
import json
import sys
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest
from tilebased_core_v85 import CONTRACT
import vqa_tilebased_v85 as vqa


def test_prepare_original_rgb_and_system_prompt_without_gt_or_model(tmp_path,monkeypatch):
    root=tmp_path/'data';(root/'val').mkdir(parents=True)
    (root/'dataset.json').write_text(json.dumps(dict(contract=CONTRACT,complete=True,dataset='pkmn',image_side=320)))
    rgb=np.zeros((320,320,3),np.uint8);rgb[:,:,2]=151
    # Only RGB and public query metadata exist: no GT arrays are available.
    np.savez(root/'val/000003.npz',image=rgb)
    record=dict(map_file='000003.npz',sample_id='pkmn/val/000003/0/0',
                source_rc=[2,3],target_rc=[18,19])
    (root/'val/tasks.jsonl').write_text(json.dumps(record)+'\n')
    def fail(*args,**kwargs):raise AssertionError('prepare-only must not load Qwen')
    monkeypatch.setattr(vqa,'load_qwen',fail)
    out=tmp_path/'vqa'
    monkeypatch.setattr(sys,'argv',['vqa','--data',str(root),'--map-indices','3',
        '--prepare-only','--output',str(out)])
    vqa.main()
    np.testing.assert_array_equal(np.asarray(Image.open(out/'map_000003_input.png')),rgb)
    assert (out/'system_prompt.txt').read_text().strip()==vqa.SYSTEM_FILE.read_text().strip()
    question=(out/'map_000003_question.txt').read_text()
    assert 'S=[2, 3]' in question and 'G=[18, 19]' in question
    assert vqa.PKMN_LEGEND in question
    assert not (out/'answers.jsonl').exists()


def test_native_vqa_passes_real_system_role_image_and_modality_ids():
    torch=pytest.importorskip('torch')
    pil=Image.new('RGB',(320,320),'blue')
    user_prompt=vqa.navigation_question(dict(source_rc=[1,2],target_rc=[8,9]))
    class Processor:
        def apply_chat_template(self,messages,**kwargs):
            assert messages[0]=={'role':'system','content':'environment context'}
            assert messages[1]['content'][0]['image'] is pil
            assert messages[1]['content'][1]['text']==user_prompt
            return 'system and user chat'
        def __call__(self,**kwargs):
            assert kwargs['images']==[pil] and kwargs['text']==['system and user chat']
            assert kwargs['min_pixels']==kwargs['max_pixels']==320**2
            return dict(input_ids=torch.tensor([[1,4,5]]),attention_mask=torch.ones(1,3),
                pixel_values=torch.ones(1,3),image_grid_thw=torch.tensor([[1,20,20]]),
                token_type_ids=torch.zeros(1,3),mm_token_type_ids=torch.tensor([[0,0,1]]))
        def batch_decode(self,ids,**kwargs):
            assert ids.tolist()==[[9,2]]
            return ['A visible band may make the direct crossing costly.']
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.zeros(1))
            self.config=SimpleNamespace(vision_config=SimpleNamespace(patch_size=16,spatial_merge_size=2))
            self.generation_config=SimpleNamespace(eos_token_id=2)
        def generate(self,**kwargs):
            assert 'mm_token_type_ids' in kwargs and 'token_type_ids' not in kwargs
            assert not kwargs['do_sample'] and kwargs['use_cache']
            return torch.tensor([[1,4,5,9,2]])
    result=vqa.answer_image(Model(),Processor(),pil,'environment context',128,user_prompt)
    assert result['answer']=='A visible band may make the direct crossing costly.' and result['generated_tokens']==1
    assert result['visual_tokens']==100 and result['finish_reason']=='eos'


def test_public_task_selection_excludes_privileged_fields_and_rejects_invalid_queries(tmp_path):
    (tmp_path/'val').mkdir()
    records=[dict(map_file='000003.npz',sample_id=f'query-{i}',source_rc=[i,2],
                  target_rc=[9,8],true_costs='SECRET_GT',gt_path_rc='SECRET_PATH') for i in range(2)]
    manifest=tmp_path/'val/tasks.jsonl'
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in records))
    tasks=vqa.load_public_tasks(tmp_path,'val',[3],1)
    assert tasks[3]['sample_id']=='query-1' and tasks[3]['source_rc']==[1,2]
    assert 'SECRET' not in json.dumps(tasks)+vqa.navigation_question(tasks[3])
    for bad_index in (-1,2):
        with pytest.raises(ValueError):vqa.load_public_tasks(tmp_path,'val',[3],bad_index)
    with pytest.raises(ValueError,match='unavailable'):
        vqa.load_public_tasks(tmp_path,'val',[4],0)
    records[0]['source_rc']=[20,0]
    manifest.write_text(json.dumps(records[0])+'\n')
    with pytest.raises(ValueError,match='coordinates'):
        vqa.load_public_tasks(tmp_path,'val',[3],0)
