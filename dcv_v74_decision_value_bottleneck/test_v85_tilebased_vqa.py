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
    # No GT or task manifests exist: this VQA loader only needs RGB.
    np.savez(root/'val/000003.npz',image=rgb)
    def fail(*args,**kwargs):raise AssertionError('prepare-only must not load Qwen')
    monkeypatch.setattr(vqa,'load_qwen',fail)
    out=tmp_path/'vqa'
    monkeypatch.setattr(sys,'argv',['vqa','--data',str(root),'--map-indices','3',
        '--prepare-only','--output',str(out)])
    vqa.main()
    np.testing.assert_array_equal(np.asarray(Image.open(out/'map_000003_input.png')),rgb)
    assert (out/'system_prompt.txt').read_text().strip()==vqa.SYSTEM_FILE.read_text().strip()
    assert not (out/'answers.jsonl').exists()


def test_native_vqa_passes_real_system_role_image_and_modality_ids():
    torch=pytest.importorskip('torch')
    pil=Image.new('RGB',(320,320),'blue')
    class Processor:
        def apply_chat_template(self,messages,**kwargs):
            assert messages[0]=={'role':'system','content':'environment context'}
            assert messages[1]['content'][0]['image'] is pil
            assert messages[1]['content'][1]['text']==vqa.QUESTIONS
            return 'system and user chat'
        def __call__(self,**kwargs):
            assert kwargs['images']==[pil] and kwargs['text']==['system and user chat']
            assert kwargs['min_pixels']==kwargs['max_pixels']==320**2
            return dict(input_ids=torch.tensor([[1,4,5]]),attention_mask=torch.ones(1,3),
                pixel_values=torch.ones(1,3),image_grid_thw=torch.tensor([[1,20,20]]),
                token_type_ids=torch.zeros(1,3),mm_token_type_ids=torch.tensor([[0,0,1]]))
        def batch_decode(self,ids,**kwargs):
            assert ids.tolist()==[[9,2]]
            return ['图中可见水域。']
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.zeros(1))
            self.config=SimpleNamespace(vision_config=SimpleNamespace(patch_size=16,spatial_merge_size=2))
            self.generation_config=SimpleNamespace(eos_token_id=2)
        def generate(self,**kwargs):
            assert 'mm_token_type_ids' in kwargs and 'token_type_ids' not in kwargs
            assert not kwargs['do_sample'] and kwargs['use_cache']
            return torch.tensor([[1,4,5,9,2]])
    result=vqa.answer_image(Model(),Processor(),pil,'environment context',128)
    assert result['answer']=='图中可见水域。' and result['generated_tokens']==1
    assert result['visual_tokens']==100 and result['finish_reason']=='eos'
