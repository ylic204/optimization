"""System-conditioned whole-image VQA using native Qwen; no selector or solver."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from tilebased_core_v85 import CONTRACT, DATASETS

SYSTEM_FILE = Path(__file__).resolve().parent / 'prompts/tilebased_vqa_system_v85.txt'
QUESTIONS = '''Use the top-down pixel-map environment defined in the system message. For this turn, perform visual question answering only. Answer questions 1-6 briefly in English and give visible evidence for each answer. State when a feature is absent or uncertain.
1. What major surface types and objects are visible, and where are they located?
2. Is there water? If so, describe its location, shape, texture, and boundaries.
3. Are there pale-yellow or tan regions? Do they look like connected paths, sand, shores, or something else? Explain using shape and surrounding features.
4. Do the green regions differ in texture or shape? Distinguish flat ground vegetation, trees, and other vegetation where possible.
5. Are there buildings or other artificial structures? Describe their approximate locations and visible features; exact counting is not required.
6. Which regions or categories are ambiguous, and what cannot be determined confidently from the image?
Describe the current image only. Do not produce a route, grid coordinates, terrain costs, an optimization model, or JSON in this VQA response.'''



def load_maps(root, split, indices):
    """Load only RGB, never true_costs, paths, source/goal labels or GT previews."""
    root = Path(root)
    config = json.loads((root/'dataset.json').read_text())
    if config.get('contract') != CONTRACT or not config.get('complete'):
        raise ValueError('expected a complete V8.5 Tilebased export')
    if config.get('dataset') != 'pkmn':
        raise ValueError('this environment prompt currently targets pkmn maps')
    spec = DATASETS['pkmn']
    side = spec['image_side']
    if config.get('image_side') != side:
        raise ValueError('unexpected image geometry')
    result = []
    for index in dict.fromkeys(indices):
        if index < 0:
            raise ValueError('map indices must be nonnegative')
        path = root/split/f'{index:06d}.npz'
        with np.load(path, allow_pickle=False) as data:
            rgb = data['image'].copy()
        if rgb.dtype != np.uint8 or rgb.shape != (side,side,3):
            raise ValueError(f'invalid RGB image: {path}')
        result.append((index, Image.fromarray(rgb)))
    return config, result


def build_messages(system_prompt, image, user_prompt=QUESTIONS):
    return [
        {'role':'system','content':system_prompt},
        {'role':'user','content':[{'type':'image','image':image},
                                 {'type':'text','text':user_prompt}]},
    ]


def load_qwen(model_path, device, load_4bit):
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    device = torch.device(device)
    kwargs = dict(local_files_only=True, attn_implementation='sdpa')
    if load_4bit:
        if device.type != 'cuda':
            raise ValueError('--load-4bit requires CUDA')
        kwargs.update(device_map={'':device.index or 0},quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16))
    else:
        kwargs['torch_dtype'] = torch.bfloat16 if device.type=='cuda' else torch.float32
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_path,**kwargs)
    if not load_4bit:
        model.to(device)
    model.eval().requires_grad_(False)
    processor = AutoProcessor.from_pretrained(model_path,local_files_only=True)
    return model, processor


def answer_image(model, processor, image, system_prompt, max_new_tokens):
    import torch
    messages = build_messages(system_prompt,image)
    chat = processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    side = image.width
    # PIL input is uint8 RGB. Let the official processor normalize and construct
    # all modality/position inputs; retain the original resolution for this check.
    inputs = processor(text=[chat],images=[image],return_tensors='pt',
                       min_pixels=side*side,max_pixels=side*side)
    inputs.pop('token_type_ids',None)  # Keep mm_token_type_ids when provided.
    grid = inputs['image_grid_thw'].tolist()
    patch = int(model.config.vision_config.patch_size)
    merge = int(model.config.vision_config.spatial_merge_size)
    if grid != [[1,side//patch,side//patch]] or side % (patch*merge):
        raise ValueError(f'processor changed native geometry: {grid}, image side={side}')
    input_length = inputs['input_ids'].shape[1]
    device = next(model.parameters()).device
    inputs = {k:v.to(device) for k,v in inputs.items()}
    with torch.inference_mode():
        output = model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=False,
                                repetition_penalty=1.0,use_cache=True)
    generated = output[0,input_length:].tolist()
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos,(list,tuple)) else [eos])
    ended = bool(generated and generated[-1] in eos)
    answer = processor.batch_decode(output[:,input_length:],skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)[0]
    return dict(answer=answer,finish_reason='eos' if ended else 'length',
                generated_tokens=len(generated)-int(ended),image_grid_thw=grid,
                visual_tokens=grid[0][0]*grid[0][1]*grid[0][2]//(merge*merge),
                chat_text=chat,input_tokens=input_length)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True)
    p.add_argument('--split',choices=['train','val','test'],default='val')
    p.add_argument('--map-indices',type=int,nargs='+',default=[0,3,6,9])
    p.add_argument('--vlm-model')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--load-4bit',action='store_true')
    p.add_argument('--max-new-tokens',type=int,default=1024)
    p.add_argument('--system-prompt-file',default=str(SYSTEM_FILE))
    p.add_argument('--prepare-only',action='store_true',help='export RGB and prompts without loading a model')
    p.add_argument('--output',required=True)
    a = p.parse_args()
    if not a.prepare_only and not a.vlm_model:p.error('--vlm-model is required for VQA generation')
    if a.max_new_tokens <= 0:p.error('--max-new-tokens must be positive')
    out = Path(a.output)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError('use a fresh output directory')
    system_prompt = Path(a.system_prompt_file).read_text(encoding='utf-8').strip()
    if not system_prompt:raise ValueError('system prompt must not be empty')
    config, maps = load_maps(a.data,a.split,a.map_indices)
    out.mkdir(parents=True,exist_ok=True)
    (out/'system_prompt.txt').write_text(system_prompt+'\n',encoding='utf-8')
    (out/'questions.txt').write_text(QUESTIONS+'\n',encoding='utf-8')
    settings = dict(vars(a),vqa_contract='tilebased_whole_image_vqa_v1',
                    dataset=config['dataset'],upstream_revision=config.get('upstream_revision'),
                    system_sha256=hashlib.sha256(system_prompt.encode()).hexdigest())
    for index,image in maps:image.save(out/f'map_{index:06d}_input.png')
    if a.prepare_only:
        (out/'config.json').write_text(json.dumps(settings,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(f'Prepared {len(maps)} original RGB images and both prompts in {out}. No VLM was run.')
        return
    import torch
    import transformers
    settings.update(torch_version=str(torch.__version__),transformers_version=transformers.__version__)
    (out/'config.json').write_text(json.dumps(settings,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    model,processor = load_qwen(a.vlm_model,a.device,a.load_4bit)
    with (out/'answers.jsonl').open('w',encoding='utf-8') as stream:
        for index,image in maps:
            started = time.perf_counter()
            result = answer_image(model,processor,image,system_prompt,a.max_new_tokens)
            result.update(map_index=index,split=a.split,input_image=f'map_{index:06d}_input.png',
                          system_prompt=system_prompt,user_prompt=QUESTIONS,
                          seconds=time.perf_counter()-started)
            stream.write(json.dumps(result,ensure_ascii=False)+'\n');stream.flush()
            (out/f'map_{index:06d}_answer.txt').write_text(result['answer']+'\n',encoding='utf-8')
            print(f"\nmap={index}, finish={result['finish_reason']}, visual_tokens={result['visual_tokens']}\n{result['answer']}",flush=True)


if __name__=='__main__':
    main()
