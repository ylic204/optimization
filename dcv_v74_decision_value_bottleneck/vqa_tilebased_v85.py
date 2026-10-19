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
QUESTIONS = '''请遵循 system 中的俯视二维像素地图设定：图像上方不是天空，阶梯状边缘是瓦片绘制特征。请根据当前图片，用中文按 1—6 简短回答。每个判断都说明可见依据；没有看到就说未见明显证据，不确定就明确说明。
1. 概括图片包含哪些主要地表或对象，以及它们的大致分布。
2. 是否有水域？如果有，位于哪里，呈现什么形状或纹理？
3. 是否有浅黄色或浅棕色区域？如果有，它们更像道路、沙滩还是其他区域？说明依据和连接形状。
4. 绿色区域是否具有不同的纹理或形状？请区分你能看出的低矮地表植被、树木或其他植被；无法区分时说明。
5. 是否有建筑或明显的人工结构？如果有，描述大致位置和外观，不要求精确计数。
6. 哪些区域最容易混淆？说明你无法仅凭这张图确定的内容。
这一步只描述图像，不进行路径规划，也不输出地形成本。'''


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
