"""One-map native/custom VQA and symbolic-model diagnostics. No training or GT.
Native-only mode loads Qwen directly, bypassing the custom backbone entirely.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

from tilebased_prompts_v85 import navigation_environment_prompt

VQA_SYSTEM = navigation_environment_prompt()
VQA_PROMPT = '''Answer briefly in English, at most 150 words. No JSON or mathematical model.
1. Is the entire map water, or are there land areas? Describe their approximate locations.
2. Where are the main water boundaries and land passages?
3. Can you identify distinct terrain appearances that could affect navigation?
State uncertainty when appropriate. Do not list nodes, edges, grid cells or routes.'''


def read_public_sample(root, split, sample_index):
    import numpy as np
    from tilebased_core_v85 import CONTRACT, LAND_PROTOCOL, DATASETS
    root=Path(root);config=json.loads((root/'dataset.json').read_text())
    if config.get('contract')!=CONTRACT or not config.get('complete'):raise ValueError('expected complete V8.5 dataset')
    if config.get('navigation_protocol')!=LAND_PROTOCOL:raise ValueError('use land data, e.g. pkmn_land_v3')
    record=None
    with (root/split/'tasks.jsonl').open() as stream:
        for i,line in enumerate(s for s in stream if s.strip()):
            if i==sample_index:record=json.loads(line);break
    if record is None:raise ValueError('sample index outside task manifest')
    name=record['map_file']
    if Path(name).name!=name or not name.endswith('.npz'):raise ValueError('invalid map basename')
    with np.load(root/split/name,allow_pickle=False) as data:rgb=data['image'].copy()
    side=config['image_side']
    if rgb.shape!=(side,side,3) or rgb.dtype!=np.uint8:raise ValueError('invalid RGB image')
    shape=config['grid_shape']
    for key in ('source_rc','target_rc'):
        rc=record[key]
        if len(rc)!=2 or any(type(v) is not int or not 0<=v<shape[k] for k,v in enumerate(rc)):raise ValueError('invalid public endpoint')
    public={k:record[k] for k in ('sample_id','source_rc','target_rc','task_text','map_file')}
    return config,public,rgb


def output_diagnostics(text):
    """Structural observations, not a visual-correctness score."""
    result={'json_complete':False}
    try:obj=json.loads(text)
    except ValueError:return result
    result['json_complete']=True
    if not isinstance(obj,dict):return result
    result['top_level_keys']=list(obj)
    result['has_model']=isinstance(obj.get('model'),dict)
    regions=obj.get('scene',{}).get('regions') if isinstance(obj.get('scene'),dict) else None
    if isinstance(regions,list):
        result['region_count']=len(regions)
        result['duplicate_region_count']=len(regions)-len({json.dumps(r,sort_keys=True) for r in regions})
    return result


def official_inputs(processor,image,system,prompt,device):
    messages=[{'role':'system','content':system},
              {'role':'user','content':[{'type':'image','image':image},{'type':'text','text':prompt}]}]
    chat=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    inputs=processor(text=[chat],images=[image],return_tensors='pt',
                     min_pixels=image.width*image.height,max_pixels=image.width*image.height)
    inputs.pop('token_type_ids',None)
    return {k:v.to(device) for k,v in inputs.items()},chat


def native_generate(model,processor,inputs,budget,trace):
    import torch
    from transformers import TextStreamer
    class TraceStreamer(TextStreamer):
        def __init__(self):
            super().__init__(processor.tokenizer,skip_prompt=True,skip_special_tokens=True)
            self.text=''
        def on_finalized_text(self,text,stream_end=False):
            self.text+=text;trace(self.text)
    with torch.inference_mode():
        output=model.generate(**inputs,max_new_tokens=budget,do_sample=False,num_beams=1,
                              repetition_penalty=1.0,use_cache=True,streamer=TraceStreamer())
    tokens=output[0,inputs['input_ids'].shape[1]:].tolist()
    eos=model.generation_config.eos_token_id
    eos=set(eos if isinstance(eos,(list,tuple)) else [eos])
    ended=bool(tokens and tokens[-1] in eos);content=tokens[:-1] if ended else tokens
    return dict(text=processor.tokenizer.decode(content,skip_special_tokens=True),
                finish_reason='eos' if ended else 'length',generated_tokens=len(content),backend='native',
                input_tokens=int(inputs['input_ids'].shape[1]),image_grid_thw=inputs['image_grid_thw'].tolist() if 'image_grid_thw' in inputs else None)


def input_parity(official,custom):
    import torch
    result={}
    for key in ('input_ids','attention_mask','image_grid_thw','pixel_values','mm_token_type_ids'):
        a,b=official.get(key),custom.get(key)
        info={'official_shape':list(a.shape) if a is not None else None,
              'custom_shape':list(b.shape) if b is not None else None}
        if a is not None and b is not None and a.shape==b.shape:
            info['equal']=bool(torch.equal(a,b))
            info['max_abs_difference']=float((a.float()-b.float()).abs().max())
        result[key]=info
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--vlm-model',required=True)
    p.add_argument('--split',choices=['train','val','test'],default='val')
    p.add_argument('--sample-index',type=int,default=0,help='zero-based TASK manifest index, not map number')
    p.add_argument('--device',default='cuda:0');p.add_argument('--load-4bit',action='store_true')
    p.add_argument('--backends',nargs='+',choices=['native','custom'],default=['native','custom'])
    p.add_argument('--probes',nargs='+',choices=['vqa','symbolic'],default=['vqa','symbolic'])
    p.add_argument('--vqa-max-new-tokens',type=int,default=384)
    p.add_argument('--model-max-new-tokens',type=int,default=4096)
    p.add_argument('--prune-layer',type=int,default=4)
    p.add_argument('--blank-control',action='store_true')
    p.add_argument('--skip-prefill-check',action='store_true')
    p.add_argument('--quiet',action='store_true')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    if a.sample_index<0 or min(a.vqa_max_new_tokens,a.model_max_new_tokens)<=0:p.error('invalid index/token limit')
    out=Path(a.output).resolve()
    if out.exists() and any(out.iterdir()):raise FileExistsError('choose a fresh --output directory')
    config,public,rgb=read_public_sample(a.data,a.split,a.sample_index)
    import numpy as np
    import torch
    import transformers
    from PIL import Image
    from tilebased_generation_v85 import GenerationTrace
    from tilebased_prompts_v85 import modeling_system_prompt,modeling_user_prompt
    torch.manual_seed(1234)
    out.mkdir(parents=True,exist_ok=True)
    (out/'public_task.json').write_text(json.dumps(public,indent=2)+'\n')
    Image.fromarray(rgb).save(out/'input.png')
    device=torch.device(a.device);backbone=None
    if 'custom' in a.backends:
        from qwen3vl_selector_v79 import FrozenQwen3VLPrunableBackbone
        backbone=FrozenQwen3VLPrunableBackbone(a.vlm_model,region_grid=config['region_grid'],
            prune_layer=a.prune_layer,max_text_length=256,local_files_only=True,
            load_4bit=a.load_4bit,attn_implementation='sdpa',device=device)
        model,processor=backbone.model,backbone.processor
        if backbone.target_image_side!=config['image_side']:raise ValueError('image geometry mismatch')
    else:
        from vqa_tilebased_v85 import load_qwen
        model,processor=load_qwen(a.vlm_model,a.device,a.load_4bit)
    model.eval().requires_grad_(False)
    full=list(range(config['region_grid']**2))
    prompts={
        'vqa':(VQA_SYSTEM,VQA_PROMPT),
        'symbolic':(modeling_system_prompt('navigation',config['dataset'],'algebra'),
                    modeling_user_prompt('navigation',public['task_text'],config['dataset'],public['source_rc'],
                        public['target_rc'],full,config['region_grid'],config['legend'],'algebra'))}
    (out/'config.json').write_text(json.dumps(dict(vars(a),torch_version=str(torch.__version__),
        transformers_version=transformers.__version__,generation_config=model.generation_config.to_dict(),
        stop_policy='EOS or token budget only, no JSON/array guard in either backend',
        input_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest()),indent=2,default=str)+'\n')
    records=[]
    for condition in (['original','blank'] if a.blank_control else ['original']):
        current=rgb if condition=='original' else np.full_like(rgb,128)
        image=Image.fromarray(current);image.save(out/f'{condition}.png')
        tensor=torch.from_numpy(current.copy()).permute(2,0,1).float().unsqueeze(0).to(device)/255
        state=None;encode_error=None
        if backbone is not None:
            try:
                with torch.inference_mode():state=backbone.encode_inputs(tensor,[public['task_text']])['vision_state']
            except Exception as exc:encode_error=f'{type(exc).__name__}: {exc}'
        for probe in dict.fromkeys(a.probes):
            system,prompt=prompts[probe];budget=a.vqa_max_new_tokens if probe=='vqa' else a.model_max_new_tokens
            prefix=f'{condition}_{probe}'
            (out/f'{prefix}.prompt.json').write_text(json.dumps({'system':system,'user':prompt},indent=2)+'\n')
            official,chat=official_inputs(processor,image,system,prompt,device)
            (out/f'{prefix}.chat.txt').write_text(chat)
            if backbone is not None and not a.skip_prefill_check:
                parity={}
                try:
                    with torch.inference_mode():
                        shared=backbone._native_image_inputs(tensor,prompt,system)
                        parity['processor_inputs']=input_parity(official,shared)
                        del shared
                        if state is not None:parity['shared_input_prefill']=backbone.compare_full_prefill(tensor,state,prompt,system)
                        else:parity['encode_error']=encode_error
                except Exception as exc:parity['error']=f'{type(exc).__name__}: {exc}'
                (out/f'{prefix}.parity.json').write_text(json.dumps(parity,indent=2)+'\n')
                print('[Input/prefill check] '+json.dumps(parity),flush=True)
            for backend in dict.fromkeys(a.backends):
                name=f'{prefix}_{backend}';trace=GenerationTrace(out/'raw',name,not a.quiet)
                record=dict(sample_id=public['sample_id'],condition=condition,probe=probe,backend=backend,
                    prompt_sha256=hashlib.sha256((system+'\n'+prompt).encode()).hexdigest(),
                    input_rgb_sha256=hashlib.sha256(current.tobytes()).hexdigest(),raw_file=str(trace.path),status='runtime_error')
                started=time.perf_counter()
                try:
                    if backend=='native':generated=native_generate(model,processor,official,budget,trace)
                    else:
                        if state is None:raise RuntimeError(encode_error)
                        with torch.inference_mode():generated=backbone.generate_selected_report(state,
                            torch.tensor([full],device=device),prompt,budget,system_prompt=system,
                            output_guard=None,text_callback=trace)
                    answer=generated.pop('text');trace.finish(answer,generated['finish_reason'])
                    record.update(model_answer=answer,generation=generated,status='generated',**output_diagnostics(answer))
                    (out/f'{name}.txt').write_text(answer)
                    if probe=='symbolic':
                        from tilebased_symbolic_v85 import compile_symbolic
                        try:
                            compiled,evidence=compile_symbolic(answer,tuple(config['grid_shape']),public['source_rc'],public['target_rc'],full,config['region_grid'])
                            record['compile_status']='valid';record['expanded_variables']=len(compiled['labels'])
                            record['predicted_blocked_fraction']=float(np.array(evidence['blocked']).mean())
                        except (ValueError,TypeError,KeyError,OverflowError) as exc:
                            record['compile_status']='invalid';record['compile_error']=str(exc)
                except Exception as exc:
                    record.update(error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
                record['seconds']=time.perf_counter()-started
                records.append(record)
                with (out/'outcomes.jsonl').open('a') as f:f.write(json.dumps(record,default=str)+'\n')
                print(f"\n[Probe] {name}: {record['status']}, finish={record.get('generation',{}).get('finish_reason')}, compile={record.get('compile_status')}",flush=True)
                if 'error' in record:print(record['error'],flush=True)
                if device.type=='cuda':torch.cuda.empty_cache()
            del official
    summary=[{k:r[k] for k in ('condition','probe','backend','status','generation','json_complete','has_model',
                'region_count','duplicate_region_count','compile_status','compile_error','predicted_blocked_fraction','error') if k in r} for r in records]
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('\nSaved diagnostics: '+str(out),flush=True)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
