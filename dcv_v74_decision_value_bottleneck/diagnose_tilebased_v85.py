"""Paired native/custom Qwen checks on identical images/prompts, without training.

Tests full matrix output and short, publicly chosen cell queries. GT is read
only for scoring after generation. Blank-image control is optional and is
never mixed with the official planning evaluation.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from planning_dataset_tilebased_v85 import TilebasedDatasetV85
from qwen3vl_selector_v79 import FrozenQwen3VLPrunableBackbone
from tilebased_core_v85 import (DATASETS, build_model_prompt, evaluate_path, matrix_diagnostics,
                                parse_model_json, shortest_path, validate_model)


DIAGNOSTIC_CONTRACT = 'tilebased_diagnostics_v2'

def query_cells(source, target, grid):
    # Fixed geometry and task endpoints only. Never use GT/costs to choose cells.
    return list(dict.fromkeys([tuple(map(int, source)), tuple(map(int, target)),
                              (2,2),(2,grid//2),(grid//2,2),(grid//2,grid//2),(grid-3,grid-3)]))


def build_cells_prompt(dataset, cells, legend):
    grid = DATASETS[dataset]['grid']
    return (f'This image is exactly a {grid} row by {grid} column terrain map. '
            'Coordinates are zero-based [row,column], row down, column right; '
            f'cell [r,c] occupies horizontal [c/{grid},(c+1)/{grid}] and vertical [r/{grid},(r+1)/{grid}] '
            'fractions of the whole image. Read ONLY the requested cells from the image. '
            f'Terrain-cost legend: {legend}\nRequested cells: {list(map(list,cells))}. '
            'Return exactly {"cells":[{"row":r,"column":c,"terrain":"name","cost":number}, ...]} '
            'with one entry per requested cell in requested order. The outer value must be an object, not an array. '
            'Use terrain="unknown", cost=null if you cannot identify a cell; do not invent a uniform map. '
            'Do not output a path or a full matrix.')


def score_cells(answer, cells, true_costs, dataset):
    obj = parse_model_json(answer)
    # A bare array preserves every prediction: score its content, but expose the
    # outer-schema violation separately. Never repair entries, costs or order.
    bare_list = isinstance(obj,list)
    if bare_list:
        entries = obj
    elif isinstance(obj,dict) and set(obj) == {'cells'}:
        entries = obj['cells']
    else:
        raise ValueError('expected a cells object or a bare cell array')
    if not isinstance(entries,list) or len(entries) != len(cells):
        raise ValueError('expected one cells entry per query')
    correct, known, rows = 0, 0, []
    for item, (r,c) in zip(entries,cells):
        if (not isinstance(item,dict) or set(item) != {'row','column','terrain','cost'} or
                type(item['row']) is not int or type(item['column']) is not int or
                (item['row'],item['column']) != (r,c) or not isinstance(item['terrain'],str)):
            raise ValueError('cell coordinates/order/fields mismatch')
        value = item['cost']
        if value is not None and (type(value) not in (int,float) or not np.isfinite(value) or
                not np.isclose(value,DATASETS[dataset]['cost_values']).any()):
            raise ValueError('invalid cell cost')
        hit = value is not None and bool(np.isclose(value,true_costs[r,c],rtol=1e-5,atol=1e-5))
        correct += int(hit); known += int(value is not None)
        rows.append(dict(item,true_cost=float(true_costs[r,c]),correct=hit))
    return {'cell_accuracy':correct/len(cells),'known_fraction':known/len(cells),'cells_scored':rows,
            'strict_format_valid':not bare_list,
            'format_normalization':'bare_list_to_cells' if bare_list else 'none',
            'correct_cells':correct,'queried_cells':len(cells)}


def summarize(records):
    groups=defaultdict(list)
    for r in records:groups[(r['probe'],r['image_condition'],r['backend'])].append(r)
    report={}
    for key,rows in groups.items():
        success=[r for r in rows if r['status']=='success']
        item={'samples':len(rows),'successful_samples':len(success),'success_rate':len(success)/len(rows)}
        for metric in ('cost_cell_accuracy','cost_mae','relative_regret','same_path_as_uniform_cost','cell_accuracy','known_fraction'):
            values=[r[metric] for r in success if metric in r]
            if values:item[metric+'_successful']=float(np.mean(values))
        if key[0]=='cells':
            item['strict_format_success_rate'] = sum(bool(r.get('strict_format_valid')) for r in success)/len(rows)
            item['normalized_outer_format_samples'] = sum(r.get('format_normalization')=='bare_list_to_cells' for r in success)
            item['correct_cells'] = sum(r.get('correct_cells',0) for r in success)
            item['scored_cells'] = sum(r.get('queried_cells',0) for r in success)
        if key[0]=='matrix':
            item['optimal_rate_all']=sum(bool(r.get('optimal')) for r in rows)/len(rows)
            item['constant_matrix_rate_successful']=(float(np.mean([r['matrix_diagnostics']['constant_matrix'] for r in success])) if success else None)
        report['/'.join(key)]=item
    pairs=defaultdict(dict)
    for r in records:pairs[(r['sample_id'],r['probe'],r['image_condition'])][r['backend']]=r
    matched=[p for p in pairs.values() if set(p)=={'native','custom'}]
    report['paired']={'pairs':len(matched),'identical_answers':sum(p['native']['model_answer']==p['custom']['model_answer'] for p in matched)}
    return report



def rescore_saved_cells(source_file, ds, out):
    """Offline scoring of existing answers; no model loading or regeneration."""
    source = Path(source_file)
    records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError('empty diagnostic log')
    lookup = {r['sample_id']:i for i,r in enumerate(ds.records)}
    for item in records:
        if item['sample_id'] not in lookup:
            raise ValueError('sample absent from selected dataset split: '+item['sample_id'])
        if item['probe'] == 'matrix':
            continue  # Matrix outputs and original planning scores are untouched.
        if item['probe'] != 'cells':
            raise ValueError('unsupported saved diagnostic probe')
        sample = ds[lookup[item['sample_id']]]
        public, labels = sample['public_inputs'], sample['supervision']
        dataset = public['dataset']
        cells = query_cells(public['source_rc'].tolist(),public['target_rc'].tolist(),DATASETS[dataset]['grid'])
        marker = 'Requested cells: '
        if marker not in item['prompt']:
            raise ValueError('saved prompt lacks explicit cell queries')
        requested, _ = json.JSONDecoder().raw_decode(item['prompt'].split(marker,1)[1])
        if requested != list(map(list,cells)):
            raise ValueError('saved cell queries do not match this dataset/task')
        item['previous_scoring'] = {k:item[k] for k in ('status','error','cell_accuracy','strict_format_valid') if k in item}
        for key in ('error','cell_accuracy','known_fraction','cells_scored','strict_format_valid',
                    'format_normalization','correct_cells','queried_cells'):
            item.pop(key,None)
        item['status'] = 'model_failure'
        item['diagnostic_contract'] = DIAGNOSTIC_CONTRACT
        try:
            item.update(score_cells(item['model_answer'],cells,labels['true_costs'].numpy(),dataset))
            item['status'] = 'success'
        except (ValueError,TypeError,KeyError,OverflowError) as exc:
            item['error'] = str(exc)
    out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps(dict(diagnostic_contract=DIAGNOSTIC_CONTRACT,
        rescore_only=True,source_log=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        dataset_signature=ds.signature),indent=2)+'\n')
    (out/'outcomes.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in records))
    report = summarize(records)
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True)
    p.add_argument('--vlm-model')
    p.add_argument('--rescore-from',help='existing diagnostic outcomes.jsonl; no model or GPU required')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--load-4bit',action='store_true')
    p.add_argument('--split',choices=['train','val','test'],default='val')
    p.add_argument('--map-indices',type=int,nargs='+',default=[0,3,6,9])
    p.add_argument('--backends',nargs='+',choices=['native','custom'],default=['native','custom'])
    p.add_argument('--probes',nargs='+',choices=['cells','matrix'],default=['cells','matrix'])
    p.add_argument('--blank-control',action='store_true')
    p.add_argument('--prune-layer',type=int,default=4)
    p.add_argument('--max-new-tokens',type=int,default=4096)
    p.add_argument('--cell-max-new-tokens',type=int,default=768)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    if min(a.max_new_tokens,a.cell_max_new_tokens)<=0 or any(i<0 for i in a.map_indices):p.error('invalid token/map limits')
    out=Path(a.output)
    if out.exists() and any(out.iterdir()):raise FileExistsError('use a fresh diagnostic output directory')
    if not a.rescore_from and not a.vlm_model:p.error('--vlm-model is required unless --rescore-from is used')
    ds=TilebasedDatasetV85(a.data,a.split)
    if ds.config.get('navigation_protocol'):
        raise ValueError('legacy cell/matrix diagnostics require original data; use land VQA or regions evaluate for land data')
    if a.rescore_from:
        rescore_saved_cells(a.rescore_from,ds,out)
        return
    # Use first public S/G query per selected map, not four repeats of each image.
    found={}
    for i,r in enumerate(ds.records):found.setdefault(int(Path(r['map_file']).stem),i)
    missing=set(a.map_indices)-set(found)
    if missing:raise ValueError(f'map indices absent from this export: {sorted(missing)}')
    device=torch.device(a.device)
    model=FrozenQwen3VLPrunableBackbone(a.vlm_model,region_grid=ds.config['region_grid'],
        prune_layer=a.prune_layer,max_text_length=256,local_files_only=True,load_4bit=a.load_4bit,
        attn_implementation='sdpa',device=device)
    if model.target_image_side!=ds.config['image_side']:raise ValueError('native image size mismatch')
    import transformers
    out.mkdir(parents=True, exist_ok=True)
    (out/'config.json').write_text(json.dumps(dict(vars(a),diagnostic_contract=DIAGNOSTIC_CONTRACT,transformers_version=transformers.__version__,
        torch_version=str(torch.__version__),dataset_signature=ds.signature),indent=2)+'\n')
    records=[]
    conditions=['original','blank'] if a.blank_control else ['original']
    with (out/'outcomes.jsonl').open('w') as stream,(out/'prefill.jsonl').open('w') as parity_stream:
        for map_index in a.map_indices:
            batch=next(iter(DataLoader([ds[found[map_index]]],batch_size=1)))
            public,labels=batch['public_inputs'],batch['supervision']
            source=public['source_rc'][0].tolist();target=public['target_rc'][0].tolist()
            dataset=public['dataset'][0];grid=DATASETS[dataset]['grid']
            cells=query_cells(source,target,grid)
            full_indices=torch.arange(model.region_grid**2,device=device)[None]
            prompts={'matrix':build_model_prompt(public['task_text'][0],dataset,source,target,
                        full_indices[0].cpu().tolist(),model.region_grid,public['legend'][0]),
                     'cells':build_cells_prompt(dataset,cells,public['legend'][0])}
            for condition in conditions:
                image=public['image'].to(device)
                if condition=='blank':image=torch.full_like(image,.5)
                encoded = None
                if 'custom' in a.backends:
                    with torch.no_grad():
                        encoded=model.encode_inputs(image,public['task_text'])
                for probe in a.probes:
                    prompt=prompts[probe];budget=a.max_new_tokens if probe=='matrix' else a.cell_max_new_tokens
                    if set(a.backends)=={'native','custom'}:
                        parity=model.compare_full_prefill(image,encoded['vision_state'],prompt)
                        parity_stream.write(json.dumps(dict(parity,sample_id=public['sample_id'][0],probe=probe,image_condition=condition))+'\n');parity_stream.flush()
                    for backend in a.backends:
                        started=time.perf_counter()
                        generated=(model.generate_native_report(image,prompt,budget) if backend=='native' else
                                   model.generate_selected_report(encoded['vision_state'],full_indices,prompt,budget))
                        item={'diagnostic_contract':DIAGNOSTIC_CONTRACT,'sample_id':public['sample_id'][0],'probe':probe,'image_condition':condition,
                              'backend':backend,'model_answer':generated['text'],
                              'generation':{k:v for k,v in generated.items() if k!='text'},
                              'prompt':prompt,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
                              'status':'model_failure','generation_seconds':time.perf_counter()-started}
                        # Only now read the ground-truth cost matrix for scoring.
                        truth=labels['true_costs'][0].numpy()
                        try:
                            if probe=='matrix':
                                item['matrix_diagnostics']=matrix_diagnostics(generated['text'])
                                predicted=validate_model(generated['text'],dataset,source,target)
                                route=shortest_path(predicted,source,target)
                                item.update(evaluate_path(route,truth,source,target,float(labels['optimal_cost'][0])),
                                    cost_cell_accuracy=float(np.isclose(predicted,truth,rtol=1e-5,atol=1e-5).mean()),
                                    cost_mae=float(np.abs(predicted-truth).mean()),
                                    same_path_as_uniform_cost=bool(np.array_equal(route,shortest_path(np.ones_like(predicted),source,target))))
                            else:item.update(score_cells(generated['text'],cells,truth,dataset))
                            item['status']='success'
                        except (ValueError,TypeError,KeyError,OverflowError) as exc:item['error']=str(exc)
                        records.append(item);stream.write(json.dumps(item,allow_nan=False)+'\n');stream.flush()
                        print(f"map={map_index} {probe}/{condition}/{backend}: {item['status']}, finish={generated['finish_reason']}",flush=True)
    report=summarize(records)
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
