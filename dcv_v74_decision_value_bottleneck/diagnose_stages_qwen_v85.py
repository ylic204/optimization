"""Native-only staged diagnosis. Text modeling is synthetic, never a visual result."""
import argparse
import copy
import json
from pathlib import Path
import traceback

from tilebased_prompts_v85 import navigation_environment_prompt, modeling_system_prompt, modeling_user_prompt
from diagnose_symbolic_qwen_v85 import read_public_sample, official_inputs, native_generate, output_diagnostics

# Public, explicitly constructed fixture, NOT labels extracted from a dataset image.
FIXTURE = {
    'shape': [3, 5], 'source_rc': [1, 0], 'target_rc': [1, 4],
    'description': 'A 3-row by 5-column grid. All cells initially have cost 1 and are traversable. '
        'Cells (1,1), (1,2), (1,3) have cost 8 and remain traversable. '
        'Cell (0,2) is water and forbidden. All other cells retain cost 1. '
        'Start (1,0), goal (1,4). Eight-neighbor moves; no diagonal water corner cutting. '
        'Minimize the sum of visited cell costs including start and goal. No other resource constraints.',
    'scene': {'regions': [
        {'geometry': {'bounds_rc': [0,0,2,4]}, 'cost': 1, 'blocked': False},
        {'geometry': {'bounds_rc': [1,1,1,3]}, 'cost': 8, 'blocked': False},
        {'geometry': {'bounds_rc': [0,2,0,2]}, 'cost': 25, 'blocked': True}]}}


def scene_prompt(config, task):
    return f'''Inspect the supplied overhead map. Output scene parameters ONLY, no mathematical model.
Task: {task['task_text']}
S={task['source_rc']}; G={task['target_rc']}; grid shape={config['grid_shape']}.
Coordinates are zero-based [row,column]. Public legend: {config['legend']}.
Return exactly {{"scene":{{"regions":[...]}}}}.
Each region must contain exactly geometry, cost, blocked. A format-only example is
{{"geometry":{{"bounds_rc":[0,0,1,1]}},"cost":1.5,"blocked":false}}.
Do not copy this example's location; use the image. Never place bounds_rc or polygon_rc directly
beside cost: both belong INSIDE geometry. Geometry is {{"bounds_rc":[r0,c0,r1,c1]}}
with inclusive cell bounds, or {{"polygon_rc":[[r,c],...]}} using grid-corner coordinates.
Use non-self-intersecting polygons with 3..64 vertices, without repeating the closing vertex.
Use at most 64 task-relevant regions. Do not enumerate cells or duplicate a region.
Water is blocked=true, cost=25 as an unused placeholder. Other terrain uses public costs and blocked=false.
Overlap combines maximum cost and logical OR of blocked; do not cover land with a water bounding box.
Uncovered cells remain unknown (cost 25, not automatically blocked).
Describe evidence from this image. Do not label the whole map blocked unless the entire image supports it.
No variables, objective, constraints, nodes, edges, or proposed path in this stage.'''


def text_model_prompt():
    from tilebased_symbolic_v85 import MODEL_EXAMPLE
    return '''This is a TEXT-ONLY synthetic modeling diagnostic. No image is provided or needed.
Use these supplied facts, not visual inference:\n''' + FIXTURE['description'] + '''
The supplied scene is already encoded. Output ONLY {"model":{...}}; do not repeat or alter the scene.
model must contain sets, parameters, variables, objective, constraints, output.
Declare sets with grid_cells and neighbor_pairs (of the cell set, connectivity=8), never explicit lists.
Parameters bind scene.cost, scene.blocked, scene.corner_blocked, task.source, task.goal,
space.tail, space.head, space.outgoing, space.incoming, space.step_length (or scalar constants).
These arrays are compiler-expanded from supplied facts. Declare binary movement variables indexed
by the adjacency set, a minimum-cost objective, flow balance and all necessary movement constraints.
Constraints have name, optional forall, lhs, sense, rhs. Expressions support sum, indexing,
linear arithmetic and parameter-only conditionals. Output binds variable and index.
The compiler adds no missing mathematical expressions. Do not enumerate nodes/edges or output a route.
The following syntax/formulation guide is the same example used in the joint test:
''' + json.dumps(MODEL_EXAMPLE, separators=(',', ':'))


def check_scene(answer, shape, indices, region_grid):
    from tilebased_symbolic_v85 import build_scene
    obj = json.loads(answer)
    if not isinstance(obj, dict) or set(obj) != {'scene'}:
        raise ValueError('expected only scene at top level')
    scene = build_scene(obj['scene'], tuple(shape), indices, region_grid)
    report = output_diagnostics(answer)
    report.update(schema_valid=True, visual_correctness='requires_human_review',
                  blocked_fraction=float(scene['blocked'].mean()),
                  unknown_fraction=float((~scene['covered']).mean()))
    report['structural_sanity_passed'] = bool(obj['scene']['regions']) and report.get('duplicate_region_count', 0) == 0 and not scene['blocked'].all()
    return report, {k: v.tolist() for k,v in scene.items()}


def check_text_model(answer):
    import numpy as np
    from tilebased_symbolic_v85 import compile_symbolic, decode_symbolic, SCHEMA
    from tilebased_algebra_v85 import solve_model
    from tilebased_core_v85 import shortest_path, path_cost, validate_path
    obj = json.loads(answer)
    if not isinstance(obj,dict) or set(obj) != {'model'}:
        raise ValueError('expected only model at top level')
    # Attach supplied scene ONLY: never fill missing variables, objectives or constraints.
    wrapped = dict(schema=SCHEMA, scene=copy.deepcopy(FIXTURE['scene']), model=obj['model'])
    m, evidence = compile_symbolic(json.dumps(wrapped), tuple(FIXTURE['shape']), FIXTURE['source_rc'], FIXTURE['target_rc'], [0], 1)
    result = solve_model(m)
    report = {'compile_status': 'valid', 'solver_status': result['status'], 'fixture_passed': False}
    artifact = {'supplied_scene_and_generated_model': wrapped, 'solver_result': result, 'evidence': evidence}
    if result['status'] != 'optimal':
        return report, artifact
    route = decode_symbolic(m, result, FIXTURE['source_rc'], FIXTURE['target_rc'])
    # Independently written fixture truth; not inferred from the generated model.
    costs = np.ones((3,5)); costs[1,1:4] = 8; costs[0,2] = 25
    allowed = np.ones((3,5),dtype=bool); allowed[0,2] = False
    validate_path(route, FIXTURE['source_rc'], FIXTURE['target_rc'], (3,5), allowed, True)
    reference = shortest_path(costs, FIXTURE['source_rc'], FIXTURE['target_rc'], allowed, True)
    true_cost = path_cost(costs, route); optimum = path_cost(costs, reference)
    report.update(true_cost=true_cost, reference_cost=optimum, objective_value=result['objective_value'],
                  fixture_passed=bool(abs(true_cost-optimum)<1e-6 and abs(result['objective_value']-true_cost)<1e-6))
    artifact.update(path_rc=route.tolist(), reference_path_rc=reference.tolist())
    return report, artifact


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', required=True, choices=['vqa','scene','model','joint'])
    p.add_argument('--data'); p.add_argument('--split', default='val', choices=['train','val','test'])
    p.add_argument('--sample-index', type=int, default=0)
    p.add_argument('--vlm-model', required=True); p.add_argument('--device', default='cuda:0')
    p.add_argument('--load-4bit', action='store_true'); p.add_argument('--max-new-tokens', type=int)
    p.add_argument('--output', required=True); p.add_argument('--quiet', action='store_true')
    a = p.parse_args()
    if a.sample_index<0 or (a.max_new_tokens is not None and a.max_new_tokens<1): p.error('invalid index/budget')
    if a.stage!='model' and not a.data: p.error('--data is required for visual stages')
    out=Path(a.output).resolve()
    if out.exists() and any(out.iterdir()): raise FileExistsError('choose a fresh output directory')
    out.mkdir(parents=True,exist_ok=True)
    system=navigation_environment_prompt(); image=None
    if a.stage=='model':
        prompt=text_model_prompt()
        (out/'fixture.json').write_text(json.dumps(FIXTURE,indent=2))
    else:
        from PIL import Image
        config,task,rgb=read_public_sample(a.data,a.split,a.sample_index)
        image=Image.fromarray(rgb); image.save(out/'input.png')
        (out/'public_task.json').write_text(json.dumps(task,indent=2))
        full=list(range(config['region_grid']**2))
        if a.stage=='vqa':
            prompt=('Describe the visible water, land and their boundaries in at most 200 English words. '
                    'Use image-relative positions. Separate observed appearance from rule-based traversability. '
                    'A drawn road or gap is not required for traversable land. Do not infer barriers from dense texture. '
                    'No JSON, grid enumeration, mathematical model or claimed solved path.')
        elif a.stage=='scene': prompt=scene_prompt(config,task)
        else:
            system=modeling_system_prompt('navigation',config['dataset'],'algebra')
            prompt=modeling_user_prompt('navigation',task['task_text'],config['dataset'],task['source_rc'],task['target_rc'],full,config['region_grid'],config['legend'],'algebra')
    (out/'prompt.json').write_text(json.dumps({'system':system,'user':prompt},indent=2))
    from vqa_tilebased_v85 import load_qwen
    from tilebased_generation_v85 import GenerationTrace
    import torch
    import transformers
    torch.manual_seed(1234)
    model,processor=load_qwen(a.vlm_model,a.device,a.load_4bit)
    if image is None:
        messages=[{'role':'system','content':system},{'role':'user','content':prompt}]
        chat=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=processor(text=[chat],return_tensors='pt')
        inputs.pop('token_type_ids',None)
        inputs={k:v.to(a.device) for k,v in inputs.items()}
    else: inputs,chat=official_inputs(processor,image,system,prompt,a.device)
    (out/'chat.txt').write_text(chat)
    (out/'config.json').write_text(json.dumps(dict(vars(a),torch_version=str(torch.__version__),transformers_version=transformers.__version__,generation_config=model.generation_config.to_dict()),indent=2,default=str))
    trace=GenerationTrace(out/'raw',a.stage,not a.quiet)
    report={'stage':a.stage, 'backend':'native', 'status':'runtime_error'}
    try:
        generated=native_generate(model,processor,inputs,a.max_new_tokens or (512 if a.stage=='vqa' else 4096),trace)
        answer=generated.pop('text'); trace.finish(answer,generated['finish_reason'])
        (out/'model_answer.txt').write_text(answer)
        report.update(status='generated',generation=generated,**output_diagnostics(answer))
        if a.stage=='scene':
            checks,artifact=check_scene(answer,config['grid_shape'],full,config['region_grid'])
            report.update(checks); (out/'scene_raster.json').write_text(json.dumps(artifact))
        elif a.stage=='model':
            checks,artifact=check_text_model(answer)
            report.update(checks); (out/'model_check.json').write_text(json.dumps(artifact,indent=2))
        elif a.stage=='joint':
            from tilebased_symbolic_v85 import compile_symbolic
            m,e=compile_symbolic(answer,tuple(config['grid_shape']),task['source_rc'],task['target_rc'],full,config['region_grid'])
            report.update(compile_status='valid',visual_correctness='requires_human_review')
            (out/'scene_raster.json').write_text(json.dumps(e))
            (out/'symbolic_model.json').write_text(json.dumps(m['symbolic_raw'],indent=2))
        else: report['visual_correctness']='requires_human_review'
    except Exception as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
    (out/'summary.json').write_text(json.dumps(report,indent=2,default=str))
    print(json.dumps(report,indent=2,default=str),flush=True)
    print('Saved: '+str(out),flush=True)

if __name__=='__main__': main()
