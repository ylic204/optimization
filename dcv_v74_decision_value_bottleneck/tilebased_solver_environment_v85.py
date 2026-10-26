"""Frozen VLM -> generated mathematical model -> solver -> independent reward.
Historical parameter-only environments remain explicit reproducibility options.
"""
import hashlib
import inspect
import time
import numpy as np
import torch
from tilebased_core_v85 import validate_model, shortest_path, evaluate_path, matrix_diagnostics, LAND_PROTOCOL

from tilebased_corridor_v85 import validate_corridor, compile_corridor, solve_corridor, corridor_evaluation
from tilebased_segments_v85 import validate_segments, SegmentOutputGuard
from tilebased_regions_v85 import validate_regions, compile_regions, RegionOutputGuard, region_evaluation
from tilebased_prompts_v85 import modeling_user_prompt, modeling_system_prompt

ENVIRONMENT_CONTRACT = 'frozen_tilebased_llm_exact_mincost_v1'
CORRIDOR_ENVIRONMENT_CONTRACT = 'frozen_tilebased_llm_sg_corridor_v1'
REGION_ENVIRONMENT_CONTRACT = 'frozen_tilebased_land_regions_v3'
SEGMENT_ENVIRONMENT_CONTRACT = 'frozen_tilebased_llm_sg_segments_v2'

def environment_contract(representation):
    if representation == 'algebra':
        from tilebased_algebra_v85 import CONTRACT
        return CONTRACT
    if representation == 'regions':return REGION_ENVIRONMENT_CONTRACT
    if representation == 'matrix':return ENVIRONMENT_CONTRACT
    if representation == 'corridor':return CORRIDOR_ENVIRONMENT_CONTRACT
    if representation == 'segments':return SEGMENT_ENVIRONMENT_CONTRACT
    raise ValueError('unknown model representation')


class FrozenTilebasedSolverEnvironment:
    def __init__(self, backbone, region_grid, max_new_tokens=4096, failure_cost=10., prompt_variant='legacy', model_representation='matrix'):
        if failure_cost <= 0 or not np.isfinite(failure_cost) or max_new_tokens <= 0:
            raise ValueError('positive failure cost and decode budget required')
        if prompt_variant not in ('legacy','navigation'):
            raise ValueError('unknown modeling prompt variant')
        self.contract = environment_contract(model_representation)
        self.model_representation = model_representation
        self.prompt_variant = prompt_variant
        self.backbone, self.region_grid = backbone, region_grid
        self.max_new_tokens, self.failure_cost = max_new_tokens, failure_cost
        # Attempts, including failed generation/solving. Training proxy solves
        # are deliberately excluded; these counters describe the formal env.
        self.calls = dict(modeling_calls=0, formal_solver_calls=0, diagnostic_solver_calls=0)

    @torch.no_grad()
    def evaluate(self, encoded, indices, public_inputs, supervision):
        indices = indices.sort(-1).values
        results = []
        for row in range(len(indices)):
            start = time.perf_counter()
            source = public_inputs['source_rc'][row].cpu().numpy()
            target = public_inputs['target_rc'][row].cpu().numpy()
            dataset = public_inputs['dataset'][row]
            chosen = indices[row].cpu().tolist()
            protocol = public_inputs.get('navigation_protocol',['original']*len(indices))[row]
            land = protocol == LAND_PROTOCOL
            if land != (self.model_representation in ('regions','algebra')):
                raise ValueError('land protocol requires algebra or legacy regions; other legacy representations require original protocol')
            system = modeling_system_prompt(self.prompt_variant,dataset,self.model_representation)
            prompt = modeling_user_prompt(self.prompt_variant,public_inputs['task_text'][row], dataset, source, target,
                                        chosen, self.region_grid, public_inputs['legend'][row],self.model_representation)
            state = {k: (tuple(x[row:row+1] for x in v) if k == 'position_groups' else v[row:row+1])
                     for k, v in encoded['vision_state'].items()}
            item = {'sample_id': public_inputs['sample_id'][row], 'model_representation':self.model_representation, 'environment_contract':self.contract, 'status': 'model_failure',
                    'loss': self.failure_cost, 'selected_indices': chosen,
                    'retained_tokens': len(chosen), 'total_tokens': self.region_grid**2,
                    'relative_regret': None, 'true_cost': None, 'optimal': None, 'path_rc': None,
                    'navigation_protocol': protocol, 'spl': 0.0, 'cost_efficiency': 0.0, 'path_length': None, 'shortest_path_length': None}
            try:
                item['modeling_prompt'] = prompt
                item['modeling_prompt_variant'] = self.prompt_variant
                item['system_prompt'] = system
                item['prompt_sha256'] = hashlib.sha256((system+'\n'+prompt).encode()).hexdigest()
                generation_kwargs = {'system_prompt':system} if system else {}
                self.calls['modeling_calls'] += 1
                if hasattr(self.backbone, 'generate_selected_report'):
                    if self.model_representation in ('segments','regions','algebra') and 'output_guard' in inspect.signature(self.backbone.generate_selected_report).parameters:
                        generation_kwargs['output_guard'] = RegionOutputGuard() if land else SegmentOutputGuard(dataset)
                    generation = self.backbone.generate_selected_report(state, indices[row:row+1], prompt, self.max_new_tokens, **generation_kwargs)
                    answer = generation['text']
                    item['generation'] = {k:v for k,v in generation.items() if k != 'text'}
                else:
                    answer = self.backbone.generate_selected_json(state, indices[row:row+1], prompt, self.max_new_tokens, **generation_kwargs)
                item['model_answer'] = answer
                item['streaming_guard_enabled'] = 'output_guard' in generation_kwargs
                finish = item.get('generation', {}).get('finish_reason')
                if finish in ('segment_limit','invalid_segment','repeated_segment','invalid_segment_separator'):
                    item['failure_category'] = finish
                    raise ValueError(f'coordinate output guard stopped generation: {finish}')
                if finish == 'length':
                    item['failure_category'] = 'length_limit'
                if self.model_representation == 'algebra':
                    from tilebased_algebra_v85 import compile_model, solve_model, validate_output, decode_route, ModelError
                    if finish == 'length':raise ModelError('length_limit','generated algebra was truncated')
                    compiled = compile_model(answer)
                    from tilebased_core_v85 import DATASETS
                    binding = validate_output(compiled, (DATASETS[dataset]['grid'],)*2)
                    item['generated_model'] = compiled['raw']
                    item['compiled_model'] = dict(variable_labels=compiled['labels'],
                        objective_coefficients=compiled['c'].tolist(), objective_constant=compiled['constant'],
                        objective_sense=compiled['sense'], lower=compiled['lower'].tolist(), upper=compiled['upper'].tolist(),
                        integrality=compiled['integrality'].tolist(),
                        constraints=[dict(coefficients={str(i):float(v) for i,v in zip(compiled['A'].getrow(r).indices,compiled['A'].getrow(r).data)},
                            lower=None if not np.isfinite(compiled['cl'][r]) else float(compiled['cl'][r]),
                            upper=None if not np.isfinite(compiled['cu'][r]) else float(compiled['cu'][r])) for r in range(len(compiled['cl']))])
                    item['status'] = 'solver_failure'
                    self.calls['formal_solver_calls'] += 1
                    solution = solve_model(compiled)
                    item['solver_result'] = solution
                    if solution['status'] != 'optimal':raise ModelError(solution['status'],solution['message'])
                    item['status'] = 'model_failure'
                    route = decode_route(compiled,solution,binding,source,target)
                    item['generated_objective_value'] = solution['objective_value']
                    item['compiled_graph'] = dict(arcs=[[list(a),list(b),0.] for path in binding[1].values() for a,b in zip(path,path[1:])])
                elif land:
                    modeled = validate_regions(answer,dataset,source,target)
                    graph = compile_regions(modeled)
                    item['status'] = 'solver_failure'
                    item.update({key:graph[key] for key in ('region_count','blocked_region_count','modeled_cell_count','modeled_arc_count')})
                    item['compiled_graph'] = dict(nodes_rc=np.argwhere(graph['allowed']).tolist(),
                        regions=[dict(geom,cost=cost) for geom,cost in modeled['regions']],
                        blocked_regions=modeled['blocked'],
                        arcs=[[list(a),list(b),c] for (a,b),c in sorted(graph['arcs'].items())])
                    self.calls['formal_solver_calls'] += 1
                    route,estimated = solve_corridor(graph) if graph['endpoints_covered'] else (None,None)
                    item['graph_connected'] = route is not None
                    if not graph['endpoints_covered']:item['failure_category'] = 'endpoints_not_covered'
                elif self.model_representation in ('corridor','segments'):
                    validator = validate_segments if self.model_representation == 'segments' else validate_corridor
                    modeled = validator(answer,dataset,source,target)
                    item['proposed_segment_count'] = len(modeled['edges'])
                    item['empty_connections'] = not modeled['edges']
                    item['constant_segment_costs'] = bool(modeled['edges']) and len({e[2] for e in modeled['edges']}) == 1
                    graph = compile_corridor(modeled)
                    item['status'] = 'solver_failure'
                    self.calls['formal_solver_calls'] += 1
                    route,estimated = solve_corridor(graph)
                    item['graph_connected'] = route is not None
                    item['compiled_graph'] = dict(nodes_rc=[list(p) for p in modeled['nodes']],
                        arcs=[[list(a),list(b),c] for (a,b),c in sorted(graph['arcs'].items())])
                else:
                    item['matrix_diagnostics'] = matrix_diagnostics(answer)
                    modeled_costs = validate_model(answer, dataset, source, target)
                    item['status'] = 'solver_failure'
                    self.calls['formal_solver_calls'] += 1
                    route = shortest_path(modeled_costs, source, target)
                if route is None:
                    item.setdefault('failure_category', 'empty_connections' if item.get('empty_connections') else 'disconnected_graph')
                    raise ValueError('generated model has no S-to-G path; no fallback routing is applied')
                # Privileged labels are accessed only AFTER generation and solving.
                truth = supervision['true_costs'][row].cpu().numpy()
                allowed = supervision['allowed_mask'][row].cpu().numpy() if land else None
                # Retain invalid predicted routes for visualization, never repair them.
                item['path_rc'] = route.tolist()
                if land:
                    item['status'] = 'constraint_failure'
                    item['failure_category'] = 'water_or_corner_violation'
                measured = evaluate_path(route, truth, source, target, float(supervision['optimal_cost'][row]),
                    allowed=allowed, no_corner_cut=land,
                    shortest_length=int(supervision['shortest_path_length'][row]) if land else None)
                item.pop('failure_category', None)
                item.update(measured, status='success')
                item['loss'] = min(measured['relative_regret'], self.failure_cost*.999)
                truth = supervision['true_costs'][row].cpu().numpy()
                if self.model_representation == 'algebra':
                    pass  # Do not replace the generated objective with a diagnostic routing objective.
                elif land:
                    self.calls['diagnostic_solver_calls'] += 1
                    item.update(region_evaluation(graph,route,estimated,truth,allowed,float(supervision['optimal_cost'][row])))
                elif self.model_representation in ('corridor','segments'):
                    self.calls['diagnostic_solver_calls'] += 1
                    item.update(corridor_evaluation(graph,route,estimated,truth,float(supervision['optimal_cost'][row])))
                else:
                    item['cost_cell_accuracy'] = float(np.isclose(modeled_costs, truth, rtol=1e-5, atol=1e-5).mean())
                    item['cost_mae'] = float(np.abs(modeled_costs-truth).mean())
                self.calls['diagnostic_solver_calls'] += 1
                uniform = shortest_path(np.ones_like(truth), source, target, allowed=allowed, no_corner_cut=land)
                item['same_path_as_uniform_cost'] = bool(np.array_equal(route, uniform))
            except (ValueError, KeyError, TypeError, OverflowError) as exc:
                item['error'] = str(exc)
                if hasattr(exc,'category'):item['failure_category'] = exc.category
                item.setdefault('failure_category', ('model_error' if self.model_representation=='algebra' else 'invalid_model') if item['status']=='model_failure' else 'solver_error')
            item['environment_seconds'] = time.perf_counter() - start
            results.append(item)
        return results


def summarize_outcomes(results):
    success = [x for x in results if x['status'] == 'success']
    if not results:
        raise ValueError('cannot summarize an empty evaluation')
    result = {'samples': len(results), 'successful_samples': len(success),
              'success_rate': len(success)/len(results),
              'task_loss': float(np.mean([x['loss'] for x in results])),
              'optimal_rate_all': sum(bool(x['optimal']) for x in results)/len(results)}
    # Failures contribute zero, with ALL cases in the denominator.
    # Do not silently assign zero to successful legacy logs lacking metrics.
    result['sr'] = result['success_rate']
    for field, name in (('spl','spl'), ('cost_efficiency','success_weighted_cost_efficiency')):
        if all(field in x for x in success):
            result[name] = sum(float(x[field]) for x in success)/len(results)
    if success and all('path_length' in x for x in success):
        result['path_length_successful'] = float(np.mean([x['path_length'] for x in success]))
    for status in ('model_failure', 'solver_failure', 'constraint_failure'):
        result[status+'_rate'] = sum(x['status'] == status for x in results)/len(results)
    for field in ('relative_regret', 'true_cost'):
        result[field+'_successful'] = float(np.mean([x[field] for x in success])) if success else None
    for field in ('retained_tokens', 'total_tokens', 'environment_seconds'):
        result[field] = float(np.mean([x[field] for x in results]))
    result['token_ratio'] = result['retained_tokens']/result['total_tokens']
    diagnostic_success = [x for x in success if 'cost_cell_accuracy' in x]
    if diagnostic_success:
        result['constant_matrix_rate_successful'] = float(np.mean([
            x['matrix_diagnostics']['constant_matrix'] for x in diagnostic_success]))
        for field in ('cost_cell_accuracy', 'cost_mae', 'same_path_as_uniform_cost'):
            result[field+'_successful'] = float(np.mean([x[field] for x in diagnostic_success]))
    region_success = [x for x in success if x.get('model_representation')=='regions']
    if region_success:
        for field in ('region_count','blocked_region_count','modeled_cell_count','modeled_arc_count',
                      'estimated_path_cost','path_cost_relative_error','region_oracle_cost',
                      'region_restriction_regret','region_cost_model_regret'):
            result[field+'_successful'] = float(np.mean([x[field] for x in region_success]))
    graph_success = [x for x in success if x.get('model_representation') in ('corridor','segments')]
    if graph_success:
        for field in ('estimated_path_cost','path_cost_relative_error','corridor_oracle_cost',
                      'corridor_restriction_regret','corridor_cost_model_regret',
                      'waypoint_count','segment_count','modeled_cell_count','modeled_arc_count',
                      'same_path_as_uniform_cost'):
            result[field+'_successful'] = float(np.mean([x[field] for x in graph_success]))
        result['single_segment_rate_successful'] = float(np.mean([x['segment_count']==1 for x in graph_success]))
        result['constant_segment_cost_rate_successful'] = float(np.mean([x['constant_segment_costs'] for x in graph_success]))
    result['failure_categories'] = {key:sum(x.get('failure_category','unknown')==key and x['status']!='success' for x in results)
        for key in sorted({x.get('failure_category','unknown') for x in results if x['status']!='success'})}
    for category in ('format_error','model_error','infeasible','invalid_solution','solver_limit'):
        result[category+'_rate'] = sum(x.get('failure_category')==category for x in results)/len(results)
    result['empty_connections_rate_all'] = sum(bool(x.get('empty_connections')) and x['status']!='success' for x in results)/len(results)
    generation = [x['generation'] for x in results if 'generation' in x]
    if generation:
        result['length_stop_rate'] = float(np.mean([x['finish_reason']=='length' for x in generation]))
        result['mean_generated_tokens'] = float(np.mean([x['generated_tokens'] for x in generation]))
        result['generation_finish_reasons'] = {reason:sum(x['finish_reason']==reason for x in generation)
            for reason in sorted({x['finish_reason'] for x in generation})}
    for field in ('input_tokens','nonvisual_input_tokens'):
        measured = [x[field] for x in generation if field in x]
        if measured:
            result['mean_'+field] = float(np.mean(measured))
    return result
