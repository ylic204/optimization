"""Frozen VLM -> corridor (or legacy matrix) model -> solver -> true-cost reward."""
import hashlib
import time
import numpy as np
import torch
from tilebased_core_v85 import validate_model, shortest_path, evaluate_path, matrix_diagnostics

from tilebased_corridor_v85 import validate_corridor, compile_corridor, solve_corridor, corridor_evaluation
from tilebased_prompts_v85 import modeling_user_prompt, modeling_system_prompt

ENVIRONMENT_CONTRACT = 'frozen_tilebased_llm_exact_mincost_v1'
CORRIDOR_ENVIRONMENT_CONTRACT = 'frozen_tilebased_llm_sg_corridor_v1'

def environment_contract(representation):
    if representation == 'matrix':return ENVIRONMENT_CONTRACT
    if representation == 'corridor':return CORRIDOR_ENVIRONMENT_CONTRACT
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
            system = modeling_system_prompt(self.prompt_variant,dataset)
            prompt = modeling_user_prompt(self.prompt_variant,public_inputs['task_text'][row], dataset, source, target,
                                        chosen, self.region_grid, public_inputs['legend'][row],self.model_representation)
            state = {k: (tuple(x[row:row+1] for x in v) if k == 'position_groups' else v[row:row+1])
                     for k, v in encoded['vision_state'].items()}
            item = {'sample_id': public_inputs['sample_id'][row], 'model_representation':self.model_representation, 'environment_contract':self.contract, 'status': 'model_failure',
                    'loss': self.failure_cost, 'selected_indices': chosen,
                    'retained_tokens': len(chosen), 'total_tokens': self.region_grid**2,
                    'relative_regret': None, 'true_cost': None, 'optimal': None, 'path_rc': None}
            try:
                item['modeling_prompt'] = prompt
                item['modeling_prompt_variant'] = self.prompt_variant
                item['system_prompt'] = system
                item['prompt_sha256'] = hashlib.sha256((system+'\n'+prompt).encode()).hexdigest()
                generation_kwargs = {'system_prompt':system} if system else {}
                self.calls['modeling_calls'] += 1
                if hasattr(self.backbone, 'generate_selected_report'):
                    generation = self.backbone.generate_selected_report(state, indices[row:row+1], prompt, self.max_new_tokens, **generation_kwargs)
                    answer = generation['text']
                    item['generation'] = {k:v for k,v in generation.items() if k != 'text'}
                else:
                    answer = self.backbone.generate_selected_json(state, indices[row:row+1], prompt, self.max_new_tokens, **generation_kwargs)
                item['model_answer'] = answer
                if self.model_representation == 'corridor':
                    modeled = validate_corridor(answer,dataset,source,target)
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
                    raise ValueError('generated model has no S-to-G path; no fallback routing is applied')
                # Privileged labels are accessed only AFTER generation and solving.
                measured = evaluate_path(route, supervision['true_costs'][row].cpu().numpy(), source,
                                         target, float(supervision['optimal_cost'][row]))
                item.update(measured, status='success')
                item['loss'] = min(measured['relative_regret'], self.failure_cost*.999)
                truth = supervision['true_costs'][row].cpu().numpy()
                if self.model_representation == 'corridor':
                    self.calls['diagnostic_solver_calls'] += 1
                    item.update(corridor_evaluation(graph,route,estimated,truth,float(supervision['optimal_cost'][row])))
                else:
                    item['cost_cell_accuracy'] = float(np.isclose(modeled_costs, truth, rtol=1e-5, atol=1e-5).mean())
                    item['cost_mae'] = float(np.abs(modeled_costs-truth).mean())
                self.calls['diagnostic_solver_calls'] += 1
                uniform = shortest_path(np.ones_like(truth), source, target)
                item['same_path_as_uniform_cost'] = bool(np.array_equal(route, uniform))
            except (ValueError, KeyError, TypeError, OverflowError) as exc:
                item['error'] = str(exc)
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
    for status in ('model_failure', 'solver_failure'):
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
    graph_success = [x for x in success if x.get('model_representation')=='corridor']
    if graph_success:
        for field in ('estimated_path_cost','path_cost_relative_error','corridor_oracle_cost',
                      'corridor_restriction_regret','corridor_cost_model_regret',
                      'waypoint_count','segment_count','modeled_cell_count','modeled_arc_count',
                      'same_path_as_uniform_cost'):
            result[field+'_successful'] = float(np.mean([x[field] for x in graph_success]))
    generation = [x['generation'] for x in results if 'generation' in x]
    if generation:
        result['length_stop_rate'] = float(np.mean([x['finish_reason']=='length' for x in generation]))
        result['mean_generated_tokens'] = float(np.mean([x['generated_tokens'] for x in generation]))
    for field in ('input_tokens','nonvisual_input_tokens'):
        measured = [x[field] for x in generation if field in x]
        if measured:
            result['mean_'+field] = float(np.mean(measured))
    return result
