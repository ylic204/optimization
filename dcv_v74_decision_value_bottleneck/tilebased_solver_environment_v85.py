"""Frozen VLM -> validated node-cost model -> Dijkstra -> true-cost reward."""
import time
import numpy as np
import torch
from tilebased_core_v85 import build_model_prompt, validate_model, shortest_path, evaluate_path

ENVIRONMENT_CONTRACT = 'frozen_tilebased_llm_exact_mincost_v1'


class FrozenTilebasedSolverEnvironment:
    def __init__(self, backbone, region_grid, max_new_tokens=4096, failure_cost=10.):
        if failure_cost <= 0 or not np.isfinite(failure_cost) or max_new_tokens <= 0:
            raise ValueError('positive failure cost and decode budget required')
        self.backbone, self.region_grid = backbone, region_grid
        self.max_new_tokens, self.failure_cost = max_new_tokens, failure_cost

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
            prompt = build_model_prompt(public_inputs['task_text'][row], dataset, source, target,
                                        chosen, self.region_grid, public_inputs['legend'][row])
            state = {k: (tuple(x[row:row+1] for x in v) if k == 'position_groups' else v[row:row+1])
                     for k, v in encoded['vision_state'].items()}
            item = {'sample_id': public_inputs['sample_id'][row], 'status': 'model_failure',
                    'loss': self.failure_cost, 'selected_indices': chosen,
                    'retained_tokens': len(chosen), 'total_tokens': self.region_grid**2,
                    'relative_regret': None, 'true_cost': None, 'optimal': None, 'path_rc': None}
            try:
                answer = self.backbone.generate_selected_json(state, indices[row:row+1], prompt, self.max_new_tokens)
                item['model_answer'] = answer
                modeled_costs = validate_model(answer, dataset, source, target)
                item['status'] = 'solver_failure'
                route = shortest_path(modeled_costs, source, target)
                if route is None:
                    raise ValueError('solver returned no path')
                # Privileged labels are accessed only AFTER generation and solving.
                measured = evaluate_path(route, supervision['true_costs'][row].cpu().numpy(), source,
                                         target, float(supervision['optimal_cost'][row]))
                item.update(measured, status='success')
                item['loss'] = min(measured['relative_regret'], self.failure_cost*.999)
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
    return result
