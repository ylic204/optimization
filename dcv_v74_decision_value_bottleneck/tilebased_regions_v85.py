"""Task-related areas -> model-only free-space graph. No GT arguments accepted."""
import numpy as np
from tilebased_core_v85 import DATASETS, parse_model_json, neighbors, diagonal_clear
from tilebased_segments_v85 import json_coordinate
from tilebased_corridor_v85 import _cost

REGION_SCHEMA = 'tilebased_land_regions_v3'
MAX_REGIONS = 64


def bounds(value, shape):
    if not isinstance(value,list) or len(value)!=4 or any(type(v) is not int for v in value):
        raise ValueError('bounds_rc must be four JSON integers [r0,c0,r1,c1]')
    r0,c0,r1,c1=value
    if not (0<=r0<=r1<shape[0] and 0<=c0<=c1<shape[1]):
        raise ValueError('invalid region bounds')
    return r0,c0,r1,c1


def validate_regions(answer, dataset, source, target):
    obj=parse_model_json(answer)
    required={'schema','source_rc','target_rc','regions','blocked_regions'}
    if not isinstance(obj,dict) or set(obj)!=required or obj['schema']!=REGION_SCHEMA:
        raise ValueError('expected five-key land-region model')
    shape=(DATASETS[dataset]['grid'],)*2
    if json_coordinate(obj['source_rc'],shape)!=tuple(source) or json_coordinate(obj['target_rc'],shape)!=tuple(target):
        raise ValueError('region endpoints mismatch')
    low,high=min(DATASETS[dataset]['cost_values']),max(DATASETS[dataset]['cost_values'])
    for key in ('regions','blocked_regions'):
        if not isinstance(obj[key],list) or len(obj[key])>MAX_REGIONS:
            raise ValueError(f'{key} must contain at most {MAX_REGIONS} rectangles')
    regions=[];seen=set()
    for region in obj['regions']:
        if not isinstance(region,dict) or set(region)!={'bounds_rc','cost'}:
            raise ValueError('regions require bounds_rc and cost')
        box=bounds(region['bounds_rc'],shape)
        if box in seen:raise ValueError('duplicate region bounds')
        seen.add(box);regions.append((box,_cost(region['cost'],low,high)))
    blocked=[bounds(b,shape) for b in obj['blocked_regions']]
    if len(set(blocked))!=len(blocked):raise ValueError('duplicate blocked bounds')
    return dict(regions=regions,blocked=blocked,source=tuple(source),target=tuple(target),shape=shape)


def compile_regions(model):
    # Dense arrays here are INTERNAL graph bookkeeping, not VLM output or labels.
    costs=np.zeros(model['shape'],dtype=float);covered=np.zeros(model['shape'],bool)
    for (r0,c0,r1,c1),cost in model['regions']:
        area=np.s_[r0:r1+1,c0:c1+1]
        costs[area]=np.maximum(costs[area],cost);covered[area]=True
    blocked=np.zeros(model['shape'],bool)
    for r0,c0,r1,c1 in model['blocked']:
        blocked[r0:r1+1,c0:c1+1]=True
    allowed=covered & ~blocked
    arcs={}
    for a in (tuple(map(int,p)) for p in np.argwhere(allowed)):
        for b in neighbors(a,model['shape']):
            if allowed[b] and diagonal_clear(a,b,allowed):arcs[a,b]=float(costs[b])
    # Do not insert missing S/G or open blocked endpoints.
    endpoints=bool(allowed[model['source']] and allowed[model['target']])
    return dict(arcs=arcs,source=model['source'],target=model['target'],shape=model['shape'],
                source_cost=float(costs[model['source']]) if endpoints else 1.,
                endpoints_covered=endpoints,allowed=allowed,costs=costs,
                region_count=len(model['regions']),blocked_region_count=len(model['blocked']),
                modeled_cell_count=int(allowed.sum()),modeled_arc_count=len(arcs))


def regions_user_prompt(task,dataset,source,target,indices,region_grid,legend):
    n=DATASETS[dataset]['grid'];side=DATASETS[dataset]['image_side']
    visible=('All visual regions are retained.' if sorted(map(int,indices))==list(range(region_grid**2)) else
             f'Retained visual regions on the original {region_grid}x{region_grid} row-major grid: {list(map(int,indices))}. '
             'Missing appearance is unknown, not known free space.')
    return (f'{task}\nImage: {side}x{side} pixels; terrain grid: {n}x{n}; each terrain cell: {side//n} pixels. '
        f'{visible}\nPublic legend: {legend}\n'
        'Model task-relevant LAND AREAS and their cost differences for a solver. '
        'Cover useful passages and plausible lower-cost detours with rectangles of visible usable width. '
        'The solver chooses each step anywhere inside their union. Do not draw a single waypoint chain. '
        'Split rectangles along coastlines, turns and terrain-cost boundaries. Use several rectangles for irregular shapes. '
        'Do not cover unrelated terrain merely to fill the map; do not output a full cost matrix. '
        'Cost is an estimated per-cell traversal cost within a region. Preserve visible cost differences. '
        'Describe water intersecting or bordering proposed areas using blocked rectangles. '
        'Blocked rectangles override all traversable rectangles. Uncovered cells are outside the proposed search space. '
        'Overlapping traversable rectangles use the maximum declared cost, independent of ordering. '
        'Neighbors are connected automatically; diagonal steps require BOTH orthogonal side cells to be covered and unblocked. '
        'No hidden true terrain or GT will repair your model. Include S and G within connected modeled land areas.\n'
        f'Return exactly one JSON object with schema="{REGION_SCHEMA}", source_rc={list(map(int,source))}, '
        f'target_rc={list(map(int,target))}, regions=[{{"bounds_rc":[r0,c0,r1,c1],"cost":number}}], '
        'blocked_regions=[[r0,c0,r1,c1]]. Bounds are inclusive integers, row down, column right, '
        f'0 through {n-1}. Costs are finite in [1,25]. At most {MAX_REGIONS} rectangles per list. '
        'An empty list is allowed but missing S/G coverage will fail. Close the JSON after the lists. '
        'No explanation, node IDs, edges, final path or extra keys.')


class RegionOutputGuard:
    def __call__(self,text):
        try:
            if isinstance(parse_model_json(text),dict):return 'json_complete'
        except (ValueError,TypeError):pass
        return None


def region_evaluation(graph,route,predicted_cost,true_costs,true_allowed,optimal_cost):
    # Evaluation ONLY, after the actual model-only route has been selected.
    from tilebased_corridor_v85 import solve_corridor
    from tilebased_core_v85 import path_cost
    oracle=dict(graph,source_cost=float(true_costs[graph['source']]),arcs={
        (a,b):float(true_costs[b]) for a,b in graph['arcs']
        if true_allowed[a] and true_allowed[b] and diagonal_clear(a,b,true_allowed)})
    oracle_route,oracle_cost=solve_corridor(oracle)
    if oracle_route is None:raise ValueError('valid predicted route absent from diagnostic graph')
    actual=path_cost(true_costs,route);scale=max(float(optimal_cost),1e-8)
    return dict(estimated_path_cost=predicted_cost,
                path_cost_relative_error=abs(predicted_cost-actual)/max(actual,1e-8),
                region_oracle_cost=oracle_cost,
                region_restriction_regret=max(0.,(oracle_cost-optimal_cost)/scale),
                region_cost_model_regret=max(0.,(actual-oracle_cost)/scale))
