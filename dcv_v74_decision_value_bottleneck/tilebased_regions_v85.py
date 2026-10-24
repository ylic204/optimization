"""Task-related areas -> model-only free-space graph. No GT arguments accepted."""
import numpy as np
from tilebased_core_v85 import DATASETS, parse_model_json, neighbors, diagonal_clear
from tilebased_segments_v85 import json_coordinate
from tilebased_corridor_v85 import _cost

REGION_SCHEMA = 'tilebased_land_polygons_v4'
LEGACY_REGION_SCHEMA = 'tilebased_land_regions_v3'
MAX_REGIONS = 64


def bounds(value, shape):
    if not isinstance(value,list) or len(value)!=4 or any(type(v) is not int for v in value):
        raise ValueError('bounds_rc must be four JSON integers [r0,c0,r1,c1]')
    r0,c0,r1,c1=value
    if not (0<=r0<=r1<shape[0] and 0<=c0<=c1<shape[1]):
        raise ValueError('invalid region bounds')
    return r0,c0,r1,c1


def _cross(a,b,c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def _on(a,b,p):
    return _cross(a,b,p)==0 and all(min(a[k],b[k])<=p[k]<=max(a[k],b[k]) for k in (0,1))


def _intersects(a,b,c,d):
    x,y,z,w=_cross(a,b,c),_cross(a,b,d),_cross(c,d,a),_cross(c,d,b)
    return (x*y<0 and z*w<0) or any((_on(a,b,c),_on(a,b,d),_on(c,d,a),_on(c,d,b)))


def geometry(value,shape):
    if not isinstance(value,dict):
        return {'bounds_rc':list(bounds(value,shape))}  # legacy blocked box
    if set(value)=={'bounds_rc'}:
        return {'bounds_rc':list(bounds(value['bounds_rc'],shape))}
    if set(value)!={'polygon_rc'}:
        raise ValueError('geometry requires polygon_rc or legacy bounds_rc')
    vertices=value['polygon_rc']
    if not isinstance(vertices,list) or not 3<=len(vertices)<=64:
        raise ValueError('polygon requires 3 to 64 vertices')
    pts=[]
    for v in vertices:
        if not isinstance(v,list) or len(v)!=2 or any(type(x) is not int for x in v):
            raise ValueError('polygon vertices must be integer grid-corner [row,col] pairs')
        if not all(0<=v[k]<=shape[k] for k in (0,1)):
            raise ValueError('polygon vertex outside grid')
        pts.append(tuple(v))
    if len(set(pts))!=len(pts):raise ValueError('duplicate polygon vertex; do not repeat first vertex')
    n=len(pts)
    if sum(pts[i][0]*pts[(i+1)%n][1]-pts[(i+1)%n][0]*pts[i][1] for i in range(n))==0:
        raise ValueError('zero-area polygon')
    for i in range(n):
        a,b,c=pts[i-1],pts[i],pts[(i+1)%n]
        if _cross(a,b,c)==0 and sum((a[k]-b[k])*(c[k]-b[k]) for k in (0,1))>0:
            raise ValueError('polygon backtracking edge')
        for j in range(i+1,n):
            if j==i+1 or (i==0 and j==n-1):continue
            if _intersects(pts[i],pts[(i+1)%n],pts[j],pts[(j+1)%n]):
                raise ValueError('self-intersecting polygon')
    return {'polygon_rc':[list(p) for p in pts]}


def geometry_mask(geom,shape):
    mask=np.zeros(shape,bool)
    if 'bounds_rc' in geom:
        r0,c0,r1,c1=geom['bounds_rc'];mask[r0:r1+1,c0:c1+1]=True
        return mask
    pts=geom['polygon_rc']
    for r,c in np.ndindex(shape):
        point=(r+.5,c+.5);inside=False
        for a,b in zip(pts,pts[1:]+pts[:1]):
            if _on(a,b,point):inside=True;break
            if (a[0]>point[0])!=(b[0]>point[0]):
                crossing=a[1]+(point[0]-a[0])*(b[1]-a[1])/(b[0]-a[0])
                if point[1]<crossing:inside=not inside
        mask[r,c]=inside
    return mask


def validate_regions(answer, dataset, source, target):
    obj=parse_model_json(answer)
    required={'schema','source_rc','target_rc','regions','blocked_regions'}
    if not isinstance(obj,dict) or set(obj)!=required or obj['schema'] not in (REGION_SCHEMA,LEGACY_REGION_SCHEMA):
        raise ValueError('expected five-key land-region model')
    shape=(DATASETS[dataset]['grid'],)*2
    if json_coordinate(obj['source_rc'],shape)!=tuple(source) or json_coordinate(obj['target_rc'],shape)!=tuple(target):
        raise ValueError('region endpoints mismatch')
    low,high=min(DATASETS[dataset]['cost_values']),max(DATASETS[dataset]['cost_values'])
    for key in ('regions','blocked_regions'):
        if not isinstance(obj[key],list) or len(obj[key])>MAX_REGIONS:
            raise ValueError(f'{key} must contain at most {MAX_REGIONS} regions')
    regions=[];seen=set()
    for region in obj['regions']:
        if not isinstance(region,dict) or 'cost' not in region:
            raise ValueError('regions require geometry and cost')
        geom=geometry({k:v for k,v in region.items() if k!='cost'},shape)
        signature=geometry_mask(geom,shape).tobytes()
        if signature in seen:raise ValueError('duplicate region coverage')
        seen.add(signature);regions.append((geom,_cost(region['cost'],low,high)))
    blocked=[geometry(b,shape) for b in obj['blocked_regions']]
    if len({geometry_mask(g,shape).tobytes() for g in blocked})!=len(blocked):
        raise ValueError('duplicate blocked coverage')
    return dict(regions=regions,blocked=blocked,source=tuple(source),target=tuple(target),shape=shape)


def compile_regions(model):
    # Dense arrays here are INTERNAL graph bookkeeping, not VLM output or labels.
    costs=np.zeros(model['shape'],dtype=float);covered=np.zeros(model['shape'],bool)
    for geom,cost in model['regions']:
        area=geometry_mask(geom,model['shape'])
        costs[area]=np.maximum(costs[area],cost);covered[area]=True
    blocked=np.zeros(model['shape'],bool)
    for geom in model['blocked']:
        blocked |= geometry_mask(geom,model['shape'])
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
        'Cover useful passages and plausible lower-cost detours with multiple simple polygons of visible usable width. '
        'The solver chooses each step anywhere inside their union. Do not draw a single waypoint chain. '
        'Follow coastlines and turns with concave polygons; split regions at terrain-cost boundaries. Multiple disconnected regions and multiple blocked polygons are allowed. Represent holes using blocked polygons. '
        'Do not cover unrelated terrain merely to fill the map; do not output a full cost matrix. '
        'Cost is an estimated per-cell traversal cost within a region. Preserve visible cost differences. '
        'Describe water intersecting or bordering proposed areas using blocked polygons. '
        'Blocked polygons override all traversable polygons. Uncovered cells are outside the proposed search space. '
        'Overlapping traversable polygons use the maximum declared cost, independent of ordering. '
        'Neighbors are connected automatically; diagonal steps require BOTH orthogonal side cells to be covered and unblocked. '
        'No hidden true terrain or GT will repair your model. Include S and G within connected modeled land areas.\n'
        f'Return exactly one JSON object with schema="{REGION_SCHEMA}", source_rc={list(map(int,source))}, '
        f'target_rc={list(map(int,target))}, regions=[{{"polygon_rc":[[r0,c0],[r1,c1],[r2,c2]],"cost":number}}], '
        'blocked_regions=[{"polygon_rc":[[r0,c0],[r1,c1],[r2,c2]]}]. '
        f'Polygon vertices are integer GRID CORNERS from 0 through {n}, row down, column right. '
        'Cell [r,c] has center [r+0.5,c+0.5]; centers inside or on a polygon are included. '
        'S/G remain cell indices. List 3 to 64 boundary vertices in order, clockwise or counterclockwise; '
        'do not repeat the first vertex, cross edges or backtrack. Concave shapes are allowed. '
        f'Costs are finite in [1,25]. At most {MAX_REGIONS} regions per list. '
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
