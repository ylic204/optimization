"""Public-only S/G corridor graph, deterministic geometry, and sparse Dijkstra.

No scene costs, ground-truth route or image-derived oracle topology is accepted
by validation, compilation, or solve_corridor. The generated graph restricts the
search space; omitted edges are unmodeled, not physically blocked terrain.
"""
import heapq
import math
from collections import defaultdict
import numpy as np
from tilebased_core_v85 import DATASETS, coordinate, parse_model_json, validate_path, path_cost

CORRIDOR_SCHEMA = 'tilebased_sg_corridor_v1'
MAX_NODES = 32
MAX_EDGES = 64


def _cost(value, low, high):
    if type(value) not in (int,float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'cost estimates must be finite numbers in [{low}, {high}]')
    return float(value)


def raster_segment(a, b):
    """Nearest-integer interpolation along the major axis; reversible tie rule.

    Connect cell centers with max(|dr|,|dc|) eight-neighbor steps. Canonical
    endpoint ordering makes reversing a segment reverse the same cell sequence.
    This uses geometry only; it never avoids terrain using hidden labels.
    """
    a,b = tuple(a),tuple(b)
    reverse = a > b
    lo,hi = (b,a) if reverse else (a,b)
    steps = max(abs(hi[0]-lo[0]),abs(hi[1]-lo[1]))
    if not steps:
        return [a]
    result=[]
    for i in range(steps+1):
        # Round half upward, with integer arithmetic on nonnegative coordinates.
        result.append(tuple((2*((steps-i)*lo[d]+i*hi[d])+steps)//(2*steps) for d in (0,1)))
    return result[::-1] if reverse else result


def validate_corridor(answer, dataset, source, target):
    obj = parse_model_json(answer)
    required = {'schema','source_rc','target_rc','source_cost','nodes_rc','edges'}
    if not isinstance(obj,dict) or set(obj) != required or obj['schema'] != CORRIDOR_SCHEMA:
        raise ValueError('expected the S/G corridor schema and exactly its six keys')
    n = DATASETS[dataset]['grid'];shape = (n,n)
    source,target = coordinate(source,shape),coordinate(target,shape)
    if coordinate(obj['source_rc'],shape)!=source or coordinate(obj['target_rc'],shape)!=target:
        raise ValueError('corridor endpoints mismatch')
    raw = obj['nodes_rc']
    if not isinstance(raw,list) or not 1 <= len(raw) <= MAX_NODES:
        raise ValueError(f'nodes_rc must contain 1..{MAX_NODES} coordinates')
    nodes = [coordinate(point,shape) for point in raw]
    if len(set(nodes)) != len(nodes):
        raise ValueError('duplicate waypoint coordinates')
    if nodes[0]!=source or (source!=target and (len(nodes)<2 or nodes[1]!=target)):
        raise ValueError('node 0 must be S and node 1 must be G (one node if S=G)')
    if source==target and len(nodes)!=1:
        raise ValueError('S=G requires one node and no edges')
    low,high = min(DATASETS[dataset]['cost_values']),max(DATASETS[dataset]['cost_values'])
    source_cost = _cost(obj['source_cost'],low,high)
    raw_edges = obj['edges']
    if not isinstance(raw_edges,list) or len(raw_edges)>MAX_EDGES:
        raise ValueError(f'edges must be a list of at most {MAX_EDGES} directed segments')
    edges=[];seen=set()
    for edge in raw_edges:
        if not isinstance(edge,list) or len(edge)!=3:
            raise ValueError('each edge must be [from_node, to_node, mean_arrival_cost]')
        u,v,cost = edge
        if type(u) is not int or type(v) is not int or not (0<=u<len(nodes) and 0<=v<len(nodes)) or u==v:
            raise ValueError('edge references invalid waypoint IDs')
        if (u,v) in seen:
            raise ValueError('duplicate directed segment')
        seen.add((u,v));edges.append((u,v,_cost(cost,low,high)))
    return dict(nodes=nodes,edges=edges,source=source,target=target,source_cost=source_cost,shape=shape)


def compile_corridor(model):
    estimates=defaultdict(list)
    for u,v,cost in model['edges']:
        cells=raster_segment(model['nodes'][u],model['nodes'][v])
        for a,b in zip(cells,cells[1:]):
            estimates[a,b].append(cost)
    # Crossings at the same raster cell connect. Duplicate directed unit arcs
    # average their estimates, independent of segment order. No free adjacency.
    arcs={key:math.fsum(sorted(values))/len(values) for key,values in estimates.items()}
    cells=set(model['nodes'])
    for a,b in arcs:cells.update((a,b))
    return dict(arcs=arcs,source=model['source'],target=model['target'],
                source_cost=model['source_cost'],shape=model['shape'],
                waypoint_count=len(model['nodes']),segment_count=len(model['edges']),
                modeled_cell_count=len(cells),modeled_arc_count=len(arcs))


def solve_corridor(graph):
    source,target=graph['source'],graph['target']
    adjacency=defaultdict(list)
    for (a,b),cost in graph['arcs'].items():adjacency[a].append((b,cost))
    distance={source:graph['source_cost']};parent={}
    heap=[(graph['source_cost'],source)]
    while heap:
        value,node=heapq.heappop(heap)
        if value!=distance[node]:continue
        if node==target:
            route=[node]
            while node!=source:
                node=parent[node];route.append(node)
            route=np.asarray(route[::-1],dtype=np.int64)
            validate_path(route,source,target,graph['shape'])
            return route,float(value)
        for nxt,cost in sorted(adjacency[node]):
            proposed=value+cost
            if proposed<distance.get(nxt,float('inf')):
                distance[nxt]=proposed;parent[nxt]=node;heapq.heappush(heap,(proposed,nxt))
    return None,None


def corridor_evaluation(graph, route, predicted_cost, true_costs, optimal_cost):
    """AFTER solving only: labels diagnose topology restriction vs cost error.

    The oracle's path is NEVER substituted for the generated model's solution.
    Both regret components are normalized by the full-map reference optimum.
    """
    oracle=dict(graph,source_cost=float(true_costs[graph['source']]),
                arcs={(a,b):float(true_costs[b]) for a,b in graph['arcs']})
    oracle_route,oracle_cost=solve_corridor(oracle)
    if oracle_route is None:raise ValueError('cannot score a disconnected corridor graph')
    actual=path_cost(true_costs,route);scale=max(float(optimal_cost),1e-8)
    return dict(estimated_path_cost=predicted_cost,
                path_cost_relative_error=abs(predicted_cost-actual)/max(actual,1e-8),
                corridor_oracle_cost=oracle_cost,
                corridor_restriction_regret=max(0.,(oracle_cost-optimal_cost)/scale),
                corridor_cost_model_regret=max(0.,(actual-oracle_cost)/scale),
                **{key:graph[key] for key in ('waypoint_count','segment_count','modeled_cell_count','modeled_arc_count')})
