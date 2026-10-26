"""Visual facts + generated indexed algebra -> expanded MILP -> grid trajectory.

Only public shape/endpoints/visibility and generated evidence enter compilation.
Spatial operators add geometry, never flow/obstacle/degree constraints.
"""
import copy
import json
import numpy as np
from tilebased_algebra_v85 import (SCHEMA as ALGEBRA_SCHEMA, ModelError, keys,
                                  compile_model, parse_answer, number)
from tilebased_regions_v85 import geometry, geometry_mask

SCHEMA='tilebased_symbolic_v1'
UNKNOWN_COST=25.0


def error(message): raise ModelError('model_error',message)


def build_scene(evidence,shape,indices,region_grid):
    keys(evidence,{'regions'})
    regions=evidence['regions']
    if not isinstance(regions,list) or len(regions)>64: error('scene requires <=64 evidence regions')
    costs=np.full(shape,UNKNOWN_COST);blocked=np.zeros(shape,dtype=bool)
    covered=np.zeros(shape,dtype=bool)
    for region in regions:
        keys(region,{'geometry','cost','blocked'})
        if type(region['blocked']) is not bool: error('blocked must be boolean')
        cost=number(region['cost'])
        if not 1<=cost<=25: error('per-cell evidence cost must be in [1,25]')
        try: mask=geometry_mask(geometry(region['geometry'],shape),shape)
        except ValueError as exc: error('invalid evidence geometry: '+str(exc))
        costs[mask & ~covered]=cost
        costs[mask & covered]=np.maximum(costs[mask & covered],cost)
        blocked[mask] |= region['blocked'];covered |= mask
    visible=np.zeros(shape,dtype=bool);chosen=set(map(int,indices))
    if type(region_grid) is not int or region_grid<=0 or any(i<0 or i>=region_grid**2 for i in chosen):error('invalid visual region positions')
    for r,c in np.ndindex(shape):
        visible[r,c]=(r*region_grid//shape[0])*region_grid+c*region_grid//shape[1] in chosen
    return dict(cost=costs,blocked=blocked,covered=covered,visible=visible)


def grid_space(shape,connectivity):
    h,w=shape;arcs=[]
    for r,c in np.ndindex(shape):
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                if (dr==dc==0) or (connectivity==4 and abs(dr)+abs(dc)!=1):continue
                rr,cc=r+dr,c+dc
                if 0<=rr<h and 0<=cc<w:arcs.append((r*w+c,rr*w+cc))
    outgoing=[[] for _ in range(h*w)];incoming=[[] for _ in range(h*w)]
    for e,(a,b) in enumerate(arcs):outgoing[a].append(e);incoming[b].append(e)
    return arcs,outgoing,incoming


def compile_symbolic(answer,shape,source,target,indices,region_grid):
    raw=parse_answer(answer)
    keys(raw,{'schema','scene','model'})
    if raw['schema']!=SCHEMA:error('expected '+SCHEMA+'; enumerated candidate routes are no longer accepted')
    if len(shape)!=2 or any(type(v) is not int or not 1<=v<=32 for v in shape):error('unsupported public grid shape')
    h,w=shape
    for rc in (source,target):
        if len(rc)!=2 or any(int(v)!=v or not 0<=v<shape[k] for k,v in enumerate(rc)):error('invalid public S/G')
    scene=build_scene(raw['scene'],shape,indices,region_grid)
    m=raw['model'];keys(m,{'sets','parameters','variables','objective','constraints','output'})
    if not isinstance(m['sets'],dict):error('sets must be operator declarations')
    declarations=m['sets']
    # Generic space operators. Their graph does not inspect evidence or labels.
    cell_names=[n for n,spec in declarations.items() if spec=={'op':'grid_cells'}]
    arc_names=[n for n,spec in declarations.items() if isinstance(spec,dict) and spec.get('op')=='neighbor_pairs']
    if len(cell_names)!=1 or len(arc_names)!=1 or len(declarations)!=2:error('declare one grid_cells set and one neighbor_pairs set')
    cells,edges=cell_names[0],arc_names[0];spec=declarations[edges]
    keys(spec,{'op','of','connectivity'})
    if spec['of']!=cells or type(spec['connectivity']) is not int or spec['connectivity'] not in (4,8):error('invalid neighbor_pairs declaration')
    arcs,outgoing,incoming=grid_space(shape,spec['connectivity'])
    flat_cost=scene['cost'].ravel().tolist();flat_blocked=scene['blocked'].astype(int).ravel().tolist()
    corner=[]
    for a,b in arcs:
        r,c=divmod(a,w);rr,cc=divmod(b,w)
        corner.append(int(r!=rr and c!=cc and (scene['blocked'][r,cc] or scene['blocked'][rr,c])))
    public={'scene.cost':flat_cost,'scene.blocked':flat_blocked,
            'scene.unknown':(~scene['covered']).astype(int).ravel().tolist(),
            'scene.corner_blocked':corner,
            'task.source':int(source[0])*w+int(source[1]),'task.goal':int(target[0])*w+int(target[1]),
            'space.tail':[a for a,b in arcs],'space.head':[b for a,b in arcs],
            'space.outgoing':outgoing,'space.incoming':incoming,'space.step_length':[1]*len(arcs)}
    if not isinstance(m['parameters'],dict):error('parameters must bind public symbols')
    parameters={}
    for name,ref in m['parameters'].items():
        if isinstance(ref,str):
            if ref not in public:error('unknown parameter binding: '+ref)
            parameters[name]=public[ref]
        elif type(ref) in (int,float): parameters[name]=number(ref)
        else:error('parameters must be scalar constants or public bindings; no numeric arrays')
    keys(m['output'],{'variable','index'})
    if m['output']['index']!=edges:error('output index must name the neighbor_pairs set')
    expanded=dict(schema=ALGEBRA_SCHEMA,sets={cells:list(range(h*w)),edges:list(range(len(arcs)))},
                  parameters=parameters,variables=copy.deepcopy(m['variables']),objective=copy.deepcopy(m['objective']),
                  constraints=copy.deepcopy(m['constraints']),output=copy.deepcopy(m['output']))
    compiled=compile_model(json.dumps(expanded),expanded_grid=True)
    family=m['output']['variable']
    if family not in compiled['domains'] or compiled['domains'][family]!=list(range(len(arcs))):error('output variable must be indexed by generated adjacency set')
    compiled['symbolic_raw']=raw
    compiled['grid_arcs']=arcs
    compiled['grid_shape']=shape
    compiled['route_variable']=family
    evidence_report=dict(shape=list(shape),cost=scene['cost'].tolist(),blocked=scene['blocked'].tolist(),
        evidence_coverage=scene['covered'].tolist(),direct_visual_coverage=scene['visible'].tolist(),
        inferred_evidence=(scene['covered'] & ~scene['visible']).tolist(),
        unknown_policy={'cost':UNKNOWN_COST,'blocked':False,'description':'fixed pessimistic cost, unknown is not forbidden'},
        unknown_fraction=float((~scene['covered']).mean()))
    return compiled,evidence_report


def decode_symbolic(compiled,solution,source,target):
    w=compiled['grid_shape'][1];family=compiled['route_variable'];selected={}
    for (name,e),value in zip(compiled['labels'],solution['values']):
        if name!=family:continue
        if abs(value)>1e-6 and abs(value-1)>1e-6:raise ModelError('invalid_solution','route variable is not binary-valued')
        if value>.5:selected[e]=compiled['grid_arcs'][e]
    start=int(source[0])*w+int(source[1]);goal=int(target[0])*w+int(target[1])
    route=[start];seen={start}
    while route[-1]!=goal:
        choices=[e for e,(a,b) in selected.items() if a==route[-1]]
        if len(choices)!=1:raise ModelError('invalid_solution','solver assignment has missing/branching continuation')
        _,b=selected.pop(choices[0])
        if b in seen:raise ModelError('invalid_solution','solver assignment contains a cycle')
        route.append(b);seen.add(b)
    if selected:raise ModelError('invalid_solution','solver assignment contains unused selected moves')
    return np.array([divmod(v,w) for v in route],dtype=np.int64)


# Prompt example supplies knowledge, not compiler defaults. Model changes are executed.
MODEL_EXAMPLE={
 'sets':{'V':{'op':'grid_cells'},'A':{'op':'neighbor_pairs','of':'V','connectivity':8}},
 'parameters':{'c':'scene.cost','b':'scene.blocked','corner':'scene.corner_blocked',
               's':'task.source','g':'task.goal','tail':'space.tail','head':'space.head',
               'outgoing':'space.outgoing','incoming':'space.incoming'},
 'variables':{'x':{'index':'A','type':'binary','lb':0,'ub':1}},
 'objective':{'sense':'min','expression':'c[s] + sum(c[head[e]] * x[e] for e in A)'},
 'constraints':[
  {'name':'balance','forall':{'v':'V'},'lhs':'sum(x[e] for e in outgoing[v]) - sum(x[e] for e in incoming[v])','sense':'==','rhs':'(1 if v == s else 0) - (1 if v == g else 0)'},
  {'name':'out_degree','forall':{'v':'V'},'lhs':'sum(x[e] for e in outgoing[v])','sense':'<=','rhs':1},
  {'name':'in_degree','forall':{'v':'V'},'lhs':'sum(x[e] for e in incoming[v])','sense':'<=','rhs':1},
  {'name':'leave_land','forall':{'e':'A'},'lhs':'x[e]','sense':'<=','rhs':'1-b[tail[e]]'},
  {'name':'enter_land','forall':{'e':'A'},'lhs':'x[e]','sense':'<=','rhs':'1-b[head[e]]'},
  {'name':'clear_corner','forall':{'e':'A'},'lhs':'x[e]','sense':'<=','rhs':'1-corner[e]'}],
 'output':{'variable':'x','index':'A'}}


def symbolic_prompt(task,dataset,source,target,indices,region_grid,legend):
    from tilebased_core_v85 import DATASETS
    n=DATASETS[dataset]['grid'];side=DATASETS[dataset]['image_side']
    visible=('Full image is visible.' if len(indices)==region_grid**2 else
             f'Visible region positions on the {region_grid}x{region_grid} row-major grid: {list(map(int,indices))}.')
    return f'''{task}
TOP-DOWN PIXEL-ART NAVIGATION. Image {side}x{side}, terrain raster {n}x{n}.
S={list(map(int,source))}, G={list(map(int,target))}; cell coordinates [row,column], zero-based.
{visible} Public terrain legend: {legend}
Produce TWO things: compact visual environment evidence and an indexed optimization model.
Do NOT generate node IDs, edge IDs, tail/head arrays, cost matrices, waypoints,
connections or candidate routes. The compiler expands geometry; the solver searches the route.
1. scene.regions describes terrain evidence using geometry, per-cell cost, blocked.
Use a few task-relevant areas and relevant alternatives; do not list every cell.
Geometry is {{"bounds_rc":[r0,c0,r1,c1]}} with INCLUSIVE cell bounds, or
{{"polygon_rc":[[r,c],...]}} with 3..64 grid-corner vertices in [0,{n}].
Polygons may be concave; no self crossings or repeated closing vertex. Cell centers
inside/on the polygon are covered. Water has blocked=true; cost=25 is a placeholder
for its unused traversal cost. Land uses the public cost and blocked=false.
At overlaps blocked wins and costs take their maximum. At most 64 evidence regions.
Uncovered cells remain UNKNOWN, cost={UNKNOWN_COST}, not automatically blocked.
Unseen evidence is an inference, not a direct observation. Never copy the example's geometry.
2. model contains sets, parameters, variables, objective, constraints, output.
Sets use grid_cells and neighbor_pairs operators (4 or 8 neighbors), never lists.
Parameter bindings: scene.cost, scene.blocked, scene.unknown, scene.corner_blocked,
task.source, task.goal, space.tail, space.head, space.outgoing, space.incoming,
space.step_length. Numeric scalar constants are allowed; parameter arrays are NOT.
These are geometry and YOUR estimated evidence; no ground truth is available.
Grid adjacency includes blocked cells: your expressions must forbid their use.
Generate all necessary flow, degree and land/corner constraints explicitly.
The compiler adds NONE. Use x[e] indexed by the adjacency set; the output binding
identifies that variable. No route geometry is required. Expressions support
linear arithmetic, sum over a set or incoming/outgoing list, parameter indexing,
parameter comparisons and conditional expressions. No imports, attributes, eval,
nonlinear decision products, or decision-dependent conditions. Use forall for
constraint families rather than expanding them. Keep JSON compact and complete.
The following model template is a syntax/formulation example, not automatically installed:
{json.dumps(MODEL_EXAMPLE,separators=(',',':'))}
Return ONLY {{"schema":"{SCHEMA}","scene":{{"regions":[{{"geometry":{{"bounds_rc":[0,0,1,1]}},"cost":1.5,"blocked":false}}]}},"model":{{...}}}}.
Replace illustrative evidence with this image's terrain. Include your complete
model expressions. Do not output a final path. Close the JSON.
'''
