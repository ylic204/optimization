"""Public task taxonomy and authored modeling guidance; never solver defaults.
Inspired by MM-Agent's analysis -> method retrieval -> formulation workflow.
No upstream code copied. Only the current benchmark family is supported.
"""
import hashlib
import json

TEMPLATE_ID='path_planning/minimum_terrain_cost/land_patch_ports_v3'


def retrieve_template(task):
    # This benchmark already specifies its task family. Do not spend another VLM
    # call reclassifying identical tasks or claim a general semantic classifier.
    text=task.lower()
    if 'minimum total terrain-cost' not in text or 'land path' not in text:
        raise ValueError('unsupported template task: expected benchmark minimum terrain-cost LAND path')
    return dict(problem_class='path_planning',subclass='minimum_terrain_cost',
                template_id=TEMPLATE_ID,selection='public_benchmark_task_rule',
                template_sha256=hashlib.sha256((GUIDANCE+json.dumps(EXAMPLE,sort_keys=True)).encode()).hexdigest())


GUIDANCE='''RETRIEVED MODELING TEMPLATE: minimum terrain-cost directed S-to-G path.
This is modeling guidance, not a model installed by the compiler. Return every variable,
objective and constraint in your JSON. Adapt the template to the actual visible scene.
1. Build task-relevant connections using the PUBLIC PATCH NODE REGISTRY below.
   N contains fixed patch-port IDs, not arbitrary labels or visual sequence indices.
   Each patch has separate cell ports to preserve mixed terrain and connectivity.
   Only reference useful ports; do not enumerate all patches or all cells.
   E contains local contiguous connection IDs. Prefer compact arrays; tail/head/cost/
   allowed each have exactly len(E) entries. Preserve useful alternative routes.
2. Supply numeric parameter arrays tail[e], head[e], cost[e], allowed[e], and scalar s,t,start_cost.
   cost[e] is a POSITIVE estimated total cost of all entered cells along that connection,
   excluding its departure and including its arrival. It is a PARAMETER, not a decision variable.
   Estimate the cost from the image and public legend; do not leave terrain_type/cost undefined.
3. A standard formulation uses binary x[e] in [0,1], minimizing
   start_cost + sum(cost[e]*x[e] for e in E).
4. At EACH node n, enforce outgoing minus incoming flow equal to 1 at s, -1 at t, 0 otherwise.
   When s=t the balance is zero. Outgoing/incoming sums filter by tail/head, not by arc ID.
   Degree bounds <=1 prevent branching. Positive arc costs discourage cycles.
5. For every e, x[e] <= allowed[e]. Use allowed=0 for visually forbidden connections,
   including diagonal water-corner cuts. Do not turn forbidden water into a large finite cost.
6. Provide output geometry for EVERY e, using the SAME index and correct endpoint coordinates.
   Polyline direction must agree with tail/head. Use bends where needed. No empty output for a
   nonempty arc variable. Do not output placeholder geometry or copy the example's coordinates.
7. Before returning, check that every referenced symbol exists, arrays cover all indices,
   all constraints are scalar linear expressions, S/G match the task, and JSON contains
   literal arrays, not Python list comprehensions, tuples, comments or missing keys.

The following COMPLETE TINY EXAMPLE uses a 4x4 terrain grid and 2x2 patch grid.
Each patch has 4 ports. Nodes 0,1,4 are cells [0,0],[0,1],[0,2].
It demonstrates executable syntax, NOT the current scene.
Replace its nodes, endpoints, costs, allowed values and geometry using the current image.
'''

EXAMPLE={
 'schema':'tilebased_algebra_v1','sets':{'N':[0,1,4],'E':[0,1,2]},
 'parameters':{'tail':[0,1,0],'head':[1,4,4],'cost':[1,1,4],
               'allowed':[1,1,1],'s':0,'t':4,'start_cost':1},
 'variables':{'x':{'index':'E','type':'binary','lb':0,'ub':1}},
 'objective':{'sense':'min','expression':'start_cost + sum(cost[e]*x[e] for e in E)'},
 'constraints':[
  {'name':'balance','forall':{'n':'N'},
   'lhs':'sum(x[e] for e in E if tail[e] == n) - sum(x[e] for e in E if head[e] == n)',
   'sense':'==','rhs':'(1 if n == s else 0) - (1 if n == t else 0)'},
  {'name':'out_degree','forall':{'n':'N'},'lhs':'sum(x[e] for e in E if tail[e] == n)','sense':'<=','rhs':1},
  {'name':'in_degree','forall':{'n':'N'},'lhs':'sum(x[e] for e in E if head[e] == n)','sense':'<=','rhs':1},
  {'name':'admissible','forall':{'e':'E'},'lhs':'x[e]','sense':'<=','rhs':'allowed[e]'}],
 'output':{'variable':'x','arcs':[
  {'index':0,'polyline_rc':[[0,0],[0,1]]},
  {'index':1,'polyline_rc':[[0,1],[0,2]]},
  {'index':2,'polyline_rc':[[0,0],[1,1],[0,2]]}]}}


def template_prompt(task):
    record=retrieve_template(task)
    return json.dumps(record)+'\n'+GUIDANCE+json.dumps(EXAMPLE,separators=(',',':'))+'\nEND EXAMPLE. Now instantiate the CURRENT task.\n'
