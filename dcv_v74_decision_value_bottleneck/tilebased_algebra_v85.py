"""Generated LP/MILP expressions -> HiGHS. No navigation constraints added.

The compiler accepts only public generated JSON. Geometry is used AFTER solving
for output decoding; neither geometry nor true labels create model constraints.
"""
import ast
import json
import math
import re
from dataclasses import dataclass
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from tilebased_corridor_v85 import raster_segment

SCHEMA = 'tilebased_algebra_v1'
CONTRACT = 'frozen_tilebased_generated_milp_v1'


class ModelError(ValueError):
    def __init__(self, category, message):
        self.category = category
        super().__init__(message)


def fail(message):
    raise ModelError('model_error', message)


def number(value):
    if type(value) not in (int,float) or not math.isfinite(value) or abs(value)>1e9:
        fail('expected finite numeric value with magnitude <= 1e9')
    return float(value)


@dataclass
class Linear:
    constant: float
    coefficients: dict

    @staticmethod
    def cast(x):
        return x if isinstance(x,Linear) else Linear(number(x),{})

    def add(self,other,sign=1):
        other=self.cast(other); co=self.coefficients.copy()
        for k,v in other.coefficients.items():co[k]=co.get(k,0)+sign*v
        return Linear(number(self.constant+sign*other.constant),{k:number(v) for k,v in co.items() if v!=0})

    def scale(self,k):
        k=number(k)
        return Linear(number(self.constant*k),{i:number(v*k) for i,v in self.coefficients.items() if v*k!=0})


def scalar(x):
    if isinstance(x,Linear):
        if x.coefficients:fail('decision-dependent condition, index or nonlinear expression')
        return x.constant
    return number(x)


class Expressions:
    """Small AST interpreter, never Python eval/exec. Bounded symbolic expansion."""
    def __init__(self,context):
        self.context=context
        self.remaining=200000

    def parse(self,text,local=None):
        if type(text) in (int,float):return number(text)
        if not isinstance(text,str) or len(text)>8192:fail('expression must be a string <=8192 characters')
        try:tree=ast.parse(text,mode='eval')
        except (SyntaxError,RecursionError) as exc:fail(f'invalid expression: {exc}')
        if sum(1 for _ in ast.walk(tree))>1024:fail('expression too complex')
        return self.visit(tree.body,dict(self.context,**(local or {})),0)

    def visit(self,node,ctx,depth):
        self.remaining-=1
        if self.remaining<0 or depth>40:fail('expression expansion limit')
        sub=lambda n:self.visit(n,ctx,depth+1)
        if isinstance(node,ast.Constant):
            return node.value if isinstance(node.value,str) else number(node.value)
        if isinstance(node,ast.Name):
            if node.id not in ctx:fail(f'undefined symbol: {node.id}')
            return ctx[node.id]
        if isinstance(node,ast.Subscript):
            obj=sub(node.value);raw=sub(node.slice)
            if isinstance(raw,str):idx=raw
            else:
                idx=scalar(raw)
                if idx!=int(idx):fail('index must be integer')
                idx=int(idx)
            if isinstance(obj,list):
                if not isinstance(idx,int) or not 0<=idx<len(obj):fail('parameter index out of range')
            elif isinstance(obj,dict):
                if idx not in obj:fail(f'undefined indexed entry: {idx}')
            else:fail('only parameter arrays/maps and indexed variables can be indexed')
            return obj[idx]
        if isinstance(node,ast.UnaryOp):
            v=Linear.cast(sub(node.operand))
            if isinstance(node.op,ast.USub):return v.scale(-1)
            if isinstance(node.op,ast.UAdd):return v
        if isinstance(node,ast.BinOp):
            a,b=Linear.cast(sub(node.left)),Linear.cast(sub(node.right))
            if isinstance(node.op,ast.Add):return a.add(b)
            if isinstance(node.op,ast.Sub):return a.add(b,-1)
            if isinstance(node.op,ast.Mult):
                if a.coefficients and b.coefficients:fail('nonlinear multiplication is unsupported')
                return a.scale(b.constant) if not b.coefficients else b.scale(a.constant)
            if isinstance(node.op,ast.Div):
                k=scalar(b)
                if k==0:fail('division by zero')
                return a.scale(1/k)
        if isinstance(node,ast.Compare):
            a=scalar(sub(node.left));result=True
            for op,rhs in zip(node.ops,node.comparators):
                b=scalar(sub(rhs))
                if isinstance(op,ast.Eq):ok=a==b
                elif isinstance(op,ast.NotEq):ok=a!=b
                elif isinstance(op,ast.Lt):ok=a<b
                elif isinstance(op,ast.LtE):ok=a<=b
                elif isinstance(op,ast.Gt):ok=a>b
                elif isinstance(op,ast.GtE):ok=a>=b
                else:fail('unsupported comparison')
                result=result and ok;a=b
            return int(result)
        if isinstance(node,ast.BoolOp) and isinstance(node.op,(ast.And,ast.Or)):
            values=[bool(scalar(sub(v))) for v in node.values]
            return int(all(values) if isinstance(node.op,ast.And) else any(values))
        if isinstance(node,ast.IfExp):
            return sub(node.body if scalar(sub(node.test)) else node.orelse)
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='sum' and not node.keywords and len(node.args)==1:
            gen=node.args[0]
            if not isinstance(gen,ast.GeneratorExp) or len(gen.generators)!=1:fail('sum requires one generator: sum(expr for i in SET if condition)')
            g=gen.generators[0]
            if g.is_async or not isinstance(g.target,ast.Name):fail('invalid sum index')
            if g.target.id in ctx:fail('sum index shadows existing symbol')
            seq=sub(g.iter)
            if not isinstance(seq,list) or len(seq)>1024:fail('sum domain must be a finite set')
            total=Linear(0,{})
            for value in seq:
                inner=dict(ctx,**{g.target.id:value})
                if all(scalar(self.visit(test,inner,depth+1)) for test in g.ifs):
                    total=total.add(self.visit(gen.elt,inner,depth+1))
            return total
        fail(f'unsupported expression syntax: {type(node).__name__}')


def keys(obj,required):
    if not isinstance(obj,dict) or set(obj)!=set(required):fail(f'expected fields {sorted(required)}')


def name(s):
    if not isinstance(s,str) or not re.fullmatch('[A-Za-z][A-Za-z0-9_]{0,31}',s) or s=='sum':fail('invalid/reserved symbol name')


def parse_answer(answer):
    def unique(pairs):
        d={}
        for k,v in pairs:
            if k in d:raise ValueError(f'duplicate JSON key: {k}')
            d[k]=v
        return d
    try:
        if not isinstance(answer,str) or len(answer)>300000:raise ValueError('output must be bounded JSON text')
        return json.loads(answer,object_pairs_hook=unique,parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite JSON number')))
    except (ValueError,TypeError,RecursionError) as exc:
        raise ModelError('format_error',str(exc)) from exc


def compile_model(answer):
    obj=parse_answer(answer)
    keys(obj,{'schema','sets','parameters','variables','objective','constraints','output'})
    if obj['schema']!=SCHEMA:fail('unsupported algebra schema')
    ctx={}
    for section in ('sets','parameters','variables'):
        if not isinstance(obj[section],dict) or len(obj[section])>64:fail(f'invalid {section}')
        for k in obj[section]:
            name(k)
            if k in ctx:fail('duplicate symbol')
            ctx[k]=None
    for k,values in obj['sets'].items():
        if not isinstance(values,list) or len(values)>1024 or any(type(v) is not int or abs(v)>100000 for v in values) or len(set(values))!=len(values):fail('sets require unique bounded integers')
        ctx[k]=values
    def parameter(v,depth=0,path='parameters'):
        if depth>4:fail('parameter nesting limit')
        if isinstance(v,list):
            if len(v)>1024:fail('parameter size limit')
            return [parameter(x,depth+1,f'{path}[{i}]') for i,x in enumerate(v)]
        if isinstance(v,dict):
            if len(v)>1024:fail(f'{path}: parameter size limit')
            return {k:parameter(x,depth+1,f'{path}.{k}') for k,x in v.items()}
        try:return number(v)
        except ModelError as exc:fail(f'{path}: {exc}')
    for k,v in obj['parameters'].items():ctx[k]=parameter(v,path=f'parameters.{k}')
    exp=Expressions(ctx);labels=[];lower=[];upper=[];integrality=[];domains={}
    for k,v in obj['variables'].items():
        keys(v,{'index','type','lb','ub'})
        if v['type'] not in ('binary','integer','continuous'):fail('unknown variable type')
        indices=[None] if v['index'] is None else obj['sets'].get(v['index']) if isinstance(v['index'],str) else None
        if indices is None:fail('unknown variable index set')
        lo=scalar(exp.parse(v['lb']));hi=scalar(exp.parse(v['ub']))
        if lo>hi:fail('inverted variable bounds')
        if v['type']=='binary' and (lo<0 or hi>1):fail('binary bounds must lie in [0,1]')
        family={}
        for i in indices:
            if len(labels)>=1024:fail('variable expansion limit')
            family[i]=Linear(0,{len(labels):1});labels.append([k,i]);lower.append(lo);upper.append(hi);integrality.append(int(v['type']!='continuous'))
        ctx[k]=family[None] if v['index'] is None else family
        domains[k]=indices
    if not labels:fail('no decision variables declared')
    keys(obj['objective'],{'sense','expression'})
    if obj['objective']['sense'] not in ('min','max'):fail('objective sense must be min or max')
    objective=Linear.cast(exp.parse(obj['objective']['expression']))
    if not isinstance(obj['constraints'],list) or len(obj['constraints'])>256:fail('invalid constraint list')
    rows=[];cl=[];cu=[]
    for rule in obj['constraints']:
        keys(rule,{'name','forall','lhs','sense','rhs'});name(rule['name'])
        if rule['sense'] not in ('==','<=','>='):fail('invalid constraint sense')
        if not isinstance(rule['forall'],dict) or len(rule['forall'])>1:fail('forall accepts zero or one index; nest sums for expressions')
        instances=[{}]
        for index,setname in rule['forall'].items():
            name(index)
            if index in ctx or not isinstance(setname,str) or setname not in obj['sets']:fail('invalid forall index/domain')
            instances=[{index:i} for i in obj['sets'][setname]]
        for local in instances:
            if len(rows)>=4096:fail('constraint expansion limit')
            row=Linear.cast(exp.parse(rule['lhs'],local)).add(exp.parse(rule['rhs'],local),-1)
            rows.append(row.coefficients);cl.append(-row.constant if rule['sense']!='<=' else -np.inf);cu.append(-row.constant if rule['sense']!='>=' else np.inf)
    c=np.zeros(len(labels))
    for i,v in objective.coefficients.items():c[i]=v
    ri=[];ci=[];data=[]
    for r,row in enumerate(rows):
        for i,v in row.items():ri.append(r);ci.append(i);data.append(v)
    A=coo_matrix((data,(ri,ci)),shape=(len(rows),len(labels))).tocsc()
    return dict(raw=obj,labels=labels,domains=domains,c=c,constant=objective.constant,
                sense=obj['objective']['sense'],lower=np.array(lower),upper=np.array(upper),
                integrality=np.array(integrality),A=A,cl=np.array(cl),cu=np.array(cu))


def solve_model(compiled,time_limit=30.):
    m=compiled
    result=milp(m['c']*(1 if m['sense']=='min' else -1),integrality=m['integrality'],
                bounds=Bounds(m['lower'],m['upper']),
                constraints=LinearConstraint(m['A'],m['cl'],m['cu']),
                options={'time_limit':time_limit,'mip_rel_gap':0.0})
    status={0:'optimal',1:'solver_limit',2:'infeasible',3:'unbounded',4:'solver_error'}.get(result.status,'solver_error')
    report=dict(status=status,message=str(result.message),variable_count=len(m['labels']),constraint_count=len(m['cl']))
    if result.status!=0:return report
    x=np.asarray(result.x);ax=m['A']@x;tol=1e-6
    if (not np.isfinite(x).all() or np.any(x<m['lower']-tol) or np.any(x>m['upper']+tol)
        or np.any(ax<m['cl']-tol) or np.any(ax>m['cu']+tol)
        or np.any(np.abs(x[m['integrality']==1]-np.round(x[m['integrality']==1]))>tol)):
        report['status']='solver_error';report['message']='returned assignment fails generated model residual check';return report
    report.update(values=x.tolist(),objective_value=float(m['c']@x+m['constant']),variable_labels=m['labels'])
    return report


def validate_output(m,shape):
    """Validate generated geometry and variable binding, without adding constraints."""
    output=m['raw']['output'];keys(output,{'variable','arcs'})
    variable=output['variable']
    if not isinstance(variable,str) or variable not in m['domains'] or None in m['domains'][variable]:fail('output variable must be an indexed family')
    if not isinstance(output['arcs'],list) or len(output['arcs'])>256:fail('output.arcs must be a list of <=256 geometry bindings')
    paths={}
    for a in output['arcs']:
        keys(a,{'index','polyline_rc'})
        i=a['index'];points=a['polyline_rc']
        if type(i) is not int or i not in m['domains'][variable] or i in paths:fail('invalid/duplicate output arc index')
        if not isinstance(points,list) or not 2<=len(points)<=64:fail('arc polyline requires 2..64 points')
        for p in points:
            if not isinstance(p,list) or len(p)!=2 or any(type(v) is not int or not 0<=v<shape[k] for k,v in enumerate(p)):fail('invalid polyline cell coordinate')
        route=[tuple(points[0])]
        for a,b in zip(points,points[1:]):route.extend(raster_segment(a,b)[1:])
        if len(route)<2 or len(set(route))!=len(route):fail('arc geometry is degenerate or self-repeating')
        paths[i]=route
    if set(paths)!=set(m['domains'][variable]):fail('each output variable index needs exactly one geometry binding')
    return variable,paths


def decode_route(m,report,binding,source,target):
    variable,paths=binding;selected={}
    for (k,i),v in zip(m['labels'],report['values']):
        if k!=variable:continue
        if abs(v)>1e-6 and abs(v-1)>1e-6:raise ModelError('invalid_solution','route variable is not a 0/1 selection')
        if v>.5:selected[i]=paths[i]
    s,t=tuple(map(int,source)),tuple(map(int,target));route=[s];remaining=dict(selected)
    while route[-1]!=t:
        choices=[i for i,p in remaining.items() if p[0]==route[-1]]
        if len(choices)!=1:raise ModelError('invalid_solution','generated assignment has missing or branching continuation')
        path=remaining.pop(choices[0]);route.extend(path[1:])
        if len(set(route))!=len(route):raise ModelError('invalid_solution','generated assignment contains a cycle or repeated cell')
    if remaining:raise ModelError('invalid_solution','generated assignment contains unused branches or disconnected selected arcs')
    return np.asarray(route,dtype=np.int64)


def algebra_prompt(task,dataset,source,target,indices,region_grid,legend):
    from tilebased_core_v85 import DATASETS
    n=DATASETS[dataset]['grid'];side=DATASETS[dataset]['image_side']
    from tilebased_model_templates_v85 import template_prompt
    guidance=template_prompt(task)
    visibility = ("Full image: all visual positions retained." if len(indices) == region_grid**2 else
                  f"Retained positions on the {region_grid}x{region_grid} visual grid: {list(map(int,indices))}.")
    return guidance + f'''{task}
Top-down pixel-art image: {side}x{side}; terrain grid {n}x{n}. Cell coordinates [row,column], zero-based.
S={list(map(int,source))}; G={list(map(int,target))}. Public terrain rules: {legend}
{visibility}
Missing appearance is unknown. Construct the optimization problem yourself from the task and visible evidence.
Declare decision variables, the objective and ALL necessary constraints. No route constraints, endpoint constraints,
connectivity constraints, costs or forbidden-region constraints will be supplied or repaired by the program.
Model useful alternatives and their estimated costs. Do not output a predetermined final route or a full terrain matrix.
The solver executes precisely your generated linear/integer model. Merely describing constraints in prose has no effect.
Return only valid JSON with these seven keys:
"schema":"{SCHEMA}", "sets":{{...}}, "parameters":{{...}}, "variables":{{...}},
"objective":{{"sense":"min" or "max","expression":"..."}}, "constraints":[...], "output":{{...}}.
sets: named lists of unique integer indices. parameters: named finite numbers or numeric arrays (indexed from zero).
variables: named declarations {{"index":"SET_NAME" or null,"type":"binary" or "integer" or "continuous","lb":"expression","ub":"expression"}}.
Use explicit finite bounds. Variables with null index are scalars. Indexed variables are referenced as x[i].
Each constraint is {{"name":"identifier","forall":{{"i":"SET_NAME"}} or {{}},"lhs":"expression","sense":"==" or "<=" or ">=","rhs":"expression"}}.
Expressions support numbers, named parameters, array indexing, variables, +, -, multiplication/division by parameters,
and sum(expr for i in SET if parameter_condition). Example of syntax only: sum(price[j]*z[j] for j in ITEMS).
Conditional expressions (a if parameter_condition else b), comparisons and and/or are allowed on PARAMETERS only.
Do not multiply decision variables, use decision-dependent conditions, Python calls other than sum, attributes, or imports.
Indices in sums/forall must not shadow existing names. Bounds must be parameter-only scalar expressions.
Expressions can be JSON strings or numeric constants. Use the retrieved path template as guidance;
instantiate and return the full mathematical model, including all constraints, yourself.
Output decoding contract: "output":{{"variable":"YOUR_ARC_SELECTION_VARIABLE","arcs":[{{"index":integer,"polyline_rc":[[row,col],...]}}]}}.
Bind each index of one declared variable family to one directed spatial connection. Selected connections must have value 1,
others 0. Each connection may bend: give its ordered polyline cell coordinates; preserve alternative connections where useful.
Adjacent polyline points are interpolated into eight-neighbor unit steps, geometrically only, without obstacle repair.
Model costs for all entered cells on each connection and include the starting cell cost in the objective.
Connections join only at their stated endpoints; geometric crossings do not add connections or constraints.
The decoder only follows the solver-selected arcs from S to G. Branches, cycles, unused selected arcs and gaps fail;
it never selects a better subset or solves another path problem. Your constraints must enforce a valid selection.
At most 256 geometry arcs, 64 points per arc, 1024 expanded variables and 4096 expanded constraints.
No bounds_rc, polygons, matrix, explanation, markdown or extra top-level keys. Close the JSON completely.'''
