"""Public geometric patch/port registry. No terrain or GT is read here."""
import math


def patch_registry(shape, region_grid, source, target):
    h,w=map(int,shape)
    if h % region_grid or w % region_grid:
        raise ValueError('terrain grid must divide evenly into visual regions')
    ph,pw=h//region_grid,w//region_grid
    slots=ph*pw
    def node(rc):
        r,c=map(int,rc)
        if not (0<=r<h and 0<=c<w): raise ValueError('endpoint outside map')
        return ((r//ph)*region_grid+c//pw)*slots+(r%ph)*pw+c%pw
    return dict(contract='patch_ports_v1',shape=[h,w],region_grid=region_grid,
                patch_shape=[ph,pw],ports_per_patch=slots,
                source_node=node(source),target_node=node(target),
                source_rc=list(map(int,source)),target_rc=list(map(int,target)))


def node_rc(node,registry):
    h,w=registry['shape'];g=registry['region_grid']
    ph,pw=registry['patch_shape'];slots=registry['ports_per_patch']
    if type(node) is not int or not 0<=node<h*w: raise ValueError('unknown patch-port node ID')
    patch,port=divmod(node,slots)
    return [(patch//g)*ph+port//pw,(patch%g)*pw+port%pw]


def registry_prompt(registry):
    g=registry['region_grid'];ph,pw=registry['patch_shape'];k=registry['ports_per_patch']
    return f'''PUBLIC PATCH NODE REGISTRY (geometry only; all patches remain available):
Visual patches form a {g}x{g} row-major grid; each covers {ph}x{pw} terrain cells.
A patch is an area, not an automatically connected point. Use fixed cell ports
inside each patch to distinguish water/land and disconnected passage positions.
For cell [r,c]: patch_id=(r//{ph})*{g}+c//{pw}; port=(r%{ph})*{pw}+c%{pw};
node_id=patch_id*{k}+port. This uniquely fixes the node coordinate.
S node={registry['source_node']} at {registry['source_rc']}; G node={registry['target_node']} at {registry['target_rc']}.
N contains only IDs referenced by your task-relevant connections plus S/G.
Do not enumerate the registry or invent IDs. E uses local contiguous IDs 0..len(E)-1.
Unretained visual patches are unknown, NOT removed or forbidden. Adjacency alone
never proves traversability. Connections can cross patches using explicit bends;
judge their feasibility and total cost from the visible evidence. Endpoints of
EVERY polyline must equal the fixed coordinates of tail[e] and head[e]. Joining
edges must use the same port. Different ports in a patch are not automatically
connected: generate an explicit connection if needed. No implicit routing occurs.
'''


def validate_patch_bindings(model,binding,registry):
    """Reject inconsistent references, never insert flow or feasibility constraints."""
    from tilebased_algebra_v85 import ModelError
    def fail(message): raise ModelError('model_error',message)
    raw=model['raw'];p=raw['parameters'];sets=raw['sets']
    if 'N' not in sets or 'E' not in sets: fail('patch interface requires N and E sets')
    nodes=sets['N'];edges=sets['E']
    try: coords={n:node_rc(n,registry) for n in nodes}
    except ValueError as e: fail(str(e))
    if edges!=list(range(len(edges))): fail('E must use contiguous local edge IDs')
    for key in ('tail','head','cost','allowed'):
        if not isinstance(p.get(key),list) or len(p[key])!=len(edges):
            fail(f'{key} must have exactly len(E)={len(edges)} entries')
    for key,expected in [('s',registry['source_node']),('t',registry['target_node'])]:
        if type(p.get(key)) is not int or p[key]!=expected or expected not in coords:
            fail(f'{key} must equal public endpoint node {expected} in N')
    if set(binding[1])!=set(edges): fail('geometry bindings must match E')
    for e in edges:
        a,b=p['tail'][e],p['head'][e]
        if type(a) is not int or type(b) is not int or a not in coords or b not in coords:
            fail(f'edge {e}: tail/head not in declared patch-port nodes')
        route=binding[1][e]
        if list(route[0])!=coords[a] or list(route[-1])!=coords[b]:
            fail(f'edge {e}: geometry endpoints disagree with patch-port IDs')
        if type(p['allowed'][e]) not in (int,float) or p['allowed'][e] not in (0,1): fail('allowed must be 0/1')
        if type(p['cost'][e]) not in (int,float) or not math.isfinite(p['cost'][e]) or p['cost'][e]<=0: fail('edge cost must be positive')
    return [{'node_id':n,'patch_id':n//registry['ports_per_patch'],
             'port':n%registry['ports_per_patch'],'rc':coords[n]} for n in nodes]
