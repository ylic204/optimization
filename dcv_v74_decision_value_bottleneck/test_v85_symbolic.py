import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from tilebased_symbolic_v85 import (SCHEMA, MODEL_EXAMPLE, compile_symbolic,
                                   decode_symbolic, build_scene, symbolic_prompt)
from tilebased_algebra_v85 import solve_model, ModelError
from tilebased_core_v85 import shortest_path, path_cost
from tilebased_generation_v85 import GenerationTrace, export_model_artifacts


def example(shape=(3,5)):
    return {'schema':SCHEMA,'scene':{'regions':[{'geometry':{'bounds_rc':[0,0,shape[0]-1,shape[1]-1]},'cost':1,'blocked':False}]},
            'model':copy.deepcopy(MODEL_EXAMPLE)}


def run(obj,shape=(3,5),source=(1,0),target=(1,4)):
    m,e=compile_symbolic(json.dumps(obj),shape,source,target,[0],1)
    s=solve_model(m)
    route=decode_symbolic(m,s,source,target) if s['status']=='optimal' else None
    return m,e,s,route


class SymbolicTests(unittest.TestCase):
    def test_cost_objective_changes_route(self):
        obj=example();obj['model']['sets']['A']['connectivity']=4
        obj['scene']['regions'].append({'geometry':{'bounds_rc':[1,1,1,3]},'cost':8,'blocked':False})
        m,e,s,route=run(obj)
        self.assertEqual(s['objective_value'],7)
        self.assertGreater(len(route)-1,4)
        obj['model']['parameters']['length']='space.step_length'
        obj['model']['objective']['expression']='sum(length[e]*x[e] for e in A)'
        m,e,s,route=run(obj)
        self.assertEqual(len(route)-1,4)
        self.assertEqual(s['objective_value'],4)
    def test_removing_block_constraint_really_changes_solver(self):
        obj=example();obj['model']['sets']['A']['connectivity']=4
        obj['scene']['regions'].append({'geometry':{'bounds_rc':[0,2,2,2]},'cost':1,'blocked':True})
        self.assertEqual(run(obj)[2]['status'],'infeasible')
        obj['model']['constraints']=[r for r in obj['model']['constraints'] if r['name'] not in ('leave_land','enter_land')]
        m,e,s,route=run(obj)
        self.assertEqual(s['status'],'optimal')
        self.assertTrue(any(e['blocked'][r][c] for r,c in route))
    def test_missing_balance_not_repaired(self):
        obj=example();obj['model']['constraints']=[]
        m,e=compile_symbolic(json.dumps(obj),(3,5),(1,0),(1,4),[0],1)
        s=solve_model(m)
        self.assertTrue(all(v==0 for v in s['values']))
        with self.assertRaises(ModelError):decode_symbolic(m,s,(1,0),(1,4))
    def test_no_corner_cut(self):
        obj=example((2,2))
        obj['scene']['regions'].extend([{'geometry':{'bounds_rc':[0,1,0,1]},'cost':1,'blocked':True},
                                      {'geometry':{'bounds_rc':[1,0,1,0]},'cost':1,'blocked':True}])
        self.assertEqual(run(obj,(2,2),(0,0),(1,1))[2]['status'],'infeasible')
        obj['model']['constraints']=[r for r in obj['model']['constraints'] if r['name']!='clear_corner']
        self.assertEqual(run(obj,(2,2),(0,0),(1,1))[3].tolist(),[[0,0],[1,1]])
    def test_unknown_nodes_not_deleted(self):
        obj=example();obj['scene']['regions']=[]
        a,ea=compile_symbolic(json.dumps(obj),(3,5),(1,0),(1,4),[0],2)
        b,eb=compile_symbolic(json.dumps(obj),(3,5),(1,0),(1,4),[0,1,2,3],2)
        self.assertEqual(a['grid_arcs'],b['grid_arcs'])
        self.assertEqual(ea['unknown_fraction'],1)
        self.assertEqual(ea['cost'],eb['cost'])
        self.assertFalse(any(any(row) for row in ea['blocked']))
        self.assertNotEqual(ea['direct_visual_coverage'],eb['direct_visual_coverage'])
    def test_full_20_grid_matches_reference(self):
        obj=example((20,20))
        m,e,s,route=run(obj,(20,20),(10,19),(8,5))
        costs=np.array(e['cost']);allowed=~np.array(e['blocked'])
        reference=shortest_path(costs,(10,19),(8,5),allowed=allowed,no_corner_cut=True)
        self.assertEqual(path_cost(costs,route),path_cost(costs,reference))
        self.assertGreater(len(m['labels']),1024)
        self.assertEqual(len(m['raw']['constraints']),len(MODEL_EXAMPLE['constraints']))
    def test_nonlinear_rejected(self):
        obj=example();obj['model']['objective']['expression']='x[0]*x[1]'
        with self.assertRaises(ModelError):run(obj)
    def test_no_arbitrary_binding_or_explicit_edges(self):
        for binding in ('truth.cost','__import__("os")'):
            obj=example();obj['model']['parameters']['c']=binding
            with self.assertRaises(ModelError):run(obj)
        obj=example();obj['model']['sets']['A']=[0,1,2]
        with self.assertRaises(ModelError):run(obj)
    def test_symbol_aliases_are_generated(self):
        obj=example();m=obj['model'];m['variables']['move']=m['variables'].pop('x');m['output']['variable']='move'
        m['objective']['expression']=m['objective']['expression'].replace('x[','move[')
        for r in m['constraints']:r['lhs']=r['lhs'].replace('x[','move[')
        self.assertEqual(run(obj)[2]['status'],'optimal')
    def test_same_endpoint(self):
        self.assertEqual(run(example(),source=(1,0),target=(1,0))[3].tolist(),[[1,0]])
    def test_polygon_rasterization_and_export(self):
        obj=example();obj['scene']['regions']=[{'geometry':{'polygon_rc':[[0,0],[0,5],[3,5],[3,0]]},'cost':1.5,'blocked':False}]
        m,e,s,route=run(obj)
        self.assertEqual(e['unknown_fraction'],0)
        with tempfile.TemporaryDirectory() as d:
            trace=GenerationTrace(d,'test',False)
            item=dict(generated_model=obj,symbolic_model=obj['model'],scene_estimate=e,expanded_model=m['raw'],solver_result=s,path_rc=route.tolist())
            export_model_artifacts(trace,item)
            for path in item['model_artifacts'].values():self.assertTrue(Path(path).is_file())
            self.assertIn('balance',Path(item['model_artifacts']['readable_model']).read_text())
    def test_prompt_no_registry_and_shared_contract(self):
        p=symbolic_prompt('find minimum cost','pkmn',[0,0],[0,2],list(range(100)),10,'grass=1.5')
        self.assertIn('neighbor_pairs',p);self.assertNotIn('node_id=',p)
        self.assertIn('Full image is visible.',p)
        self.assertNotIn('polyline_rc',p)

if __name__=='__main__':unittest.main()
