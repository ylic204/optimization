import copy
import json
import tempfile
import unittest
from pathlib import Path
from tilebased_patch_nodes_v85 import patch_registry,node_rc,validate_patch_bindings,registry_prompt
from tilebased_model_templates_v85 import EXAMPLE
from tilebased_algebra_v85 import compile_model,validate_output,solve_model,decode_route,ModelError
from tilebased_generation_v85 import GenerationTrace,export_model_artifacts

class PatchTests(unittest.TestCase):
    def setUp(self):
        self.registry=patch_registry((4,4),2,[0,0],[0,2])
    def test_roundtrip_all_cells(self):
        for r in range(20):
            for c in range(20):
                reg=patch_registry((20,20),10,[r,c],[r,c])
                self.assertEqual(node_rc(reg['source_node'],reg),[r,c])
    def test_example_solved(self):
        m=compile_model(json.dumps(EXAMPLE));b=validate_output(m,(4,4))
        nodes=validate_patch_bindings(m,b,self.registry)
        self.assertEqual([x['patch_id'] for x in nodes],[0,0,1])
        self.assertEqual(decode_route(m,solve_model(m),b,[0,0],[0,2]).tolist(),[[0,0],[0,1],[0,2]])
    def test_wrong_port_rejected(self):
        m=compile_model(json.dumps(EXAMPLE));b=validate_output(m,(4,4))
        m['raw']['parameters']['head'][0]=4
        with self.assertRaises(ModelError):validate_patch_bindings(m,b,self.registry)
    def test_wrong_cardinality_rejected(self):
        m=compile_model(json.dumps(EXAMPLE));b=validate_output(m,(4,4))
        m['raw']['parameters']['tail'].append(0)
        with self.assertRaises(ModelError):validate_patch_bindings(m,b,self.registry)
    def test_no_implicit_constraints(self):
        raw=copy.deepcopy(EXAMPLE);raw['constraints']=[]
        m=compile_model(json.dumps(raw));validate_patch_bindings(m,validate_output(m,(4,4)),self.registry)
        self.assertEqual(solve_model(m)['values'],[0,0,0])
    def test_same_patch_distinct_endpoints(self):
        reg=patch_registry((20,20),10,[0,0],[0,1])
        self.assertNotEqual(reg['source_node'],reg['target_node'])
        self.assertIn('NOT removed or forbidden',registry_prompt(reg))
    def test_artifacts_on_model_failure(self):
        with tempfile.TemporaryDirectory() as d:
            t=GenerationTrace(d,'sample',False)
            item={'sample_id':'sample','status':'model_failure','generated_model':EXAMPLE}
            export_model_artifacts(t,item)
            self.assertEqual(json.loads(Path(item['model_artifacts']['generated_model']).read_text()),EXAMPLE)
            self.assertNotIn('compiled_model',item['model_artifacts'])
            self.assertTrue(Path(item['model_artifacts']['readable_model']).is_file())

if __name__=='__main__': unittest.main()
