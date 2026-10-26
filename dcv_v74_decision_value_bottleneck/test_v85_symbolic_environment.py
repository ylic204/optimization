"""CPU integration test with a frozen response backend, no torch dependency."""
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from test_v85_symbolic import example

class Tensor:
    def __init__(self,value): self.a=np.asarray(value)
    @property
    def shape(self):return self.a.shape
    def __len__(self):return len(self.a)
    def __getitem__(self,index):return Tensor(self.a[index])
    def sort(self,dim):return types.SimpleNamespace(values=Tensor(np.sort(self.a,axis=dim)))
    def cpu(self):return self
    def numpy(self):return self.a
    def tolist(self):return self.a.tolist()
    def __float__(self):return float(self.a)
    def __int__(self):return int(self.a)

class Backend:
    def __init__(self):self.answer=json.dumps(example((20,20)))
    def generate_selected_report(self,state,indices,prompt,max_new_tokens,system_prompt=None,output_guard=None,text_callback=None):
        assert 'neighbor_pairs' in prompt and 'polyline_rc' not in prompt
        if text_callback:text_callback(self.answer)
        return dict(text=self.answer,finish_reason=output_guard(self.answer),generated_tokens=1)

class EnvironmentTests(unittest.TestCase):
    def test_pipeline_and_independent_evaluation(self):
        fake=types.ModuleType('torch');fake.no_grad=lambda:lambda f:f
        path=Path(__file__).with_name('tilebased_solver_environment_v85.py')
        spec=importlib.util.spec_from_file_location('symbolic_env_test',path);module=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{'torch':fake}):spec.loader.exec_module(module)
        from tilebased_core_v85 import LAND_PROTOCOL
        backend=Backend()
        pub=dict(source_rc=Tensor([[0,0]]),target_rc=Tensor([[0,2]]),dataset=['pkmn'],
                 navigation_protocol=[LAND_PROTOCOL],sample_id=['test'],task_text=['Find a minimum total terrain-cost LAND path'],legend=['grass=1.5'])
        truth=dict(true_costs=Tensor(np.ones((1,20,20))),allowed_mask=Tensor(np.ones((1,20,20),bool)),
                   optimal_cost=Tensor([3]),shortest_path_length=Tensor([2]))
        encoded={'vision_state':{'tokens':np.zeros((1,4,2))}}
        with tempfile.TemporaryDirectory() as d:
            env=module.FrozenTilebasedSolverEnvironment(backend,2,prompt_variant='navigation',model_representation='algebra',generation_log_dir=d)
            good=env.evaluate(encoded,Tensor([[0,1,2,3]]),pub,truth)[0]
            self.assertEqual(good['status'],'success')
            self.assertEqual(good['solver_result']['objective_value'],3)
            self.assertNotIn('patch_node_registry',good)
            self.assertIn('symbolic_model',good['model_artifacts'])
            self.assertIn('compiled_model',good['model_artifacts'])
            self.assertIn('path_rc',good['model_artifacts'])
            r,c=good['path_rc'][1];truth['allowed_mask'].a[0,r,c]=False
            bad=env.evaluate(encoded,Tensor([[0,1,2,3]]),pub,truth)[0]
            self.assertEqual(bad['status'],'constraint_failure')
            self.assertEqual(good['generated_model'],bad['generated_model'])
            self.assertEqual(good['solver_result'],bad['solver_result'])
            self.assertEqual(good['path_rc'],bad['path_rc'])
            obj=example((20,20));obj['model']['constraints']=[];backend.answer=json.dumps(obj)
            missing=env.evaluate(encoded,Tensor([[0,1,2,3]]),pub,truth)[0]
            self.assertEqual(missing['failure_category'],'invalid_solution')
            self.assertEqual(missing['solver_result']['constraint_count'],0)
            self.assertIn('solver_result',missing['model_artifacts'])
            backend.answer='{}'
            invalid=env.evaluate(encoded,Tensor([[0,1,2,3]]),pub,truth)[0]
            self.assertEqual(invalid['failure_category'],'model_error')
            self.assertIn('generated_model',invalid['model_artifacts'])
            self.assertNotIn('compiled_model',invalid['model_artifacts'])

if __name__=='__main__':unittest.main()
