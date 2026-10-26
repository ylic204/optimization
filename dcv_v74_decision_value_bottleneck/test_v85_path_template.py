import copy
import json
import unittest
from tilebased_algebra_v85 import compile_model,solve_model,validate_output,decode_route,ModelError
from tilebased_model_templates_v85 import EXAMPLE,retrieve_template

class TemplateTests(unittest.TestCase):
    def test_executable_example(self):
        m=compile_model(json.dumps(EXAMPLE));s=solve_model(m)
        self.assertEqual(s['objective_value'],3)
        self.assertEqual(decode_route(m,s,validate_output(m,(20,20)),[0,0],[0,2]).tolist(),[[0,0],[0,1],[0,2]])
    def test_no_hidden_template_constraints(self):
        x=copy.deepcopy(EXAMPLE);x['constraints']=[]
        self.assertEqual(solve_model(compile_model(json.dumps(x)))['values'],[0,0,0])
    def test_parameter_map_and_numeric_expressions(self):
        x=copy.deepcopy(EXAMPLE);x['parameters']['lookup']={'grass':2}
        x['objective']['expression']='lookup["grass"]*sum(x[e] for e in E)'
        self.assertEqual(solve_model(compile_model(json.dumps(x)))['objective_value'],2)
    def test_nonlinear_still_fails(self):
        x=copy.deepcopy(EXAMPLE);x['objective']['expression']='x[0]*x[1]'
        with self.assertRaises(ModelError):compile_model(json.dumps(x))
    def test_classification_scope(self):
        self.assertEqual(retrieve_template('Find a minimum total terrain-cost LAND path')['problem_class'],'path_planning')
        with self.assertRaises(ValueError):retrieve_template('forecast demand')

if __name__=='__main__':unittest.main()
