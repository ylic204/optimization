import copy
import json
import unittest
from diagnose_stages_qwen_v85 import check_scene, check_text_model, text_model_prompt
from diagnose_symbolic_qwen_v85 import VQA_SYSTEM
from tilebased_prompts_v85 import navigation_environment_prompt, modeling_system_prompt
from tilebased_symbolic_v85 import MODEL_EXAMPLE

class StagedTests(unittest.TestCase):
    def test_shared_environment_is_connected(self):
        common=navigation_environment_prompt()
        self.assertEqual(VQA_SYSTEM,common)
        self.assertTrue(modeling_system_prompt('navigation','pkmn','algebra').startswith(common))
        self.assertIn('There is no sky',common)
    def test_repetition_is_not_scene_success(self):
        r={'geometry':{'bounds_rc':[0,0,2,4]},'cost':25,'blocked':True}
        report,_=check_scene(json.dumps({'scene':{'regions':[r]*20}}),(3,5),[0],1)
        self.assertTrue(report['schema_valid'])
        self.assertFalse(report['structural_sanity_passed'])
        self.assertEqual(report['duplicate_region_count'],19)
        self.assertEqual(report['blocked_fraction'],1)
    def test_fixture_solves_and_independently_checks(self):
        report,artifact=check_text_model(json.dumps({'model':MODEL_EXAMPLE}))
        self.assertTrue(report['fixture_passed'])
        self.assertEqual(report['true_cost'],5)
        self.assertNotIn([0,2],artifact['path_rc'])
    def test_missing_model_not_completed(self):
        with self.assertRaises(ValueError): check_text_model('{}')
    def test_wrong_objective_is_not_passed(self):
        m=copy.deepcopy(MODEL_EXAMPLE)
        m['objective']['expression']='0'
        report,_=check_text_model(json.dumps({'model':m}))
        self.assertFalse(report['fixture_passed'])
    def test_model_prompt_is_explicit_text_control(self):
        self.assertIn('No image is provided',text_model_prompt())
        self.assertIn('do not repeat or alter the scene',text_model_prompt())

if __name__=='__main__': unittest.main()
