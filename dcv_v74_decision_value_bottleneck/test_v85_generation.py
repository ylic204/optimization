import json
import tempfile
import unittest
from pathlib import Path
from tilebased_generation_v85 import AlgebraOutputGuard, GenerationTrace

class GenerationTests(unittest.TestCase):
    def test_legal_equal_costs(self):
        self.assertEqual(AlgebraOutputGuard()(json.dumps({'cost':[1.5]*1024})), 'json_complete')
    def test_partial_overflow(self):
        g=AlgebraOutputGuard()
        self.assertEqual(g('{"cost":[' + '1.5,'*1024+'1'), 'array_limit')
        self.assertIn('1024', g.error)
    def test_strings_and_nested_arrays(self):
        data={'expression':'[1,'*2000, 'geometry':[[1,2]]*100}
        self.assertEqual(AlgebraOutputGuard()(json.dumps(data)), 'json_complete')
    def test_incomplete(self):
        self.assertIsNone(AlgebraOutputGuard()('{"cost":[1,1,'))
    def test_trace_unique_preserves_raw(self):
        with tempfile.TemporaryDirectory() as d:
            a=GenerationTrace(d,'map/0',False);b=GenerationTrace(d,'map/0',False)
            self.assertNotEqual(a.path,b.path)
            a('{"cost":[');a.finish('{"cost":[1,','array_limit')
            self.assertEqual(a.path.read_text(),'{"cost":[1,')

if __name__=='__main__': unittest.main()
