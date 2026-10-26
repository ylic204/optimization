import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from diagnose_symbolic_qwen_v85 import output_diagnostics, read_public_sample
from tilebased_core_v85 import CONTRACT, LAND_PROTOCOL


class DiagnosticTests(unittest.TestCase):
    def test_repeated_regions_without_model(self):
        region = {'geometry': {'bounds_rc': [0, 0, 19, 19]}, 'cost': 25, 'blocked': True}
        result = output_diagnostics(json.dumps({'scene': {'regions': [region] * 20}}))
        self.assertEqual(result['duplicate_region_count'], 19)
        self.assertFalse(result['has_model'])

    def test_partial_json_is_not_complete(self):
        self.assertFalse(output_diagnostics('{"scene":')['json_complete'])

    def test_reads_image_without_ground_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'val').mkdir()
            config = dict(contract=CONTRACT, complete=True, navigation_protocol=LAND_PROTOCOL,
                          image_side=4, grid_shape=[2, 2])
            (root / 'dataset.json').write_text(json.dumps(config))
            record = dict(sample_id='demo', source_rc=[0, 0], target_rc=[1, 1],
                          task_text='Navigate to G.', map_file='map.npz', private_gt='must not propagate')
            (root / 'val' / 'tasks.jsonl').write_text(json.dumps(record) + '\n')
            # Deliberately no costs or GT arrays: diagnostic must only require RGB.
            np.savez(root / 'val' / 'map.npz', image=np.zeros((4, 4, 3), dtype=np.uint8))
            _, public, image = read_public_sample(root, 'val', 0)
            self.assertNotIn('private_gt', public)
            self.assertEqual(image.shape, (4, 4, 3))
            with self.assertRaises(ValueError):
                read_public_sample(root, 'val', 1)
            record['map_file'] = '../map.npz'
            (root / 'val' / 'tasks.jsonl').write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                read_public_sample(root, 'val', 0)


if __name__ == '__main__':
    unittest.main()
