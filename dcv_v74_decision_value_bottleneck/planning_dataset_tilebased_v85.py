"""V8.5 tile-map reader; inputs and privileged evaluation labels are separate."""
from collections import OrderedDict
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from tilebased_core_v85 import CONTRACT, DATASETS, coordinate, task_text


def region_bounds(region_grid, shape):
    # Existing selector consumes normalized centers; here x=column, y=row.
    rows, cols = shape
    return np.asarray([[c*cols/region_grid, (c+1)*cols/region_grid,
                        r*rows/region_grid, (r+1)*rows/region_grid]
                       for r in range(region_grid) for c in range(region_grid)], np.float32)


class TilebasedDatasetV85(Dataset):
    def __init__(self, root, split='train', limit=0, cache_maps=4):
        root = Path(root)
        self.config = json.loads((root / 'dataset.json').read_text())
        if self.config.get('contract') != CONTRACT or not self.config.get('complete'):
            raise ValueError('incomplete or incompatible Tilebased export; rerun preprocessing')
        if split not in ('train', 'val', 'test'):
            raise ValueError('split must be train, val or test')
        self.dataset = self.config['dataset']
        spec = DATASETS[self.dataset]
        if (self.config['grid_shape'] != [spec['grid']]*2 or
                self.config['image_side'] != spec['image_side'] or self.config['region_grid'] != spec['region_grid']):
            raise ValueError('tile geometry differs from the native dataset contract')
        self.folder = root / split
        self.records = [json.loads(line) for line in (self.folder / 'tasks.jsonl').read_text().splitlines() if line.strip()]
        if limit:
            self.records = self.records[:limit]
        if not self.records:
            raise ValueError('empty Tilebased split')
        self._cache, self.cache_maps = OrderedDict(), max(1, cache_maps)
        self.bounds = region_bounds(self.config['region_grid'], self.config['grid_shape'])
        self.signature = json.dumps({k: self.config[k] for k in
            ('contract', 'dataset', 'upstream_revision', 'grid_shape', 'image_side', 'region_grid',
             'coordinate_order', 'graph', 'cost_convention', 'legend')}, sort_keys=True)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        name = record['map_file']
        if Path(name).name != name or not name.endswith('.npz'):
            raise ValueError('map_file must be a local NPZ basename')
        if name not in self._cache:
            with np.load(self.folder / name, allow_pickle=False) as d:
                self._cache[name] = {k: d[k] for k in d.files}
            if len(self._cache) > self.cache_maps:
                self._cache.popitem(last=False)
        self._cache.move_to_end(name)
        data = self._cache[name]
        ti, si = record['target_index'], record['source_index']
        shape = tuple(self.config['grid_shape'])
        source, target = data['sources_rc'][ti, si].astype(np.int64), data['targets_rc'][ti].astype(np.int64)
        if list(coordinate(source, shape)) != record['source_rc'] or list(coordinate(target, shape)) != record['target_rc']:
            raise ValueError('task manifest endpoints do not match labels')
        expected = task_text(self.dataset, source, target)
        if expected != record['task_text']:
            raise ValueError('task text differs from its endpoint/graph contract')
        image = data['image']
        side = self.config['image_side']
        if image.shape != (side, side, 3) or image.dtype != np.uint8 or data['true_costs'].shape != shape:
            raise ValueError('invalid native map or label shape')
        # Explicit separation: only public_inputs is passed to the model/prompt.
        return {'public_inputs': {
                    'image': torch.from_numpy(image.copy()).permute(2,0,1).float()/255.,
                    'task_text': expected, 'source_rc': torch.from_numpy(source.copy()),
                    'target_rc': torch.from_numpy(target.copy()),
                    'region_world_bounds': torch.from_numpy(self.bounds.copy()),
                    'dataset': self.dataset, 'legend': self.config['legend'],
                    'sample_id': record['sample_id']},
                'supervision': {
                    'true_costs': torch.from_numpy(data['true_costs'].copy()).float(),
                    'gt_path_mask': torch.from_numpy(data['gt_path_mask'][ti, si].copy()),
                    'gt_path_rc': torch.from_numpy(data['gt_path_rc'][ti, si].copy()).long(),
                    'gt_path_length': torch.tensor(int(data['gt_path_length'][ti, si])),
                    'optimal_cost': torch.tensor(float(data['optimal_cost'][ti, si]), dtype=torch.float64)}}
