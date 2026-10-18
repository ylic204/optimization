"""Convert official Tilebased NPZ into V8.5 map records and task manifests.

No image rerendering, no new start/goal sampling, no step constraints. Labels
are checked against an exact solver and stored separately from public inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from tilebased_core_v85 import CONTRACT, DATASETS, PKMN_LEGEND, UPSTREAM_REV, coordinate, task_text, verify_gt
from download_tilebased_v85 import SHA256, sha256


def preprocess(raw_root, output_root, dataset='pkmn', max_maps=0, previews=3, legend_file=None):
    raw_root, output = Path(raw_root), Path(output_root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'{output} is not empty; choose a fresh directory (no mixed exports)')
    spec = DATASETS[dataset]
    if dataset == 'warcraft' and not legend_file:
        raise ValueError('Warcraft requires --legend-file with a verified terrain-to-cost legend; use pkmn for the ready-made legend')
    legend = Path(legend_file).read_text().strip() if legend_file else PKMN_LEGEND
    if not legend:
        raise ValueError('legend cannot be empty')
    # Fail before export on incomplete/corrupt archives, including offline copies.
    hashes = {}
    for split in ('train', 'val', 'test'):
        path = raw_root / dataset / f'{split}.npz'
        hashes[split] = sha256(path)
        if hashes[split] != SHA256[f'{dataset}/{split}']:
            raise ValueError(f'{path}: expected the pinned official archive')
    output.mkdir(parents=True, exist_ok=True)
    contract = dict(contract=CONTRACT, dataset=dataset, upstream_revision=UPSTREAM_REV,
                    image_side=spec['image_side'], grid_shape=[spec['grid']] * 2,
                    region_grid=spec['region_grid'], coordinate_order='row_column_zero_based',
                    graph='8_neighbors_all_cells_finite_no_diagonal_multiplier',
                    cost_convention='sum_visited_nodes_including_source_and_target',
                    legend=legend, max_maps_per_split=max_maps,
                    source_sha256=hashes, complete=False)
    config_path = output / 'dataset.json'
    config_path.write_text(json.dumps(contract, indent=2) + '\n')
    summary = {'splits': {}, 'exact_duplicate_images_across_splits': [],
               'note': 'Original map splits preserved. Overlapping crops from the same upstream source world are not an unseen-world split.'}
    seen_hashes = {}
    for split in ('train', 'val', 'test'):
        folder = output / split
        folder.mkdir()
        with np.load(raw_root / dataset / f'{split}.npz', allow_pickle=False) as archive:
            arrays = {k: archive[k] for k in ('maps', 'costs', 'sources', 'targets', 'paths')}
        n = len(arrays['maps'])
        g, side = spec['grid'], spec['image_side']
        t, s = arrays['sources'].shape[1:3]
        shapes = {'maps': (n, side, side, 3), 'costs': (n, g, g),
                  'sources': (n, t, s, 2), 'targets': (n, t, 2), 'paths': (n, t, s, g, g)}
        for key, shape in shapes.items():
            if arrays[key].shape != shape:
                raise ValueError(f'{split}/{key}: expected {shape}, got {arrays[key].shape}')
        count = min(n, max_maps) if max_maps else n
        with (folder / 'tasks.jsonl').open('w') as stream:
            for i in range(count):
                image = arrays['maps'][i]
                if image.dtype != np.uint8:
                    raise ValueError('maps must be native uint8 RGB')
                costs = arrays['costs'][i]
                if not np.isclose(costs[..., None], spec['cost_values'], atol=1e-5).any(-1).all():
                    raise ValueError('unknown terrain cost')
                digest = hashlib.sha256(image.tobytes()).hexdigest()
                if digest in seen_hashes and seen_hashes[digest][0] != split:
                    summary['exact_duplicate_images_across_splits'].append([seen_hashes[digest], [split, i]])
                seen_hashes.setdefault(digest, [split, i])
                ordered = np.full((t, s, g * g, 2), -1, dtype=np.int16)
                lengths = np.zeros((t, s), np.int16)
                optimum = np.zeros((t, s), np.float64)
                for ti in range(t):
                    target = coordinate(arrays['targets'][i, ti], costs.shape)
                    for si in range(s):
                        source = coordinate(arrays['sources'][i, ti, si], costs.shape)
                        route, value = verify_gt(costs, arrays['paths'][i, ti, si], source, target)
                        ordered[ti, si, :len(route)] = route
                        lengths[ti, si], optimum[ti, si] = len(route), value
                        record = {'sample_id': f'{dataset}/{split}/{i:06d}/{ti}/{si}',
                                  'map_file': f'{i:06d}.npz', 'map_sha256': digest,
                                  'target_index': ti, 'source_index': si,
                                  'source_rc': list(source), 'target_rc': list(target),
                                  'task_text': task_text(dataset, source, target)}
                        stream.write(json.dumps(record) + '\n')
                np.savez_compressed(folder / f'{i:06d}.npz', image=image, true_costs=costs,
                                    sources_rc=arrays['sources'][i], targets_rc=arrays['targets'][i],
                                    gt_path_mask=arrays['paths'][i].astype(bool),
                                    gt_path_rc=ordered, gt_path_length=lengths, optimal_cost=optimum)
                if i < previews:
                    preview = output / 'previews' / split
                    preview.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(image).save(preview / f'{i:06d}_input.png')
                    # Explicitly evaluation-only; never loaded by the Dataset.
                    marked = image.copy()
                    tile = side // g
                    mask = arrays['paths'][i, 0, 0].astype(bool)
                    for r, c in np.argwhere(mask):
                        y, x = int((r + .5) * tile), int((c + .5) * tile)
                        marked[max(0,y-1):y+2, max(0,x-1):x+2] = [255, 40, 180]
                    Image.fromarray(marked).save(preview / f'{i:06d}_GT_ONLY.png')
                if (i + 1) % 100 == 0:
                    print(f'{dataset}/{split}: {i+1}/{count} maps verified', flush=True)
        summary['splits'][split] = {'maps': count, 'tasks': count*t*s, 'all_gt_verified': True}
        print(f'{dataset}/{split}: {count} maps, {count*t*s} tasks', flush=True)
    summary['exact_duplicate_images_across_splits_count'] = len(summary['exact_duplicate_images_across_splits'])
    (output / 'preprocess_report.json').write_text(json.dumps(summary, indent=2) + '\n')
    contract['complete'] = True
    config_path.write_text(json.dumps(contract, indent=2) + '\n')
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root', required=True)
    p.add_argument('--output-root', required=True)
    p.add_argument('--dataset', choices=list(DATASETS), default='pkmn')
    p.add_argument('--max-maps-per-split', type=int, default=0, help='0=all; smaller values are smoke subsets')
    p.add_argument('--previews', type=int, default=3)
    p.add_argument('--legend-file', help='public, verified terrain legend; required for Warcraft')
    a = p.parse_args()
    if a.max_maps_per_split < 0 or a.previews < 0:
        p.error('map and preview counts must be nonnegative')
    preprocess(a.raw_root, a.output_root, a.dataset, a.max_maps_per_split, a.previews, a.legend_file)

if __name__ == '__main__':
    main()
