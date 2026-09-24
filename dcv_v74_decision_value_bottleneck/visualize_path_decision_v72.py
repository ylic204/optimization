import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

from config import CFG
from dataset import DCVDataset
from decision import semantic_hard_decision, true_path_costs


def node_positions():
    # Positions for the controlled layered graph; visualization only.
    layers = [[0], [1,2,3], [4,5,6], [7,8,9], [10,11,12], [13]]
    pos = {}
    for x, layer in enumerate(layers):
        ys = np.linspace(0.15, 0.85, len(layer)) if len(layer) > 1 else [0.5]
        for n, y in zip(layer, ys):
            pos[n] = (x, float(y))
    return pos


def edge_set_from_path_mask(path_mask_row):
    return set(np.where(path_mask_row > 0.5)[0].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data_v7/test')
    ap.add_argument('--index', type=int, default=0)
    ap.add_argument('--selected', default='', help='comma-separated patch IDs, e.g. 2,7,18,23,31')
    ap.add_argument('--out', default='decision_visualization.png')
    args = ap.parse_args()

    sample = DCVDataset(args.data)[args.index]
    selected = [int(x) for x in args.selected.split(',') if x.strip()]

    w = torch.zeros(1, CFG.n_patches)
    for j in selected:
        w[0, j] = 1.0

    batch = {k: (v[None] if torch.is_tensor(v) else [v]) for k, v in sample.items() if k != 'name'}
    out = semantic_hard_decision(w, batch)
    pred_idx = int(out['chosen_path_idx'][0])
    pred_cost = float(out['chosen_true_cost'][0])
    opt_cost = float(out['optimal_true_cost'][0])
    regret = float(out['hard_regret'][0])
    is_opt = bool(out['optimal'][0])

    true_pc = true_path_costs(batch)[0]
    opt_indices = torch.where(
        (true_pc - true_pc.min()).abs() <= (CFG.optimal_cost_atol + CFG.optimal_cost_rtol * true_pc.min().abs())
    )[0].cpu().tolist()
    # display one full-information optimum; the metric accepts all tied optima.
    opt_idx = int(opt_indices[0])

    image = sample['image'].permute(1,2,0).numpy()
    edges = sample['edges'].numpy()
    pm = sample['path_mask'].numpy()
    pred_edges = edge_set_from_path_mask(pm[pred_idx])
    opt_edges = edge_set_from_path_mask(pm[opt_idx])

    fig = plt.figure(figsize=(13, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(image)
    ax1.set_title(f'Selected visual regions: K={len(selected)}/{CFG.n_patches}')
    ax1.axis('off')

    for order, j in enumerate(selected, start=1):
        r, c = divmod(j, CFG.grid)
        rect = patches.Rectangle(
            (c * CFG.tile, r * CFG.tile), CFG.tile, CFG.tile,
            linewidth=2, fill=False
        )
        ax1.add_patch(rect)
        ax1.text(c*CFG.tile+3, r*CFG.tile+12, str(order), fontsize=10,
                 bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.8))

    ax2 = fig.add_subplot(1, 2, 2)
    pos = node_positions()
    for ei, (u, v) in enumerate(edges):
        x1, y1 = pos[int(u)]; x2, y2 = pos[int(v)]
        if ei in pred_edges and ei in opt_edges:
            lw, alpha = 5.0, 0.85
        elif ei in pred_edges or ei in opt_edges:
            lw, alpha = 3.0, 0.75
        else:
            lw, alpha = 0.8, 0.20
        ax2.plot([x1, x2], [y1, y2], linewidth=lw, alpha=alpha)

    for n, (x, y) in pos.items():
        ax2.scatter([x], [y], s=90)
        ax2.text(x, y+0.045, str(n), ha='center', fontsize=9)

    ax2.set_title(
        'Final task-cost decision\n'
        f'pred path={pred_idx}, shown optimum={opt_idx}, '
        f'cost={pred_cost:.3f}, C*={opt_cost:.3f}, regret={regret:.4f}, optimal={is_opt}'
    )
    ax2.set_xlim(-0.3, 5.3); ax2.set_ylim(0, 1)
    ax2.axis('off')

    fig.suptitle(
        'V7.2: selected patches + minimum true task-cost path evaluation\n'
        'Different path IDs are accepted if their full-information true cost equals C*.'
    )
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches='tight')
    print('saved', args.out)
    print('selected patches:', selected)
    print('predicted path idx:', pred_idx)
    print('all optimal path indices:', opt_indices)
    print('predicted true cost:', pred_cost)
    print('optimal true cost:', opt_cost)
    print('hard regret:', regret)
    print('optimal by cost:', is_opt)


if __name__ == '__main__':
    main()
