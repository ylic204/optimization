"""Create leakage-free train/val/test folders by log or scene group."""

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


def split_groups(input_root, output_root, val_ratio, test_ratio, seed, group_depth):
    input_root = Path(input_root)
    output_root = Path(output_root)
    groups = defaultdict(list)
    for path in sorted(input_root.rglob("*.npz")):
        relative = path.relative_to(input_root)
        key = "/".join(relative.parts[:group_depth])
        groups[key].append(path)

    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    test_count = round(len(keys) * test_ratio)
    val_count = round(len(keys) * val_ratio)
    assignment = {}
    for index, key in enumerate(keys):
        if index < test_count:
            split = "test"
        elif index < test_count + val_count:
            split = "val"
        else:
            split = "train"
        assignment[key] = split
        for source_path in groups[key]:
            relative = source_path.relative_to(input_root)
            destination = output_root / split / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination)

    manifest = {
        "seed": seed,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "group_depth": group_depth,
        "groups": assignment,
        "sample_counts": {
            split: sum(
                len(groups[key])
                for key, assigned in assignment.items()
                if assigned == split
            )
            for split in ("train", "val", "test")
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--group-depth", type=int, default=1)
    args = parser.parse_args()
    manifest = split_groups(
        args.input_root,
        args.output_root,
        args.val_ratio,
        args.test_ratio,
        args.seed,
        args.group_depth,
    )
    print(json.dumps(manifest["sample_counts"], indent=2))


if __name__ == "__main__":
    main()
