"""CLI for turning source-specific raw NPZ records into V8.4 samples."""

import argparse

from planning_data_v84 import MAX_CANDIDATES, MIN_CANDIDATES, convert_directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--source", choices=("nuplan", "pointnav"), required=True)
    parser.add_argument("--min-candidates", type=int, default=MIN_CANDIDATES)
    parser.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES)
    args = parser.parse_args()
    convert_directory(
        args.raw_root,
        args.output_root,
        args.source,
        min_candidates=args.min_candidates,
        max_candidates=args.max_candidates,
    )


if __name__ == "__main__":
    main()
