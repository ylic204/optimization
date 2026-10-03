"""CLI: raw nuPlan BEV records -> V8.5 candidate-planning samples."""

import argparse

from planning_data_bev_v85 import convert_bev_directory
from planning_data_v84 import MAX_CANDIDATES, MIN_CANDIDATES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--min-candidates", type=int, default=MIN_CANDIDATES)
    parser.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES)
    args = parser.parse_args()
    convert_bev_directory(
        args.raw_root,
        args.output_root,
        min_candidates=args.min_candidates,
        max_candidates=args.max_candidates,
    )


if __name__ == "__main__":
    main()
