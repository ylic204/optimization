"""CLI for turning source-specific raw NPZ records into V8.4 samples."""

import argparse

from planning_data_v84 import convert_directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--source", choices=("nuplan", "pointnav"), required=True)
    args = parser.parse_args()
    convert_directory(args.raw_root, args.output_root, args.source)


if __name__ == "__main__":
    main()
