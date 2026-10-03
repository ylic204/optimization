"""Evaluate a V8.5 BEV checkpoint with full/random/learned token masks."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from gradient_flow_selector_v85 import build_v85_model
from optimization_spec_v83 import OptimizationTask
from planning_dataset_bev_v85 import PlanningDatasetBEVV85
from raw_record_bev_v85 import INPUT_MODE
from train_gradient_flow_selector_v85 import validate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--nuplan-task", required=True)
    parser.add_argument("--pointnav-task", required=True)
    parser.add_argument("--vlm-model", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", default="results/v85_bev_evaluation.json")
    parser.add_argument("--load-4bit", action="store_true")
    return parser.parse_args()


def main():
    cli = parse_args()
    checkpoint = torch.load(cli.checkpoint, map_location="cpu")
    if "args" not in checkpoint:
        raise ValueError("checkpoint does not contain its V8.5 architecture args")
    if checkpoint.get("input_mode") != INPUT_MODE:
        raise ValueError(
            f"checkpoint is not tagged with input_mode={INPUT_MODE!r}; "
            "cross-view evaluation is not valid"
        )

    saved = dict(checkpoint["args"])
    saved.update(
        {
            "vlm_model": cli.vlm_model or saved["vlm_model"],
            "device": cli.device,
            "batch": cli.batch,
            "workers": cli.workers,
            "load_4bit": cli.load_4bit or saved.get("load_4bit", False),
            "nuplan_task": cli.nuplan_task,
            "pointnav_task": cli.pointnav_task,
        }
    )
    saved.setdefault("nuplan_text", "")
    saved.setdefault("pointnav_text", "")
    args = SimpleNamespace(**saved)

    device = torch.device(cli.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = build_v85_model(args, device)
    model.selector.load_state_dict(checkpoint["selector"])
    model.metric_head.load_state_dict(checkpoint["metric_head"])
    model.mask_flow.load_state_dict(checkpoint["mask_flow"])

    tasks = [
        OptimizationTask.load(cli.nuplan_task),
        OptimizationTask.load(cli.pointnav_task),
    ]
    image_side = args.region_grid * 32
    dataset = PlanningDatasetBEVV85(cli.data, image_side)
    if checkpoint.get("bev_config_json") != dataset.bev_config_json:
        raise ValueError("checkpoint and evaluation data use different BEV geometry")
    loader = DataLoader(
        dataset,
        cli.batch,
        shuffle=False,
        num_workers=cli.workers,
    )
    metrics = validate(model, loader, args, device, tasks)
    output = Path(cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
