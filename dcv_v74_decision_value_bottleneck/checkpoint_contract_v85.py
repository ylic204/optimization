"""Fail-fast checkpoint checks shared by training and evaluation."""

from bev_geometry_v85 import SCHEMA_REVISION
from raw_record_bev_v85 import INPUT_MODE
from bev_solver_environment_v85 import SELECTOR_TEXT_CONTRACT, ENVIRONMENT_CONTRACT


def validate_checkpoint_contract(checkpoint, args, dataset, stage):
    mode = checkpoint.get("input_mode", "legacy_first_person_rgb")
    cross_view = mode != INPUT_MODE
    if cross_view:
        if stage != "metric" or not getattr(args, "allow_cross_view_resume", False):
            raise ValueError("cross-view weights are allowed only as explicit metric-stage initialization")
    else:
        if checkpoint.get("bev_config_json") != dataset.bev_config_json:
            raise ValueError("checkpoint and data use different BEV geometry")
        if checkpoint.get("data_schema_revision") != SCHEMA_REVISION:
            raise ValueError("checkpoint uses stale BEV labels; regenerate data and retrain the metric stage")
        if checkpoint.get("selector_text_contract") != SELECTOR_TEXT_CONTRACT:
            raise ValueError("checkpoint lacks task/model requirements conditioning; retrain the metric stage")
        if stage == "rl" and checkpoint.get("training_stage") == "rl":
            expected = ENVIRONMENT_CONTRACT if args.rl_backend == "llm_solver" else "candidate_metric_proxy_v1"
            if checkpoint.get("reward_environment") != expected:
                raise ValueError("RL checkpoint reward environment mismatch")
    saved = checkpoint.get("args", {})
    for key in ("region_grid", "prune_layer", "hidden", "latent_dim", "mask_steps", "max_text_length"):
        if key in saved and saved[key] != getattr(args, key):
            raise ValueError(f"checkpoint architecture mismatch: {key}")
    previous = checkpoint.get("training_stage")
    if stage == "distill" and previous not in ("metric", "distill", "rl"):
        raise ValueError("distillation requires a metric-pretrained checkpoint")
    if stage == "rl" and previous not in ("distill", "rl"):
        raise ValueError("RL post-training requires a distilled checkpoint")
