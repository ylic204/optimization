
## Diagnose repeated Qwen output before evaluation

`diagnose_symbolic_qwen_v85.py` runs one public RGB image and task without loading
GT costs or training the selector. It separates simple English VQA from the exact
current symbolic-model prompt. First run native-only VQA to bypass the custom
backbone completely:

```bash
python diagnose_symbolic_qwen_v85.py \
  --data /data/v85_tilebased/pkmn_land_v3 \
  --vlm-model /data/lyi/models/Qwen3-VL-8B-Instruct \
  --device cuda:0 --load-4bit --sample-index 0 \
  --backends native --probes vqa \
  --output results/qwen_native_vqa_r1
```

Then compare both backends and both prompts:

```bash
python diagnose_symbolic_qwen_v85.py \
  --data /data/v85_tilebased/pkmn_land_v3 \
  --vlm-model /data/lyi/models/Qwen3-VL-8B-Instruct \
  --device cuda:0 --load-4bit --sample-index 0 \
  --backends native custom --probes vqa symbolic \
  --model-max-new-tokens 4096 \
  --output results/qwen_backend_diagnostic_r1
```

Use a new output directory for each run. `--sample-index` selects a zero-based
**task manifest index**, not a map ID. `--blank-control` adds a gray-image control;
image sensitivity alone does not establish visual correctness.

Outputs include `input.png`, `public_task.json`, exact `*.prompt.json`/`*.chat.txt`,
`original_vqa_native.txt`, `original_vqa_custom.txt`,
`original_symbolic_native.txt`, `original_symbolic_custom.txt`, `outcomes.jsonl`,
and `summary.json`. Raw answers are saved incrementally under `raw/` and printed
unless `--quiet` is used. `*.parity.json` compares official processor tensors and
custom inputs; a separate first-step logit comparison uses shared custom inputs.
Equal first-step logits do **not** prove subsequent cache/position behavior is correct.
Both diagnostic generation paths stop at EOS or the token budget, without the
production JSON/array guard. Symbolic output is compiled for validation, but this
script does not solve or report navigation SR.

If native answers are sensible but custom answers repeat, investigate the custom
inference path. If both understand the image but fail symbolic generation,
investigate modeling prompt/schema complexity. VQA correctness requires inspecting
the saved image and answer; JSON completion and compilation are not accuracy scores.
CPU tests cover output diagnosis and image-only loading. GPU inference must be run
with the local Qwen model and dataset; it is not covered by these CPU tests.
