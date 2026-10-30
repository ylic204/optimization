
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

## Staged native diagnosis with the shared environment prompt

The full `prompts/tilebased_land_system_v85.txt` is now loaded by formal symbolic
modeling, the native/custom diagnostic VQA, and all stages below. The formal
modeling suffix only adds output requirements. The Full validation signature now
includes this prompt file, so validation from an older prompt cannot qualify a
new selector run.

Run each stage separately in the script directory. Review the visual stages
before running `joint`; no automatic metric can certify image interpretation.
All stages below use native Qwen inference, not the custom selector backend.

```bash
export VLM_MODEL=/data/lyi/models/Qwen3-VL-8B-Instruct
export TILE_DATA=/data/v85_tilebased/pkmn_land_v3

# 1. Visual description: compare model_answer.txt against input.png.
python diagnose_stages_qwen_v85.py --stage vqa --data "$TILE_DATA" \
  --vlm-model "$VLM_MODEL" --device cuda:0 --load-4bit \
  --sample-index 0 --output results/stage_vqa_r2

# 2. Scene evidence ONLY: no variables, objective or constraints requested.
python diagnose_stages_qwen_v85.py --stage scene --data "$TILE_DATA" \
  --vlm-model "$VLM_MODEL" --device cuda:0 --load-4bit \
  --sample-index 0 --output results/stage_scene_r2

# 3. Pure text modeling: no image, dataset, or ground-truth map loaded.
python diagnose_stages_qwen_v85.py --stage model \
  --vlm-model "$VLM_MODEL" --device cuda:0 --load-4bit \
  --output results/stage_model_r2

# 4. Only after reviewing stages 1/2 and checking stage 3:
python diagnose_stages_qwen_v85.py --stage joint --data "$TILE_DATA" \
  --vlm-model "$VLM_MODEL" --device cuda:0 --load-4bit \
  --sample-index 0 --output results/stage_joint_r2
```

Each run prints the answer and saves `model_answer.txt`, `summary.json`, exact
`prompt.json`, `chat.txt`, and generation configuration. Generation stops at EOS
or budget, without repairing output. Use fresh output directories for reruns.

- `vqa`: manually check viewpoint, water/land locations and boundaries; no automated pass.
- `scene`: validates geometry/schema and reports duplicate regions, blocked fraction,
  unknown fraction, and structural sanity. `scene_raster.json` contains the estimated
  cost/blocked/coverage grids. Structural sanity is NOT visual correctness.
- `model`: uses an explicit, hand-constructed 3x5 scene saved as `fixture.json`.
  Costs are shown below; X is forbidden water. S=(1,0), G=(1,4), eight neighbors,
  no water corner cuts, cost includes both endpoints. The optimal cost is 5.

| Row / column | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| 0 | 1 | 1 | X | 1 | 1 |
| 1 | S:1 | 8 | 8 | 8 | G:1 |
| 2 | 1 | 1 | 1 | 1 | 1 |

  Qwen returns `{"model":{...}}`. The diagnostic attaches the **supplied scene**
  for compilation, never inserts missing model expressions. It solves the generated
  model and independently checks the decoded path, true cost, and objective value.
  `fixture_passed=true` means this one synthetic case passed, not that the model is
  correct for all maps. The usual formulation example remains in the prompt, so this
  measures guided executable modeling, not unaided discovery of a formulation.
  Artifacts are in `model_check.json`; these results never count as visual navigation SR.
- `joint`: uses exactly the formal shared modeling prompt, checks compilation and
  saves the estimated scene and symbolic model. It does not certify visual accuracy
  or replace the full independent evaluation.

Validation: 39 CPU tests passed, including supplied-fixture solve/cost validation,
rejection of missing model fields, detection of repeated whole-map blocks, shared
prompt wiring, and existing symbolic pipeline tests. Qwen GPU outputs have not been
validated here; run the commands with the local model and inspect the saved answers.

### Explicit terrain rules in all shared navigation prompts

The shared environment prompt now includes the complete PKMN cost legend and
explicitly distinguishes original finite water costs (lake=6, ocean=10) from the
current forbidden-water protocol. Plants, mud and wall retain finite costs;
absence of a drawn road and dense texture do not imply a blocked region.
Water's scene cost=25 is an unused placeholder, distinct from traversable wall=25
and the unknown-region cost=25 assumption. Synthetic text fixture facts override
the terrain lookup. No per-image labels are added to the prompt.

VQA now asks for boundaries and rule-based traversability rather than a search
for visible roads. Scene-only output instructions show the complete nested
`geometry` object; invalid generated objects are still rejected, not repaired.
No dataset preprocessing or solver rule changes are needed. Rerun `vqa` and
`scene` with fresh output directories (for example `stage_vqa_r4` and
`stage_scene_r4`), then the text model and joint stages as described above.
These prompt changes have not yet been validated with Qwen GPU inference.
