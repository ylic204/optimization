# Qwen3-VL V8.1 environment export

Run the exporter inside the same activated environment and on the same server
used for training.  This is important because `torch`, its bundled CUDA
runtime, the NVIDIA driver and `bitsandbytes` must be recorded together.

```bash
conda activate dl
cd /home/lyi/neurodynamic/dcv_v74_decision_value_bottleneck

python export_qwen3vl_environment.py \
  --model-path /data/lyi/models/Qwen3-VL-8B-Instruct \
  --out environment_qwen3vl_v81
```

Equivalently, the model path may be supplied through the existing variable:

```bash
VLM_MODEL=/data/lyi/models/Qwen3-VL-8B-Instruct \
python export_qwen3vl_environment.py --out environment_qwen3vl_v81
```

The output directory contains:

| File | Contents |
|---|---|
| `environment_report.json` | Python, Torch, CUDA, cuDNN, GPU, Transformers and Qwen checkpoint metadata |
| `requirements_lock.txt` | Exact `pip freeze` output |
| `conda_environment.yml` | Conda environment without platform-specific build strings |
| `conda_explicit.txt` | Exact Conda package URLs for same-platform reproduction |
| `nvidia_smi.txt` | GPU model, driver and memory |
| `nvcc.txt` | System CUDA toolkit version, when installed |

## Direct runtime dependencies

The current Qwen3-VL selector uses:

```text
torch
numpy
pillow
tqdm
pytest>=8.0
transformers>=4.57,<5.18
accelerate>=1.0
bitsandbytes>=0.45        # Linux; needed only for --load-4bit
```

`Qwen3-VL-8B-Instruct` is a model checkpoint, not a Python package.  Its
identity is exported from `config.json`, the Hugging Face snapshot revision
when available, and SHA-256 hashes of the configuration/index files.

The current code uses `attn_implementation="sdpa"`, so `flash-attn` is not a
required dependency.  `sentence-transformers` is also not used by the V7.9 or
V8.1 Qwen selector and should not be added merely for task-text encoding.

Do not select a Torch wheel only from the version printed by `nvcc`.  Use the
PyTorch installation selector for a wheel supported by the installed NVIDIA
driver, then preserve the working result with this exporter.
