"""Export the exact software, CUDA and Qwen3-VL checkpoint environment.

Run this script inside the same activated Python/Conda environment used for
training.  It does not install or modify packages.
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


TRACKED_DISTRIBUTIONS = (
    "torch",
    "torchvision",
    "transformers",
    "accelerate",
    "bitsandbytes",
    "safetensors",
    "tokenizers",
    "numpy",
    "Pillow",
    "tqdm",
    "pytest",
    "sentence-transformers",
    "flash-attn",
)

MODEL_METADATA_FILES = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer_config.json",
    "chat_template.json",
    "model.safetensors.index.json",
)


def distribution_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def run_command(command):
    executable = shutil.which(command[0])
    if executable is None:
        return {"available": False, "command": command, "output": None}
    completed = subprocess.run(
        [executable, *command[1:]],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "available": True,
        "command": command,
        "returncode": completed.returncode,
        "output": completed.stdout.strip(),
    }


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {"_read_error": str(exc)}


def snapshot_revision(model_path):
    parts = model_path.resolve().parts
    if "snapshots" in parts:
        index = parts.index("snapshots")
        if index + 1 < len(parts):
            return parts[index + 1]
    refs_main = model_path / "refs" / "main"
    if refs_main.is_file():
        return refs_main.read_text(encoding="utf-8").strip()
    return None


def model_report(model_path):
    if model_path is None:
        return {"provided": False}
    path = Path(model_path).expanduser().resolve()
    report = {
        "provided": True,
        "path": str(path),
        "exists": path.exists(),
        "snapshot_revision": None,
        "files": {},
    }
    if not path.is_dir():
        return report

    report["snapshot_revision"] = snapshot_revision(path)
    for filename in MODEL_METADATA_FILES:
        file_path = path / filename
        if not file_path.is_file():
            continue
        entry = {
            "size_bytes": file_path.stat().st_size,
            "sha256": sha256(file_path),
        }
        if file_path.suffix == ".json":
            entry["content"] = read_json(file_path)
        report["files"][filename] = entry

    config = report["files"].get("config.json", {}).get("content", {})
    vision = config.get("vision_config", {}) if isinstance(config, dict) else {}
    text = config.get("text_config", {}) if isinstance(config, dict) else {}
    report["identity"] = {
        "name_or_path": config.get("_name_or_path") or config.get("name_or_path"),
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "config_transformers_version": config.get("transformers_version"),
        "vision_hidden_size": vision.get("hidden_size"),
        "vision_patch_size": vision.get("patch_size"),
        "vision_spatial_merge_size": vision.get("spatial_merge_size"),
        "text_hidden_size": text.get("hidden_size"),
        "text_num_hidden_layers": text.get("num_hidden_layers"),
    }

    index = report["files"].get(
        "model.safetensors.index.json", {}
    ).get("content", {})
    if isinstance(index, dict):
        weight_map = index.get("weight_map", {})
        metadata = index.get("metadata", {})
        report["weights"] = {
            "tensor_count": len(weight_map) if isinstance(weight_map, dict) else None,
            "shards": sorted(set(weight_map.values()))
            if isinstance(weight_map, dict)
            else None,
            "total_size_bytes": metadata.get("total_size")
            if isinstance(metadata, dict)
            else None,
        }
    return report


def torch_report():
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        return {"importable": False, "error": repr(exc)}

    report = {
        "importable": True,
        "version": getattr(torch, "__version__", None),
        "module_path": getattr(torch, "__file__", None),
        "compiled_cuda": getattr(torch.version, "cuda", None),
        "compiled_git_version": getattr(torch.version, "git_version", None),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
        "devices": [],
    }
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            with torch.cuda.device(index):
                bfloat16_supported = bool(torch.cuda.is_bf16_supported())
            report["devices"].append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "compute_capability": [properties.major, properties.minor],
                    "bfloat16_supported": bfloat16_supported,
                }
            )
    try:
        from torch.utils.collect_env import get_pretty_env_info

        report["collect_env"] = get_pretty_env_info()
    except Exception as exc:
        report["collect_env_error"] = repr(exc)
    return report


def transformers_report():
    try:
        transformers = importlib.import_module("transformers")
    except Exception as exc:
        return {"importable": False, "error": repr(exc)}
    names = (
        "AutoProcessor",
        "BitsAndBytesConfig",
        "Qwen3VLForConditionalGeneration",
    )
    return {
        "importable": True,
        "version": getattr(transformers, "__version__", None),
        "module_path": getattr(transformers, "__file__", None),
        "qwen3vl_symbols": {
            name: hasattr(transformers, name) for name in names
        },
    }


def write_command_output(path, result):
    if result.get("available"):
        path.write_text(result.get("output") or "", encoding="utf-8")


def write_portable_conda_yaml(path, result):
    if not result.get("available"):
        return
    lines = (result.get("output") or "").splitlines()
    portable = "\n".join(
        line for line in lines if not line.startswith("prefix:")
    )
    path.write_text(portable + ("\n" if portable else ""), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default=os.getenv("VLM_MODEL"),
        help="Local Qwen3-VL checkpoint directory; defaults to VLM_MODEL.",
    )
    parser.add_argument(
        "--out",
        default="environment_qwen3vl_v81",
        help="Output directory for the exported environment files.",
    )
    args = parser.parse_args()

    output = Path(args.out).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    pip_freeze = run_command([sys.executable, "-m", "pip", "freeze"])
    # run_command resolves only command[0]; sys.executable is already absolute.
    if not pip_freeze["available"]:
        pip_freeze = {
            "available": True,
            "command": [sys.executable, "-m", "pip", "freeze"],
            "returncode": 1,
            "output": "Unable to execute pip freeze",
        }

    conda_export = run_command(["conda", "env", "export", "--no-builds"])
    conda_explicit = run_command(["conda", "list", "--explicit"])
    nvidia_smi = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader",
        ]
    )
    nvcc = run_command(["nvcc", "--version"])

    report = {
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "system": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "python_prefix": sys.prefix,
            "conda_default_env": os.getenv("CONDA_DEFAULT_ENV"),
            "conda_prefix": os.getenv("CONDA_PREFIX"),
        },
        "distributions": {
            name: distribution_version(name) for name in TRACKED_DISTRIBUTIONS
        },
        "torch": torch_report(),
        "transformers": transformers_report(),
        "qwen3vl_checkpoint": model_report(args.model_path),
        "nvidia_smi": nvidia_smi,
        "nvcc": nvcc,
    }

    (output / "environment_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_command_output(output / "requirements_lock.txt", pip_freeze)
    write_portable_conda_yaml(output / "conda_environment.yml", conda_export)
    write_command_output(output / "conda_explicit.txt", conda_explicit)
    write_command_output(output / "nvidia_smi.txt", nvidia_smi)
    write_command_output(output / "nvcc.txt", nvcc)

    summary = {
        "python": report["system"]["python_version"],
        "torch": report["torch"].get("version"),
        "torch_compiled_cuda": report["torch"].get("compiled_cuda"),
        "cuda_available": report["torch"].get("cuda_available"),
        "transformers": report["transformers"].get("version"),
        "accelerate": report["distributions"].get("accelerate"),
        "bitsandbytes": report["distributions"].get("bitsandbytes"),
        "model": report["qwen3vl_checkpoint"].get("identity"),
        "model_revision": report["qwen3vl_checkpoint"].get(
            "snapshot_revision"
        ),
        "output": str(output),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
