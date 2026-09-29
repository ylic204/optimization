"""Language agent that turns a planning instruction into a solver task.

Qwen does not generate executable Python.  It generates a small JSON object
whose fields have fixed mathematical meanings.  The deterministic solver in
``candidate_milp_solver_v83.py`` converts that object into a MILP.
"""

import argparse
import json
from dataclasses import asdict

import torch

from optimization_spec_v83 import OptimizationTask


SYSTEM_PROMPT = """You are a mathematical optimization modeling agent.
Map a natural-language path-planning request to a JSON task specification.
All metrics are normalized to [0,1] and lower is better.
Use exactly these objective keys:
collision, non_traversable, safety_risk, route_deviation,
lack_of_progress, path_length, discomfort, goal_error.

Return only one JSON object with this schema:
{
  "name": "short task name",
  "domain": "nuplan or pointnav",
  "weights": {all eight keys with non-negative numbers},
  "limits": {a subset of metric keys with upper bounds in [0,1]},
  "constraint_penalty": 100.0,
  "explanation": "one sentence"
}

Safety-critical objectives should put large weights on collision and
non_traversable.  A limit is a soft constraint implemented with a non-negative
slack variable and the stated penalty.
"""


class QwenOptimizationAgent:
    """Use the already loaded Qwen3-VL model as a text task agent."""

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    @torch.no_grad()
    def parse(self, task_text, domain):
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Domain: {domain}\nPlanning request: {task_text}",
                    }
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[prompt], padding=True, return_tensors="pt"
        ).to(self.model.device)
        generated = self.model.generate(
            **inputs,
            max_new_tokens=320,
            do_sample=False,
            use_cache=True,
        )
        new_tokens = generated[:, inputs["input_ids"].shape[1] :]
        answer = self.processor.batch_decode(
            new_tokens, skip_special_tokens=True
        )[0]
        payload = answer[answer.index("{") : answer.rindex("}") + 1]
        return OptimizationTask(**json.loads(payload))


def load_qwen(model_path, device, load_4bit):
    """Standalone loader used once to create a cached task JSON."""
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen3VLForConditionalGeneration,
    )

    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    kwargs = {"local_files_only": True, "torch_dtype": torch.bfloat16}
    if load_4bit:
        kwargs = {
            "local_files_only": True,
            "device_map": {"": int(device.split(":")[-1])},
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
        }
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, **kwargs)
    if not load_4bit:
        model.to(device)
    model.eval()
    return model, processor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--task-text", required=True)
    parser.add_argument("--domain", choices=("nuplan", "pointnav"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--load-4bit", action="store_true")
    args = parser.parse_args()

    model, processor = load_qwen(args.model, args.device, args.load_4bit)
    task = QwenOptimizationAgent(model, processor).parse(
        args.task_text, args.domain
    )
    task.save(args.output)
    print(json.dumps(asdict(task), indent=2))


if __name__ == "__main__":
    main()
