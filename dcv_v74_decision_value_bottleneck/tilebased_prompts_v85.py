"""Public-only prompt variants for frozen modeling and controlled comparisons."""
import hashlib
from pathlib import Path
from tilebased_core_v85 import DATASETS, build_model_prompt

SYSTEM_FILE = Path(__file__).resolve().parent/'prompts/tilebased_vqa_system_v85.txt'


def modeling_system_prompt(variant, dataset):
    if variant == 'legacy':
        return ''
    if variant != 'navigation':
        raise ValueError(f'unknown modeling prompt variant: {variant}')
    if dataset != 'pkmn':
        raise ValueError('navigation environment prompt targets pkmn; use --modeling-prompt legacy for other datasets')
    prompt = SYSTEM_FILE.read_text(encoding='utf-8').strip()
    if not prompt:
        raise ValueError('empty navigation environment prompt')
    return prompt


def modeling_user_prompt(variant, task, dataset, source, target, indices, region_grid, legend):
    original = build_model_prompt(task,dataset,source,target,indices,region_grid,legend)
    if variant == 'legacy':
        return original
    if variant != 'navigation':
        raise ValueError(f'unknown modeling prompt variant: {variant}')
    n = DATASETS[dataset]['grid']
    side = DATASETS[dataset]['image_side']
    return ("Current stage: optimization modeling. Use the navigation environment in the system message. "
            "Focus on cost-relevant region boundaries, connections and crossings for the supplied S/G. "
            "Exact object naming matters only when it changes the public traversal cost. "
            "A high-cost region is not a forbidden region. Preserve spatial variation supported by the image; "
            "do not assume a uniform map merely because a direct route is feasible. "
            f"The full image is {side}x{side} pixels and covers {n}x{n} terrain cells "
            f"({side//n}x{side//n} pixels per cell). S/G are coordinates, not necessarily drawn markers. "
            "Retained region indices refer to original image positions; missing visual regions are unknown. "
            "Do not claim to have inspected missing regions. The schema requires a complete estimated cost grid, "
            "so fill unobserved cells with estimates using the allowed costs, without claiming certainty. "
            "Output only the requested JSON, with no VQA discussion, category inventory, or route.\n" + original)


def prompt_metadata(variant, dataset):
    system = modeling_system_prompt(variant,dataset)
    return dict(modeling_prompt_variant=variant,system_prompt=system,
                system_prompt_sha256=hashlib.sha256(system.encode()).hexdigest())
