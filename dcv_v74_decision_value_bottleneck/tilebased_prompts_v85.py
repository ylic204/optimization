"""Public-only prompt variants for frozen modeling and controlled comparisons."""
import hashlib
from pathlib import Path
from tilebased_core_v85 import DATASETS, build_model_prompt
from tilebased_corridor_v85 import CORRIDOR_SCHEMA, MAX_NODES, MAX_EDGES
from tilebased_segments_v85 import SEGMENT_SCHEMA
from tilebased_regions_v85 import regions_user_prompt

SYSTEM_FILE = Path(__file__).resolve().parent/'prompts/tilebased_vqa_system_v85.txt'


def modeling_system_prompt(variant, dataset, model_representation="matrix"):
    if variant == 'legacy':
        return ''
    if variant != 'navigation':
        raise ValueError(f'unknown modeling prompt variant: {variant}')
    if dataset != 'pkmn':
        raise ValueError('navigation environment prompt targets pkmn; use --modeling-prompt legacy for other datasets')
    system_file = SYSTEM_FILE.with_name('tilebased_land_system_v85.txt') if model_representation=='regions' else SYSTEM_FILE
    prompt = system_file.read_text(encoding='utf-8').strip()
    if not prompt:
        raise ValueError('empty navigation environment prompt')
    return prompt


def modeling_user_prompt(variant, task, dataset, source, target, indices, region_grid, legend,
                         model_representation='matrix'):
    if model_representation == 'regions':
        if variant not in ('legacy','navigation'):raise ValueError('unknown prompt variant')
        return regions_user_prompt(task,dataset,source,target,indices,region_grid,legend)
    if model_representation == 'segments':
        if variant not in ('legacy','navigation'):
            raise ValueError('unknown prompt variant')
        return segments_user_prompt(task,dataset,source,target,indices,region_grid,legend)
    if model_representation == 'corridor':
        if variant not in ('legacy','navigation'):
            raise ValueError('unknown prompt variant')
        return corridor_user_prompt(task,dataset,source,target,indices,region_grid,legend)
    if model_representation != 'matrix':
        raise ValueError('unknown model representation')
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


def prompt_metadata(variant, dataset, model_representation='matrix'):
    system = modeling_system_prompt(variant,dataset,model_representation)
    return dict(modeling_prompt_variant=variant,model_representation=model_representation,system_prompt=system,
                system_prompt_sha256=hashlib.sha256(system.encode()).hexdigest())


def corridor_user_prompt(task, dataset, source, target, indices, region_grid, legend):
    n=DATASETS[dataset]['grid'];side=DATASETS[dataset]['image_side']
    low,high=min(DATASETS[dataset]['cost_values']),max(DATASETS[dataset]['cost_values'])
    return (f"{task}\nCurrent stage: construct an S-to-G corridor graph for a minimum-cost path solver. "
        "Model route-relevant connections and cost differences, including plausible detours when the image supports them. "
        "Do not reconstruct the entire terrain map and do not choose a final route. The solver will choose among your connections. "
        "Identify waypoint coordinates at turns, junctions and cost-changing boundaries. "
        "Preserve alternative connections when they could change the decision; do not invent alternatives just to meet a quota. "
        f"The image is {side}x{side} pixels covering {n}x{n} terrain cells. "
        "Coordinates are zero-based [row,column], row down, column right. S/G need not have drawn markers. "
        f"Public terrain-cost rules: {legend}\n"
        "All finite-cost terrain remains traversable. An omitted connection means unmodeled search space, not a physical obstacle. "
        f"Retained visual regions on the original {region_grid}x{region_grid} row-major grid: {list(map(int,indices))}. "
        "These visual regions are not terrain cells. Use their original positions; missing regions have unknown appearance. "
        "Estimate costs only for proposed connections and S; no costs for unrelated terrain are required. "
        "Do not claim to see unobserved terrain. Exact object names are unnecessary unless they change the public cost.\n"
        f"Return exactly one JSON object with six keys: schema='{CORRIDOR_SCHEMA}', "
        f"source_rc={list(map(int,source))}, target_rc={list(map(int,target))}, source_cost, nodes_rc, edges. "
        "Use double-quoted JSON keys and strings. source_cost is your estimated cost of S. "
        "nodes_rc is a list of unique integer coordinates: node 0 is S, node 1 is G, followed by any useful waypoints. "
        f"Use at most {MAX_NODES} waypoints and {MAX_EDGES} edges. "
        "Each edge is [from_node_index,to_node_index,mean_arrival_cost]. Edges are DIRECTED. "
        "An edge represents a straight segment between its two cell centers. The compiler samples max(abs(dr),abs(dc)) "
        "eight-neighbor steps along it using nearest-integer interpolation (half upward, canonical endpoint order). "
        "Use extra waypoints for bends or changes of terrain cost. mean_arrival_cost estimates the average cost of cells "
        "entered on that segment, excluding its departure cell and including its arrival cell. "
        f"It may be any finite number in [{low},{high}], since a segment can cross mixed terrain; source_cost uses the same range. "
        "The solver adds source_cost once, then one estimated arrival cost per step; diagonals have no multiplier. "
        "Segments crossing at a sampled cell connect; overlapping directed steps average their estimates. "
        "No off-graph connections or hidden ground-truth routing will be added. Supply a connected S-to-G graph. "
        "If S equals G, use one node and an empty edge list. Output JSON only: no node_costs matrix, no path_rc, "
        "no executable code and no extra keys.")


def segments_user_prompt(task, dataset, source, target, indices, region_grid, legend):
    n = DATASETS[dataset]['grid']; side = DATASETS[dataset]['image_side']
    low,high = min(DATASETS[dataset]['cost_values']),max(DATASETS[dataset]['cost_values'])
    chosen = sorted(map(int,indices))
    regions = ('All visual regions are retained.' if chosen == list(range(region_grid**2)) else
               f'Retained regions on the original {region_grid}x{region_grid} row-major visual grid: {chosen}. '
               'These are visual regions, not terrain cells. Missing appearance is unknown; use estimates without claiming observation.')
    return (f'{task}\nThe RGB image is {side}x{side}; each terrain cell is {side//n}x{side//n} pixels. '
        f'S/G are supplied coordinates and may not be drawn. {regions}\n'
        f'Terrain costs: {legend}\n'
        'Build a small spatial connection graph for the minimum-cost solver. '
        'Include useful turns and alternatives supported by the image, especially detours around expensive terrain. '
        'Use long straight connections within similar terrain; split at turns or cost changes, not at every cell. '
        'Use original [row,column] coordinates directly, never node IDs. The solver will choose the route. '
        'Do not reconstruct a full cost matrix.\n'
        f'Return one JSON object: schema="{SEGMENT_SCHEMA}", source_rc={list(map(int,source))}, '
        f'target_rc={list(map(int,target))}, source_cost=<estimated start-cell cost>, segments=[...]. '
        'Each segment is {"from_rc":[row,column],"to_rc":[row,column],"mean_cost":number}. '
        'It connects its two cell centers by straight eight-neighbor steps, in the stated direction. '
        'mean_cost is the average cost per entered cell on that connection, excluding its departure cell. '
        f'Cost estimates must be finite numbers in [{low},{high}]. The solver counts the start cost once. '
        'Connect turns by using exactly the same endpoint coordinates. Keep the cost differences visible in the image; '
        'do not assign one default cost to every connection. '
        'Use an empty segments list only when S=G or you cannot propose any connection; an unconnected S/G graph fails. '
        f'At most {MAX_EDGES} segments; do not repeat a directed connection. '
        'End the JSON immediately after your finite connection list. No explanation, node list, or numbered edges.')
