"""Coordinate connections: no generated node IDs, no GT-assisted completion.

The streaming guard stops completed JSON or provably invalid completed segments.
It does NOT constrain logits or repair incomplete answers into valid models.
"""
import json
import re
from tilebased_core_v85 import DATASETS, coordinate, parse_model_json
from tilebased_corridor_v85 import _cost, MAX_EDGES

SEGMENT_SCHEMA = 'tilebased_sg_segments_v2'


def json_coordinate(value, shape):
    # Validate before NumPy conversion: [True, 0] would otherwise become [1, 0].
    if not isinstance(value, list) or len(value) != 2 or any(type(v) is not int for v in value):
        raise ValueError('coordinates must be two JSON integers, not booleans')
    return coordinate(value, shape)


def validate_segment(segment, shape, low, high):
    if not isinstance(segment, dict) or set(segment) != {'from_rc', 'to_rc', 'mean_cost'}:
        raise ValueError('each segment requires exactly from_rc, to_rc, mean_cost')
    a = json_coordinate(segment['from_rc'], shape)
    b = json_coordinate(segment['to_rc'], shape)
    if a == b:
        raise ValueError('zero-length segment')
    return a, b, _cost(segment['mean_cost'], low, high)


def validate_segments(answer, dataset, source, target):
    obj = parse_model_json(answer)
    if not isinstance(obj, dict) or set(obj) != {'schema','source_rc','target_rc','source_cost','segments'} or obj['schema'] != SEGMENT_SCHEMA:
        raise ValueError('expected coordinate-segment schema and exactly its five keys')
    n = DATASETS[dataset]['grid']; shape = (n, n)
    source, target = coordinate(source, shape), coordinate(target, shape)
    if json_coordinate(obj['source_rc'], shape) != source or json_coordinate(obj['target_rc'], shape) != target:
        raise ValueError('segment endpoints mismatch')
    low, high = min(DATASETS[dataset]['cost_values']), max(DATASETS[dataset]['cost_values'])
    source_cost = _cost(obj['source_cost'], low, high)
    segments = obj['segments']
    if not isinstance(segments, list) or len(segments) > MAX_EDGES:
        raise ValueError(f'segments must be a list of at most {MAX_EDGES} connections')
    if source == target and segments:
        raise ValueError('S=G requires no segments')
    nodes = [source] if source == target else [source, target]
    indices = {point:i for i,point in enumerate(nodes)}
    edges, seen = [], set()
    for segment in segments:
        a, b, cost = validate_segment(segment, shape, low, high)
        if (a, b) in seen:
            raise ValueError('duplicate directed segment')
        seen.add((a, b))
        for point in (a, b):
            if point not in indices:
                indices[point] = len(nodes); nodes.append(point)
        edges.append((indices[a], indices[b], cost))
    # The existing compiler receives internally assigned IDs only. Never add
    # a missing connection, reverse a directed segment, or consult scene costs.
    return dict(nodes=nodes, edges=edges, source=source, target=target,
                source_cost=source_cost, shape=shape)


class SegmentOutputGuard:
    """Stateless check of the complete decoded prefix, safe for BPE redecoding."""
    def __init__(self, dataset):
        n = DATASETS[dataset]['grid']; self.shape = (n,n)
        self.low, self.high = min(DATASETS[dataset]['cost_values']), max(DATASETS[dataset]['cost_values'])

    def __call__(self, text):
        # Preserve original bytes. A closed but malformed semantic model still
        # fails final validation; no JSON completion/repair is performed here.
        try:
            obj = parse_model_json(text)
        except (ValueError, TypeError):
            obj = None
        if isinstance(obj, dict):
            return 'json_complete'
        # Parse only fully emitted list items. Partial entries are not failures.
        match = re.search(r'"segments"\s*:\s*\[', text)
        if match is None:
            return None
        position, seen, count = match.end(), set(), 0
        decoder = json.JSONDecoder()
        while True:
            while position < len(text) and text[position].isspace():
                position += 1
            if position >= len(text) or text[position] == ']':
                return None
            if count >= MAX_EDGES:
                return 'segment_limit'
            try:
                segment, end = decoder.raw_decode(text, position)
            except ValueError:
                return None
            try:
                a, b, _ = validate_segment(segment, self.shape, self.low, self.high)
            except (ValueError, TypeError, OverflowError):
                return 'invalid_segment'
            if (a, b) in seen:
                return 'repeated_segment'
            seen.add((a, b)); count += 1; position = end
            while position < len(text) and text[position].isspace():
                position += 1
            if position >= len(text) or text[position] == ']':
                return None
            if text[position] != ',':
                return 'invalid_segment_separator'
            position += 1
