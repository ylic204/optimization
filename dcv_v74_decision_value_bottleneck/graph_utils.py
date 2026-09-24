import heapq
import numpy as np


def fixed_layered_graph(rng):
    """
    V4 DAG:
      node 0       : source
      nodes 1..3   : layer 1
      nodes 4..6   : layer 2
      nodes 7..9   : layer 3
      nodes 10..12 : layer 4
      node 13      : goal

    Edges = 3 + 9 + 9 + 9 + 3 = 33
    Paths = 3^4 = 81
    """
    layers = [
        [0],
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
        [13],
    ]

    edges = []
    for left, right in zip(layers[:-1], layers[1:]):
        for u in left:
            for v in right:
                edges.append((u, v))

    edge_to_idx = {e: i for i, e in enumerate(edges)}

    paths = []
    for a in layers[1]:
        for b in layers[2]:
            for c in layers[3]:
                for d in layers[4]:
                    paths.append([
                        edge_to_idx[(0, a)],
                        edge_to_idx[(a, b)],
                        edge_to_idx[(b, c)],
                        edge_to_idx[(c, d)],
                        edge_to_idx[(d, 13)],
                    ])

    path_mask = np.zeros((81, 33), dtype=np.float32)
    for i, p in enumerate(paths):
        path_mask[i, p] = 1.0

    # Keep geometry deliberately close so semantic/risk evidence matters.
    base_cost = rng.uniform(0.98, 1.02, size=33).astype(np.float32)

    return np.asarray(edges, dtype=np.int64), base_cost, path_mask


def edge_graph_features(base_cost, path_mask):
    path_freq = path_mask.mean(axis=0).astype(np.float32)
    geom_path_cost = path_mask @ base_cost
    geo_best = int(np.argmin(geom_path_cost))
    on_geo_best = path_mask[geo_best].astype(np.float32)

    return np.stack([
        base_cost,
        path_freq,
        on_geo_best,
    ], axis=-1).astype(np.float32)


def costs_from_states(base_cost, states, cfg):
    risk = np.asarray([
        cfg.risk_normal,
        cfg.risk_rough,
        cfg.risk_hazard,
        0.0,
    ], dtype=np.float32)

    c = base_cost + risk[states]
    blocked = states == 3
    c = c + blocked.astype(np.float32) * cfg.blocked_penalty

    return c.astype(np.float32), (~blocked).astype(np.float32)


def choose_path(path_mask, edge_cost):
    pc = path_mask @ edge_cost
    idx = int(np.argmin(pc))
    return idx, float(pc[idx]), pc.astype(np.float32)


def dijkstra_exact(edges, edge_cost, source=0, goal=None):
    if goal is None:
        goal = int(edges.max())

    n_nodes = int(max(edges.max(), goal)) + 1
    adj = [[] for _ in range(n_nodes)]

    for ei, (u, v) in enumerate(edges):
        adj[int(u)].append((int(v), float(edge_cost[ei]), ei))

    dist = [float("inf")] * n_nodes
    prev_node = [-1] * n_nodes
    prev_edge = [-1] * n_nodes

    dist[source] = 0.0
    pq = [(0.0, source)]

    while pq:
        d, u = heapq.heappop(pq)

        if d != dist[u]:
            continue

        if u == goal:
            break

        for v, w, ei in adj[u]:
            nd = d + w

            if nd < dist[v]:
                dist[v] = nd
                prev_node[v] = u
                prev_edge[v] = ei
                heapq.heappush(pq, (nd, v))

    if not np.isfinite(dist[goal]):
        raise RuntimeError("No path found by Dijkstra.")

    path_edges = []
    cur = goal

    while cur != source:
        ei = prev_edge[cur]

        if ei < 0:
            raise RuntimeError("Broken predecessor chain.")

        path_edges.append(ei)
        cur = prev_node[cur]

    path_edges.reverse()

    return np.asarray(path_edges, dtype=np.int64), float(dist[goal])


def path_edges_to_mask(path_edges, n_edges):
    m = np.zeros(n_edges, dtype=np.float32)
    m[path_edges] = 1.0
    return m
