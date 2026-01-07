# track_planner_grid/core/loop_select.py
import numpy as np
from track_planner_grid.occgrid import cell_to_world, world_to_cell

def select_best_loop_through_start(G, start_xy, resolution, origin_xy,
                                   require_through_start=True,
                                   min_loop_length_m=20.0,
                                   edt=None):
    nodes = G["nodes"]
    edges = G["edges"]
    if len(edges) == 0:
        return None

    # build adjacency
    adj = {i: [] for i in nodes.keys()}
    for idx, e in enumerate(edges):
        adj[e["u"]].append((e["v"], idx))
        adj[e["v"]].append((e["u"], idx))

    # find closest node to start (in world)
    start_node = _closest_node(nodes, start_xy, resolution, origin_xy)

    # DFS cycle search
    best = None
    best_score = -1e18

    max_depth = 80  # safety
    def dfs(cur, target, path_nodes, path_edges, length_m):
        nonlocal best, best_score
        if len(path_nodes) > max_depth:
            return

        for (nxt, eidx) in adj[cur]:
            if len(path_nodes) >= 2 and nxt == path_nodes[-2]:
                # don't instantly go back
                continue

            e = edges[eidx]
            seg_len = (len(e["pixels"]) - 1) * resolution
            new_len = length_m + seg_len

            if nxt == target and len(path_nodes) > 2:
                # closed a cycle
                if new_len >= min_loop_length_m:
                    poly = _edges_to_world_polyline(path_edges + [eidx], edges, resolution, origin_xy)
                    score = _score_cycle(poly, edt, resolution, origin_xy)
                    if score > best_score:
                        best_score = score
                        best = poly
                continue

            if nxt in path_nodes:
                continue

            dfs(nxt, target, path_nodes + [nxt], path_edges + [eidx], new_len)

    # If require_through_start: force cycle starting/ending at start_node
    if require_through_start:
        dfs(start_node, start_node, [start_node], [], 0.0)
        return best

    # else: try cycles from multiple seeds (small graphs)
    for seed in list(nodes.keys())[:50]:
        dfs(seed, seed, [seed], [], 0.0)
    return best

def _closest_node(nodes, start_xy, resolution, origin_xy):
    best = None
    best_d2 = 1e18
    for i, nd in nodes.items():
        p = cell_to_world(nd["yx"], resolution, origin_xy)
        d2 = float(np.sum((p - start_xy)**2))
        if d2 < best_d2:
            best_d2 = d2
            best = i
    return best

def _edges_to_world_polyline(edge_indices, edges, resolution, origin_xy):
    # concatenate edge pixel polylines, trying to maintain consistent direction
    out = []
    last = None
    for eidx in edge_indices:
        pix = edges[eidx]["pixels"]
        # choose direction based on proximity to last
        if last is not None:
            p0 = cell_to_world(pix[0], resolution, origin_xy)
            p1 = cell_to_world(pix[-1], resolution, origin_xy)
            if np.linalg.norm(p0 - last) <= np.linalg.norm(p1 - last):
                seq = pix
            else:
                seq = list(reversed(pix))
        else:
            seq = pix

        for yx in seq:
            wp = cell_to_world(yx, resolution, origin_xy)
            out.append(wp)
        last = out[-1]
    # remove duplicates
    out2 = [out[0]]
    for p in out[1:]:
        if np.linalg.norm(p - out2[-1]) > 1e-6:
            out2.append(p)
    return np.array(out2, dtype=float)

def _score_cycle(poly_world, edt, resolution, origin_xy):
    # Higher is better
    length = float(np.sum(np.linalg.norm(poly_world[1:] - poly_world[:-1], axis=1)))
    if edt is None:
        return length

    # clearance sample
    clear = 0.0
    for p in poly_world[::5]:
        yx = world_to_cell(p, resolution, origin_xy)
        y, x = int(yx[0]), int(yx[1])
        if 0 <= y < edt.shape[0] and 0 <= x < edt.shape[1]:
            clear += float(edt[y, x])
    # prefer long + safe
    return 1.0 * length + 2.0 * clear
