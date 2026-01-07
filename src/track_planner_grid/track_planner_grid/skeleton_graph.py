# track_planner_grid/core/skeleton_graph.py
import numpy as np
from skimage.morphology import skeletonize

_NEI8 = [(dy,dx) for dy in (-1,0,1) for dx in (-1,0,1) if not (dy==0 and dx==0)]

def skeletonize_free(free_mask):
    # free_mask: bool, True=free
    skel = skeletonize(free_mask.astype(np.uint8) > 0)
    return skel.astype(bool)

def _neighbors(yx, skel):
    y, x = yx
    h, w = skel.shape
    out = []
    for dy, dx in _NEI8:
        ny, nx = y+dy, x+dx
        if 0 <= ny < h and 0 <= nx < w and skel[ny, nx]:
            out.append((ny, nx))
    return out

def build_skeleton_graph(skel):
    pts = np.argwhere(skel)  # (N,2) y,x
    skel_set = set(map(tuple, pts))

    deg = {}
    for yx in skel_set:
        deg[yx] = len([nb for nb in _neighbors(yx, skel) if nb in skel_set])

    # nodes: deg != 2
    node_pixels = [yx for yx, d in deg.items() if d != 2]
    if len(node_pixels) == 0:
        # Sonderfall: perfekter Ring ohne Junction/Enden -> mach beliebige Knoten
        node_pixels = [tuple(pts[0])]

    node_id = {yx:i for i,yx in enumerate(node_pixels)}
    nodes = {i: {"yx": yx} for yx,i in node_id.items()}

    visited_segment = set()  # mark pixel-to-pixel traversals: (a,b) und (b,a)

    edges = []

    for yx in node_pixels:
        u = node_id[yx]
        for nb in _neighbors(yx, skel):
            if nb not in skel_set:
                continue
            a = yx
            b = nb
            key = (a,b)
            if key in visited_segment:
                continue

            # trace until next node
            pixels = [a, b]
            visited_segment.add((a,b)); visited_segment.add((b,a))

            prev = a
            cur = b
            while True:
                if cur in node_id and cur != yx:
                    v = node_id[cur]
                    edges.append({"u": u, "v": v, "pixels": pixels})
                    break

                nbs = [p for p in _neighbors(cur, skel) if p in skel_set and p != prev]
                if len(nbs) == 0:
                    # dead end reached (cur should be node usually, but handle anyway)
                    if cur not in node_id:
                        # create node for safety
                        v = len(nodes)
                        node_id[cur] = v
                        nodes[v] = {"yx": cur}
                    else:
                        v = node_id[cur]
                    edges.append({"u": u, "v": v, "pixels": pixels})
                    break

                # if junction-like ambiguity inside deg==2 area, pick one deterministically (rare)
                nxt = nbs[0] if len(nbs) == 1 else _choose_straight(prev, cur, nbs)
                if (cur, nxt) in visited_segment:
                    # already traversed; stop to avoid loops
                    if cur not in node_id:
                        v = len(nodes)
                        node_id[cur] = v
                        nodes[v] = {"yx": cur}
                    else:
                        v = node_id[cur]
                    edges.append({"u": u, "v": v, "pixels": pixels})
                    break

                pixels.append(nxt)
                visited_segment.add((cur, nxt)); visited_segment.add((nxt, cur))
                prev, cur = cur, nxt

    return {"nodes": nodes, "edges": edges}

def _choose_straight(prev, cur, candidates):
    # prefer minimal turning angle
    pv = np.array([cur[0]-prev[0], cur[1]-prev[1]], dtype=float)
    pv_norm = np.linalg.norm(pv) + 1e-9
    pv /= pv_norm
    best = None
    best_dot = -1e9
    for c in candidates:
        cv = np.array([c[0]-cur[0], c[1]-cur[1]], dtype=float)
        cv /= (np.linalg.norm(cv) + 1e-9)
        d = float(np.dot(pv, cv))
        if d > best_dot:
            best_dot = d
            best = c
    return best

def prune_spurs(G, min_len_m, resolution):
    # Remove edges that end in degree-1 nodes and are shorter than threshold, iteratively.
    # Simple but effective.
    nodes = G["nodes"]
    edges = G["edges"]

    changed = True
    while changed:
        changed = False
        deg = {i:0 for i in nodes.keys()}
        for e in edges:
            deg[e["u"]] += 1
            deg[e["v"]] += 1

        new_edges = []
        removed_nodes = set()
        for e in edges:
            length_m = (len(e["pixels"]) - 1) * resolution
            u, v = e["u"], e["v"]
            if length_m < min_len_m and (deg[u] == 1 or deg[v] == 1):
                removed_nodes.add(u if deg[u] == 1 else v)
                changed = True
            else:
                new_edges.append(e)

        # keep nodes; removing nodes fully is optional; we just keep them harmlessly
        edges = new_edges

    G["edges"] = edges
    return G
