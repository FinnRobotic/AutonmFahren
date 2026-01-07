# track_planner_grid/core/occgrid.py
import numpy as np

def occgrid_to_numpy(msg):
    w = msg.info.width
    h = msg.info.height
    res = msg.info.resolution
    ox = msg.info.origin.position.x
    oy = msg.info.origin.position.y
    grid = np.array(msg.data, dtype=np.int16).reshape((h, w))
    return grid, float(res), (float(ox), float(oy))

def to_binary_occupied(grid, occ_thresh=50, treat_unknown_as_occupied=True):
    occ = grid >= occ_thresh
    if treat_unknown_as_occupied:
        occ = occ | (grid < 0)
    return occ

def world_to_cell(xy, resolution, origin_xy):
    ox, oy = origin_xy
    x, y = xy
    cx = int(np.floor((x - ox) / resolution))
    cy = int(np.floor((y - oy) / resolution))
    return np.array([cy, cx], dtype=int)  # (y,x)

def cell_to_world(yx, resolution, origin_xy):
    oy, ox = origin_xy[1], origin_xy[0]
    y, x = int(yx[0]), int(yx[1])
    wx = ox + (x + 0.5) * resolution
    wy = oy + (y + 0.5) * resolution
    return np.array([wx, wy], dtype=float)
