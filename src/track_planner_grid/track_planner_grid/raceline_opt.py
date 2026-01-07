# track_planner_grid/core/raceline_opt.py
import numpy as np
from track_planner_grid.occgrid import world_to_cell

def _edt_at(edt, p, resolution, origin_xy):
    yx = world_to_cell(p, resolution, origin_xy)
    y, x = int(yx[0]), int(yx[1])
    if 0 <= y < edt.shape[0] and 0 <= x < edt.shape[1]:
        return float(edt[y, x])
    return 0.0

def _edt_grad(edt, p, resolution, origin_xy):
    # central differences in grid space
    yx = world_to_cell(p, resolution, origin_xy)
    y, x = int(yx[0]), int(yx[1])
    h, w = edt.shape
    if not (1 <= y < h-1 and 1 <= x < w-1):
        return np.zeros(2)
    dy = (edt[y+1, x] - edt[y-1, x]) / (2.0 * resolution)
    dx = (edt[y, x+1] - edt[y, x-1]) / (2.0 * resolution)
    # convert (dy,dx) in grid to world (x,y): dx affects world-x, dy affects world-y
    return np.array([dx, dy], dtype=float)

def optimize_raceline_phase1(center_xy, edt, resolution, origin_xy,
                             d_safe=0.35, iters=80, lam_smooth=1.0, lam_wall=5.0,
                             closed=True):
    xy = center_xy.copy()
    n = len(xy)
    if n < 5:
        return xy

    # step sizes (stable defaults)
    alpha_s = 0.15 * lam_smooth
    alpha_w = 0.10 * lam_wall

    for _ in range(iters):
        new_xy = xy.copy()

        for i in range(n):
            if not closed and (i == 0 or i == n-1):
                continue

            im1 = (i - 1) % n
            ip1 = (i + 1) % n

            # Laplacian smooth
            smooth_force = (xy[im1] + xy[ip1] - 2.0 * xy[i])

            # Wall repulsion if too close
            d = _edt_at(edt, xy[i], resolution, origin_xy)
            if d < d_safe:
                g = _edt_grad(edt, xy[i], resolution, origin_xy)
                # push in direction of increasing distance
                wall_force = (d_safe - d) * g
            else:
                wall_force = 0.0

            new_xy[i] = xy[i] + alpha_s * smooth_force + alpha_w * wall_force

        xy = new_xy

    return xy
