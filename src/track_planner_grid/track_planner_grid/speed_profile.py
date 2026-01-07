# track_planner_grid/core/speed_profile.py
import numpy as np

def _curvature(xy, closed=True):
    n = len(xy)
    k = np.zeros(n, dtype=float)
    for i in range(n):
        if not closed and (i == 0 or i == n-1):
            continue
        im1 = (i-1) % n
        ip1 = (i+1) % n
        p0, p1, p2 = xy[im1], xy[i], xy[ip1]
        a = p1 - p0
        b = p2 - p1
        la = np.linalg.norm(a) + 1e-9
        lb = np.linalg.norm(b) + 1e-9
        # angle change over arc length approx
        cosang = np.clip(np.dot(a, b) / (la*lb), -1.0, 1.0)
        ang = np.arccos(cosang)
        ds = 0.5 * (la + lb)
        k[i] = ang / (ds + 1e-9)
    return k

def compute_speed_profile(xy, ds, a_lat_max=7.0, a_long_max=3.0, v_max=12.0, closed=True):
    k = _curvature(xy, closed=closed)
    # lateral limit
    v_lat = np.sqrt(np.maximum(0.0, a_lat_max / (np.abs(k) + 1e-6)))
    v = np.minimum(v_lat, v_max)

    # forward pass (acceleration)
    for i in range(1, len(v)):
        v[i] = min(v[i], np.sqrt(v[i-1]**2 + 2.0*a_long_max*ds))

    # backward pass (braking)
    for i in range(len(v)-2, -1, -1):
        v[i] = min(v[i], np.sqrt(v[i+1]**2 + 2.0*a_long_max*ds))

    return v
