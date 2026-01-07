# track_planner_grid/core/spline.py
import numpy as np
from scipy.interpolate import splprep, splev

def _arc_length(xy):
    d = np.linalg.norm(xy[1:] - xy[:-1], axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    return s

def fit_and_resample_closed(xy, ds=0.1):
    # ensure periodic by repeating start
    if np.linalg.norm(xy[0] - xy[-1]) > 1e-6:
        xy = np.vstack([xy, xy[0]])
    s = _arc_length(xy)
    L = float(s[-1])
    if L < 1e-6:
        return xy.copy()

    u = s / L
    tck, _ = splprep([xy[:,0], xy[:,1]], u=u, s=0.5, per=True)
    n = max(50, int(np.ceil(L / ds)))
    uu = np.linspace(0.0, 1.0, n, endpoint=False)
    x, y = splev(uu, tck)
    return np.stack([x, y], axis=1)

def fit_and_resample_open(xy, ds=0.1):
    s = _arc_length(xy)
    L = float(s[-1])
    if L < 1e-6:
        return xy.copy()
    u = s / L
    tck, _ = splprep([xy[:,0], xy[:,1]], u=u, s=0.5, per=False)
    n = max(50, int(np.ceil(L / ds)) + 1)
    uu = np.linspace(0.0, 1.0, n, endpoint=True)
    x, y = splev(uu, tck)
    return np.stack([x, y], axis=1)
