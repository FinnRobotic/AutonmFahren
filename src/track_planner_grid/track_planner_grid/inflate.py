# track_planner_grid/core/inflate.py
import numpy as np
from scipy.ndimage import binary_dilation

def inflate(occ, r_cells):
    if r_cells <= 0:
        return occ.copy()
    yy, xx = np.ogrid[-r_cells:r_cells+1, -r_cells:r_cells+1]
    se = (xx*xx + yy*yy) <= (r_cells*r_cells)
    return binary_dilation(occ, structure=se)