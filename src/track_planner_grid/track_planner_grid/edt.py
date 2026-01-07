# track_planner_grid/core/edt.py
from scipy.ndimage import distance_transform_edt

def compute_edt(occ_inflated, resolution):
    free = ~occ_inflated
    return distance_transform_edt(free) * resolution