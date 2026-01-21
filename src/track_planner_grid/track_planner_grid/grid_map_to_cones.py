import cv2
import numpy as np
import yaml
from scipy.spatial import KDTree

# =========================
# PARAMETER
# =========================
PGM_FILE = "/home/nvidia/theta_ws/maps/track_map3.pgm"
YAML_FILE = "/home/nvidia/theta_ws/maps/track_map3.yaml"
OUTPUT_FILE = "/home/nvidia/theta_ws/src/track_planner_grid/cones/cones.csv"

OCCUPIED_THRESHOLD = 50      # Pixel < 50 = belegt
CONE_DISTANCE_M = 0.2        # Mindestabstand zwischen Cones (Meter)

# =========================
# MAP LADEN
# =========================
img = cv2.imread(PGM_FILE, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise RuntimeError("PGM konnte nicht geladen werden")

height, width = img.shape

with open(YAML_FILE, "r") as f:
    map_info = yaml.safe_load(f)

resolution = map_info["resolution"]
origin_x, origin_y, _ = map_info["origin"]

# =========================
# BINÄRISIERUNG
# =========================
occupied = img < OCCUPIED_THRESHOLD
free = img >= OCCUPIED_THRESHOLD

# =========================
# RANDZELLEN FINDEN
# =========================
def neighbors_8(x, y):
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                yield nx, ny

boundary_cells = []

for y in range(height):
    for x in range(width):
        if not occupied[y, x]:
            continue
        for nx, ny in neighbors_8(x, y):
            if free[ny, nx]:
                boundary_cells.append((x, y))
                break

boundary_cells = np.array(boundary_cells)
print(f"Randzellen gefunden: {len(boundary_cells)}")

# =========================
# CONES MIT MINDESTABSTAND
# =========================
cone_distance_px = CONE_DISTANCE_M / resolution
tree = KDTree(boundary_cells)

used = np.zeros(len(boundary_cells), dtype=bool)
cones_px = []

for i, p in enumerate(boundary_cells):
    if used[i]:
        continue
    cones_px.append(p)
    idxs = tree.query_ball_point(p, cone_distance_px)
    used[idxs] = True

cones_px = np.array(cones_px)
print(f"Virtuelle Cones platziert: {len(cones_px)}")

# =========================
# PIXEL → WELTKOORDINATEN
# =========================
cones_world = []

for px, py in cones_px:
    x_world = origin_x + px * resolution
    y_world = origin_y + (height - py) * resolution  # y-Achse invertiert
    cones_world.append((x_world, y_world))

# =========================
# AUSGABE
# =========================
with open(OUTPUT_FILE, "w") as f:
    f.write("x,y\n")
    for x, y in cones_world:
        f.write(f"{x:.3f},{y:.3f}\n")

print(f"Cones gespeichert in {OUTPUT_FILE}")
