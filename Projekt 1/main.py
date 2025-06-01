import cv2
import numpy as np
import scipy.ndimage as sciimg
import scipy.spatial as scispat
import os
import numpy.random as rnd
import matplotlib.pyplot as plt

data_folder_path = "../Data/Odciski/DB1_B"

# LOAD FINGERPRINT
fingerprint = cv2.imread(f"{data_folder_path}/101_2.tif")
rows = fingerprint.shape[0]
columns = fingerprint.shape[1]

# cv2.imshow("Original", cv2.resize(fingerprint, None, fx=1, fy=1))

# INTENSITY MAP
fingerprint_intensity = cv2.cvtColor(fingerprint, cv2.COLOR_RGB2GRAY)[..., np.newaxis]

# cv2.imshow("Intensity", cv2.resize(fingerprint_intensity, None, fx=1, fy=1))

# NORMALIZE FINGERPRINT
maxI = np.max(fingerprint_intensity)
minI = np.min(fingerprint_intensity)

fingerprint_norm = ((fingerprint_intensity - minI) / (maxI - minI)) * 255

# cv2.imshow("Normalized", cv2.resize(fingerprint, None, fx=1, fy=1))

# CALCULATE ORIENTATION FIELD
# CALCULATE CORSE ORIENTATION FIELD
Gx = cv2.Sobel(fingerprint_norm, cv2.CV_64F, dx=1, dy=0, ksize=3)
Gy = cv2.Sobel(fingerprint_norm, cv2.CV_64F, dx=0, dy=1, ksize=3)

# cv2.imshow("Gx", cv2.resize(Gx, None, fx=1, fy=1))
# cv2.imshow("Gy", cv2.resize(Gy, None, fx=1, fy=1))

Gx2 = np.pow(Gx, 2)
Gy2 = np.pow(Gy, 2)
GxGy = Gx * Gy

# cv2.imshow("Gx2", cv2.resize(Gx2, None, fx=1, fy=1))
# cv2.imshow("Gy2", cv2.resize(Gy2, None, fx=1, fy=1))
# cv2.imshow("GxGy", cv2.resize(GxGy, None, fx=1, fy=1))

box_size = 3
kernel = np.ones((box_size, box_size), dtype=np.float64)

sum_gx2 = cv2.filter2D(Gx2, -1, kernel)
sum_gy2 = cv2.filter2D(Gy2, -1, kernel)
sum_gxgy = cv2.filter2D(GxGy, -1, kernel)

# cv2.imshow("Sum Gx2", cv2.resize(sum_gx2, None, fx=1, fy=1))
# cv2.imshow("Sum Gy2", cv2.resize(sum_gy2, None, fx=1, fy=1))
# cv2.imshow("Sum GxGy", cv2.resize(sum_gxgy, None, fx=1, fy=1))

sum_gx_sqr_sub_gy_sqr = sum_gx2 - sum_gy2
sum_gx_sqr_sum_gy_sqr = sum_gx2 + sum_gy2

# cv2.imshow("Sum Gx2 - Sum Gy2", cv2.resize(sum_gx_sqr_sub_gy_sqr, None, fx=1, fy=1))
# cv2.imshow("Sum Gx2 + Sum Gy2", cv2.resize(sum_gx_sqr_sum_gy_sqr, None, fx=1, fy=1))

fingerprint_O = 0.5 * np.arctan(2 * sum_gxgy / (sum_gx_sqr_sub_gy_sqr + 1e-10))

k = np.zeros_like(fingerprint_O)
cond_half = ((fingerprint_O < 0) & (2 * sum_gxgy < 0)) | ((fingerprint_O >= 0) & (2 * sum_gxgy > 0))
cond_one = (fingerprint_O < 0) & (2 * sum_gxgy >= 0)
cond_zero = (fingerprint_O >= 0) & (2 * sum_gxgy <= 0)

k[cond_half] = 0.5
k[cond_one] = 1.0
k[cond_zero] = 0.0

fingerprint_O += k * np.pi
fingerprint_O = cv2.GaussianBlur(fingerprint_O, (5, 5), sigmaX=0.77)
fingerprint_O = fingerprint_O.squeeze()
# cv2.imshow("Fingerprint Orientation", cv2.resize(fingerprint_O, None, fx=1, fy=1))

fingerprint_W = (sum_gx_sqr_sub_gy_sqr ** 2 + 4 * sum_gxgy ** 2) / (sum_gx_sqr_sum_gy_sqr ** 2 + 1e-10)
fingerprint_W = cv2.GaussianBlur(fingerprint_W, (5, 5), sigmaX=0.77)
fingerprint_W = fingerprint_W.squeeze()
# cv2.imshow("Fingerprint Weight", cv2.resize(fingerprint_W, None, fx=1, fy=1))

cos2O = np.cos(2 * fingerprint_O)
sin2O = np.sin(2 * fingerprint_O)

# cv2.imshow("Cos(2O)", cv2.resize(cos2O, None, fx=1, fy=1))
# cv2.imshow("Sin(2O)", cv2.resize(sin2O, None, fx=1, fy=1))

plt.figure(figsize=(20, 5))

x, y = np.meshgrid(np.arange(columns), np.arange(rows))

plt.subplot(1, 4, 1)
plt.title("Orginalne pole orientacji")
plt.quiver(x, y, np.cos(2 * fingerprint_O), np.sin(2 * fingerprint_O), fingerprint_O)
plt.gca().invert_yaxis()
plt.axis("equal")

plt.tight_layout()

# POINCARE INDEX
poincare_index_map = np.zeros_like(fingerprint_O)

for y in range(1, rows - 1):
    for x in range(1, columns - 1):
        angles = [
            fingerprint_O[y - 1, x - 1],
            fingerprint_O[y, x - 1],
            fingerprint_O[y + 1, x - 1],
            fingerprint_O[y + 1, x],
            fingerprint_O[y + 1, x + 1],
            fingerprint_O[y, x + 1],
            fingerprint_O[y - 1, x + 1],
            fingerprint_O[y - 1, x]
        ]
        
        angles_diff = 0
        for i in range(8):
            angles_diff += (angles[(i + 1) % 8] - angles[i] + np.pi * 0.5) % np.pi - np.pi * 0.5
        
        poincare_index_map[y, x] = angles_diff

core_mask = np.isclose(poincare_index_map, +np.pi, atol=np.pi * 0.1) & (fingerprint_W > 0.9)
delta_mask = np.isclose(poincare_index_map, -np.pi, atol=np.pi * 0.1) & (fingerprint_W > 0.9)

core_map = np.where(core_mask, fingerprint_W, 0)
core_max = sciimg.maximum_filter(core_map, size=15)
core_mask = (core_map == core_max) & (core_map > 0)

delta_map = np.where(delta_mask, fingerprint_W, 0)
delta_max = sciimg.maximum_filter(delta_map, size=15)
delta_mask = (delta_map == delta_max) & (delta_map > 0)

border = 16
core_mask[:border, :] = False
core_mask[-border:, :] = False
core_mask[:, :border] = False
core_mask[:, -border:] = False

delta_mask[:border, :] = False
delta_mask[-border:, :] = False
delta_mask[:, :border] = False
delta_mask[:, -border:] = False

def filter_clusters(coords, strengths=None, radius=20):
    """
    Redukuje skupiska punktów — zostawia tylko jeden punkt w promieniu `radius`.
    
    Args:
        coords: tablica Nx2 współrzędnych (y, x) punktów.
        strengths: opcjonalna tablica N sił punktów — większa = ważniejsza.
        radius: promień (w pikselach), w którym usuwamy bliskie punkty.

    Returns:
        Lista współrzędnych (y, x) przefiltrowanych punktów.
    """
    if len(coords) == 0:
        return []

    coords = np.array(coords)
    if strengths is None:
        strengths = np.ones(len(coords))

    # Posortuj punkty od najsilniejszego
    idx_sorted = np.argsort(-strengths)
    coords = coords[idx_sorted]
    strengths = strengths[idx_sorted]

    kept = []
    tree = scispat.cKDTree(coords)

    removed = np.zeros(len(coords), dtype=bool)

    for i, pt in enumerate(coords):
        if removed[i]:
            continue
        kept.append(tuple(pt))
        neighbors = tree.query_ball_point(pt, r=radius)
        for n in neighbors:
            removed[n] = True  # oznacz wszystko w pobliżu jako usunięte
        removed[i] = False  # ale siebie samego zostawiamy

    return kept

core_yx = np.argwhere(core_mask)
core_strengths = fingerprint_W[core_mask]

filtered_cores = filter_clusters(core_yx, strengths=core_strengths, radius=20)
for cy, cx in core_yx:
    if not any(item == (cy, cx) for item in filtered_cores):
        core_mask[cy, cx] = False

delta_yx = np.argwhere(delta_mask)
delta_strengths = fingerprint_W[delta_mask]

filtered_deltas = filter_clusters(delta_yx, strengths=delta_strengths, radius=20)
for dy, dx in delta_yx:
    if not any(item == (dy, dx) for item in filtered_deltas):
        delta_mask[dy, dx] = False

# print(np.count_nonzero(core_mask), np.count_nonzero(delta_mask))

plt.subplot(1, 4, 2)
plt.imshow(fingerprint_O, cmap='gray')
plt.scatter(*np.where(core_mask)[::-1], color='red', label='Core', s=10)
plt.scatter(*np.where(delta_mask)[::-1], color='blue', label='Delta', s=10)
plt.legend()
plt.title("Singular Points (Poincare Index)")
plt.tight_layout()
plt.axis("equal")

plt.subplot(1, 4, 3)
plt.hist(poincare_index_map.ravel(), bins=100)
plt.title("Histogram wartości Poincarégo")
plt.tight_layout()
plt.axis("equal")

plt.subplot(1, 4, 4)
plt.imshow(fingerprint_O, cmap='hsv')
plt.title("Orientation Field")
plt.colorbar()
plt.tight_layout()
plt.axis("equal")

# POINT CHARGE
PR_PI = np.stack((cos2O, sin2O), axis=-1)

core_R = 80
delta_R = 40

points_charges = []
points_weights = []

yy, xx = np.meshgrid(np.arange(rows), np.arange(columns), indexing='ij')

for c_y, c_x in np.argwhere(core_mask):
    dy = yy - c_y
    dx = xx - c_x
    r = np.sqrt(dx**2 + dy**2) + 1e-10

    mask = r <= core_R

    core_PC = np.zeros((rows, columns, 2))
    core_PC[..., 0] = np.where(mask, dy / r, 0)
    core_PC[..., 1] = np.where(mask, -dx / r, 0)

    core_weight = 1 - np.clip(r / core_R, 0, 1)

    points_charges.append(core_PC)
    points_weights.append(core_weight)

for d_y, d_x in np.argwhere(delta_mask):
    dy = yy - d_y
    dx = xx - d_x
    r = np.sqrt(dx**2 + dy**2) + 1e-10

    mask = r <= delta_R

    delta_PC = np.zeros((rows, columns, 2))
    delta_PC[..., 0] = np.where(mask, -dy / r, 0)
    delta_PC[..., 1] = np.where(mask, dx / r, 0)

    delta_weight = 1 - np.clip(r / delta_R, 0, 1)

    points_charges.append(delta_PC)
    points_weights.append(delta_weight)

model_weights = np.maximum(1 - np.sum(points_weights, axis=0), 0)

weights_stack = np.stack(points_weights, axis=0)
charges_stack = np.stack(points_charges, axis=0)

combined = weights_stack[..., np.newaxis] * charges_stack
U = model_weights[..., np.newaxis] * PR_PI + np.sum(combined, axis=0)

RE, IM = U[..., 0], U[..., 1]

final_orientation = 0.5 * np.arctan2(RE, IM)

plt.figure(figsize=(15, 5))

plt.subplot(1, 4, 1)
plt.imshow(RE, cmap='gray')
plt.title("Real Part (cos2O)")
plt.tight_layout()
plt.axis("equal")

plt.subplot(1, 4, 2)
plt.imshow(IM, cmap='gray')
plt.title("Imaginary (sin2O)")
plt.tight_layout()
plt.axis("equal")

plt.subplot(1, 4, 3)
plt.imshow(final_orientation, cmap='gray')
plt.title("Orientation Field")
plt.tight_layout()
plt.axis("equal")

plt.subplot(1, 4, 4)
plt.imshow(final_orientation, cmap='hsv')
plt.title("Orientation Field")
plt.colorbar()
plt.tight_layout()
plt.axis("equal")
plt.show()


def draw_orientation_field(img, orientation_field, step=16, line_length=10, color=(0, 255, 0), thickness=1):
    """
    Rysuje pole orientacji jako linie na obrazie.
    
    img - obraz w tle (np. odcisk palca, może być grayscale lub kolorowy)
    orientation_field - macierz kątów O w radianach (np. o rozdzielczości co step pikseli)
    step - odległość między punktami siatki (np. 16 pikseli)
    line_length - długość linii
    """
    # Upewnij się, że obraz jest w kolorze (do rysowania)
    img_color = img.copy()

    rows, cols = orientation_field.shape
    for y in range(0, rows):
        for x in range(0, cols):
            angle = orientation_field[y, x]

            # Środek bloku w obrazie
            center_x = x * step + step // 2
            center_y = y * step + step // 2

            # Kierunek linii
            dx = int(np.cos(angle) * line_length / 2)
            dy = int(np.sin(angle) * line_length / 2)

            pt1 = (center_x - dx, center_y - dy)
            pt2 = (center_x + dx, center_y + dy)

            # Rysuj linię
            cv2.line(img_color, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)

    return img_color

def average_orientation_field(O_full, block_size=16):
    rows, cols = O_full.shape
    small_rows = rows // block_size
    small_cols = cols // block_size

    O_small = np.zeros((small_rows, small_cols), dtype=np.float32)

    for i in range(small_rows):
        for j in range(small_cols):
            block = O_full[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]

            # Uśrednianie przez reprezentację wektorową
            cos2O = np.cos(2 * block)
            sin2O = np.sin(2 * block)

            avg_cos2O = np.mean(cos2O)
            avg_sin2O = np.mean(sin2O)

            O_avg = 0.5 * np.arctan2(avg_sin2O, avg_cos2O)
            O_small[i, j] = O_avg

    return O_small

fingerprint_O_small = average_orientation_field(fingerprint_O, block_size=16)
overlay = draw_orientation_field(fingerprint_norm, fingerprint_O_small, step=16, line_length=12)
cv2.imshow("Orientation Field", overlay)

final_orientation_small = average_orientation_field(final_orientation, block_size=16)
overlay = draw_orientation_field(fingerprint_norm, final_orientation_small, step=16, line_length=12)
cv2.imshow("Final Orientation Field", overlay)

cv2.waitKey(0)
cv2.destroyAllWindows()

# TODO: Policzyć Orientation Field dla 16x16 bloków i na nich operować nie na pełnym obrazie (zniweczy szum i pozwoli na lepszy obraz poprzez rysowanie lini)
# TODO: Zrobić repo
# TODO: pod liniami by było widać analizowany obraz
# TODO: Podzielić na funkcje
# TODO: Zrobić z tego API
# TODO: Użyć numby lub coś z cudą