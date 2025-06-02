import cv2
import numpy as np
import typing

def load_img(img_path: str) -> cv2.typing.MatLike:
    return cv2.imread(img_path)

def get_img_size(img: cv2.typing.MatLike) -> tuple[int, int]:
    '''
    rows, columns
    '''
    return img.shape[0], img.shape[1]

def show_img(img: cv2.typing.MatLike, title: str = "Image", scaleX: float = 1, scaleY: float = 1) -> None:
    cv2.imshow(title, cv2.resize(img, None, fx=scaleX, fy=scaleY))

def grayscale_img(img: cv2.typing.MatLike) -> cv2.typing.MatLike:
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[..., np.newaxis]

def normalize_img(img: cv2.typing.MatLike) -> cv2.typing.MatLike:
    intensity = grayscale_img(img)
    maxI = np.max(intensity)
    minI = np.min(intensity)

    return ((intensity - minI) / (maxI - minI)) * 255

def gradient_orientation_field(normalized_img: cv2.typing.MatLike, 
                               sum_kernel_size: int = 3, 
                               blur_kernel_size: int = 5, 
                               blur_power: float = 0.77) -> tuple[cv2.typing.MatLike, cv2.typing.MatLike]:
    '''
    Orientation Field, Weigths
    '''

    Gx = cv2.Sobel(normalized_img, cv2.CV_64F, dx=1, dy=0, ksize=3)
    Gy = cv2.Sobel(normalized_img, cv2.CV_64F, dx=0, dy=1, ksize=3)

    Gx2 = np.pow(Gx, 2)
    Gy2 = np.pow(Gy, 2)
    GxGy = Gx * Gy

    kernel = np.ones((sum_kernel_size, sum_kernel_size), dtype=np.float64)

    sum_gx2 = cv2.filter2D(Gx2, -1, kernel)
    sum_gy2 = cv2.filter2D(Gy2, -1, kernel)
    sum_gxgy = cv2.filter2D(GxGy, -1, kernel)

    sum_gx_sqr_sub_gy_sqr = sum_gx2 - sum_gy2
    sum_gx_sqr_sum_gy_sqr = sum_gx2 + sum_gy2

    orientation_field = 0.5 * np.arctan(2 * sum_gxgy / (sum_gx_sqr_sub_gy_sqr + 1e-10))

    k = np.zeros_like(orientation_field)
    cond_half = ((orientation_field < 0) & (2 * sum_gxgy < 0)) | ((orientation_field >= 0) & (2 * sum_gxgy > 0))
    cond_one = (orientation_field < 0) & (2 * sum_gxgy >= 0)
    cond_zero = (orientation_field >= 0) & (2 * sum_gxgy <= 0)

    k[cond_half] = 0.5
    k[cond_one] = 1.0
    k[cond_zero] = 0.0

    orientation_field += k * np.pi
    orientation_field = cv2.GaussianBlur(orientation_field, (blur_kernel_size, blur_kernel_size), sigmaX=blur_power)

    weights = (sum_gx_sqr_sub_gy_sqr ** 2 + 4 * sum_gxgy ** 2) / (sum_gx_sqr_sum_gy_sqr ** 2 + 1e-10)
    weights = cv2.GaussianBlur(weights, (blur_kernel_size, blur_kernel_size), sigmaX=blur_power)

    return orientation_field, weights

def average_orientation_field(orientation_field: cv2.typing.MatLike, 
                              weights: cv2.typing.MatLike, 
                              block_size: int = 16) -> tuple[cv2.typing.MatLike, cv2.typing.MatLike]:
    '''
    Orientation field, Weights
    '''
    assert weights.shape == orientation_field.shape
    small_rows = orientation_field.shape[0] // block_size
    small_cols = orientation_field.shape[1] // block_size

    small_orientation_field = np.zeros((small_rows, small_cols), dtype=np.float64)
    small_weights = np.zeros((small_rows, small_cols), dtype=np.float64)

    for i in range(small_rows):
        for j in range(small_cols):
            two_orientation_field_block = 2 * orientation_field[i * block_size: (i + 1) * block_size, j * block_size: (j + 1) * block_size]
            weight_block = weights[i * block_size: (i + 1) * block_size, j * block_size: (j + 1) * block_size]

            small_orientation_field[i, j] = 0.5 * np.arctan2(np.mean(np.sin(two_orientation_field_block)), np.mean(np.cos(two_orientation_field_block)))
            small_weights[i, j] = 0.5 * np.mean(weight_block)

    return small_orientation_field, small_weights

def filter_clusters(coords: cv2.typing.MatLike, strengths: cv2.typing.MatLike = None, radius: int = 20) -> cv2.typing.MatLike:
    """
    Redukuje skupiska punktów — zostawia tylko jeden punkt w promieniu `radius`.
    
    Args:
        coords: tablica Nx2 współrzędnych (y, x) punktów.
        strengths: opcjonalna tablica N sił punktów — większa = ważniejsza.
        radius: promień (w pikselach), w którym usuwamy bliskie punkty.

    Returns:
        Lista współrzędnych (y, x) przefiltrowanych punktów.
    """
    from scipy.spatial import cKDTree

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
    tree = cKDTree(coords)

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

def filter_poincare_points(points_mask: np.typing.ArrayLike, 
                           weights: cv2.typing.MatLike, 
                           maximum_box_size: int = 15, border_size: int = 16) -> cv2.typing.MatLike:
    from scipy.ndimage import maximum_filter

    points_map = np.where(points_mask, weights, 0)
    points_max = maximum_filter(points_map, size=maximum_box_size)
    points_mask = (points_map == points_max) & (points_map > 0)

    points_mask[:border_size, :] = False
    points_mask[-border_size:, :] = False
    points_mask[:, :border_size] = False
    points_mask[:, -border_size:] = False

    points_yx = np.argwhere(points_mask)
    points_strengths = weights[points_mask]

    filtered_points = filter_clusters(points_yx, strengths=points_strengths, radius=20)
    for py, px in points_yx:
        if not any(item == (py, px) for item in filtered_points):
            points_mask[py, px] = False

    return points_mask

def poincare_index(orientation_field: cv2.typing.MatLike, weights: cv2.typing.MatLike, 
                   weights_min_power: float = 0.2, close_error: float = 0.5 * np.pi) -> tuple[cv2.typing.MatLike, cv2.typing.MatLike]:
    """
    Returns:
        cores_mask, deltas_mask
    """
    poincare_index_map = np.zeros_like(orientation_field)
    rows, columns = orientation_field.shape

    for y in range(1, rows - 1):
        for x in range(1, columns - 1):
            angles = [
                orientation_field[y - 1, x - 1],
                orientation_field[y, x - 1],
                orientation_field[y + 1, x - 1],
                orientation_field[y + 1, x],
                orientation_field[y + 1, x + 1],
                orientation_field[y, x + 1],
                orientation_field[y - 1, x + 1],
                orientation_field[y - 1, x]
            ]
            
            angles_diff = 0
            for i in range(8):
                angles_diff += (angles[(i + 1) % 8] - angles[i] + np.pi * 0.5) % np.pi - np.pi * 0.5
            
            poincare_index_map[y, x] = angles_diff

    core_mask = np.isclose(poincare_index_map, +np.pi, atol=close_error) & (weights > weights_min_power)
    delta_mask = np.isclose(poincare_index_map, -np.pi, atol=close_error) & (weights > weights_min_power)

    core_mask = filter_poincare_points(core_mask, weights)
    delta_mask = filter_poincare_points(delta_mask, weights)

    return core_mask, delta_mask

def weight_least_square(values: cv2.typing.MatLike, weights: cv2.typing.MatLike, polymonial_term: int = 4) -> cv2.typing.MatLike:
    rows, columns = values.shape
    Y, X = np.mgrid[0: rows, 0: columns]
    
    x = X.flatten()
    y = Y.flatten()
    v = values.flatten()
    w = weights.flatten()
    
    A = []
    for i in range(polymonial_term + 1):
        for j in range(polymonial_term + 1 - i):
            A.append((x ** i) * (y ** j))
    A = np.vstack(A).T
    
    W_sqrt = np.sqrt(w)
    A_weighted = A * W_sqrt[:, np.newaxis]
    V_weighted = v * W_sqrt
    
    coeffs, *_ = np.linalg.lstsq(A_weighted, V_weighted, rcond=None)
    return coeffs

def calculate_point_charge(all_points_y: np.typing.ArrayLike, 
                           all_points_x: np.typing.ArrayLike,
                           rows: int,
                           columns: int, 
                           points_mask: cv2.typing.MatLike, 
                           points_R: float, 
                           re_charge_func: typing.Callable[[float, float], float], 
                           im_charge_func: typing.Callable[[float, float], float]) -> tuple[list, list]:
    """
    Returns:
        charges, weights
    """    
    points_charges = []
    points_weights = []

    for py, px in np.argwhere(points_mask):
        dy = all_points_y - py
        dx = all_points_x - px
        r = np.sqrt(dx**2 + dy**2) + 1e-10

        mask = r <= points_R

        point_PC = np.zeros((rows, columns, 2))
        point_PC[..., 0] = np.where(mask, re_charge_func(dy, r), 0)
        point_PC[..., 1] = np.where(mask, im_charge_func(dx, r), 0)

        point_weight = 1 - np.minimum(r, points_R) / points_R

        points_charges.append(point_PC)
        points_weights.append(point_weight)

    return points_charges, points_weights

def point_charge_orientation_field(orientation_field: cv2.typing.MatLike,
                                   weights: cv2.typing.MatLike,
                                   cores_mask: cv2.typing.MatLike, 
                                   deltas_mask: cv2.typing.MatLike, 
                                   cores_R: int = 80, deltas_R: int = 40) -> cv2.typing.MatLike:
    O2 = 2 * orientation_field
    cos2O = np.cos(O2)
    sin2O = np.sin(O2)

    coeffs_cos = weight_least_square(cos2O, weights)
    coeffs_sin = weight_least_square(sin2O, weights)

    PR_PI = np.stack((cos2O, sin2O), axis=-1)

    rows, columns = orientation_field.shape

    yy, xx = np.meshgrid(np.arange(rows), np.arange(columns), indexing='ij')

    cores_charges, cores_weights = calculate_point_charge(yy, xx, rows, columns, cores_mask, cores_R, 
                                                          lambda dy, r: dy / r, lambda dx, r: -dx / r)
    deltas_charges, deltas_weights = calculate_point_charge(yy, xx, rows, columns, deltas_mask, deltas_R, 
                                                            lambda dy, r: -dy / r, lambda dx, r: -dx / r)

    points_charges = [*cores_charges, *deltas_charges]
    points_weights = [*cores_weights, *deltas_weights]
    
    model_weights = np.maximum(1 - np.sum(points_weights, axis=0), 0)

    weights_stack = np.stack(points_weights, axis=0)
    charges_stack = np.stack(points_charges, axis=0)

    combined = weights_stack[..., np.newaxis] * charges_stack
    U = model_weights[..., np.newaxis] * PR_PI + np.sum(combined, axis=0)

    return 0.5 * np.arctan2(U[..., 1], U[..., 0])

def draw_orientation_field(img: cv2.typing.MatLike, 
                           orientation_field: cv2.typing.MatLike, 
                           step: int = 16, line_length: float = 10, 
                           color: tuple[float, float, float] = (0, 255, 0), 
                           thickness: float = 1) -> cv2.typing.MatLike:
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