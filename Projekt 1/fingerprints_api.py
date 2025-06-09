import cv2
import numpy as np
import typing
import json
import matplotlib.pyplot as plt
import matplotlib.markers as markers

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
    return ((intensity - minI) / (maxI - minI))

def binarize_img(img: cv2.typing.MatLike, box_size: int = 5) -> cv2.typing.MatLike:
    binarized = np.zeros_like(img)

    rows = img.shape[0]
    columns = img.shape[1]

    small_rows = rows // box_size
    small_columns = columns // box_size

    for y in range(small_rows):
        for x in range(small_columns):
            # Get box values
            box = img[y * box_size:min(y * box_size + box_size, rows), x * box_size:min(x * box_size + box_size, columns)]
            local_threshold = np.mean(box)
            box_mask = box >= local_threshold
            binarized_box = np.where(box_mask, 1, 0)
            binarized[y * box_size:min(y * box_size + box_size, rows), x * box_size:min(x * box_size + box_size, columns)] = binarized_box

    return binarized

def skeletonize_img(img: cv2.typing.MatLike) -> cv2.typing.MatLike:
    img_u8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.ximgproc.thinning(img_u8, thinningType=cv2.ximgproc.THINNING_GUOHALL)

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

    orientation_field_2 = 2 * orientation_field

    cos2O = cv2.GaussianBlur(np.cos(orientation_field_2), (blur_kernel_size, blur_kernel_size), sigmaX=blur_power)
    sin2O = cv2.GaussianBlur(np.sin(orientation_field_2), (blur_kernel_size, blur_kernel_size), sigmaX=blur_power)

    orientation_field = 0.5 * np.arctan2(sin2O, cos2O)

    weights = (sum_gx_sqr_sub_gy_sqr ** 2 + 4 * sum_gxgy ** 2) / (sum_gx_sqr_sum_gy_sqr ** 2 + 1e-10)
    weights = cv2.GaussianBlur(weights, (blur_kernel_size, blur_kernel_size), sigmaX=blur_power)

    return (orientation_field + np.pi) % np.pi, weights

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

def get_largest_reliable_region(reliability_map: cv2.typing.MatLike, threshold: float = 0.3) -> np.ndarray:
    """
    Returns:
        array of contour points
    """

    mask = (reliability_map > threshold).astype(np.uint8)

    # Denoising: otwarcie morfologiczne
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # kontur
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("Brak obszarów powyżej progu wiarygodności")
    
    # Największy kontur
    return max(contours, key=cv2.contourArea).squeeze()

def get_reliable_region_border(contour: np.ndarray, border_size: int = 16) -> np.ndarray:
    center = np.array([contour[..., 0].mean(), contour[..., 1].mean()])

    border_contour = np.zeros_like(contour)
    for i in range(len(contour)):
        direction = center - contour[i]
        direction /= np.sqrt(direction[0] ** 2 + direction[1] ** 2)

        border_contour[i] = contour[i] + direction * border_size
    return border_contour

def get_reliable_region_rectangle(contour: np.ndarray) -> tuple[int, int, int, int]:
    """
    Returns:
        start_y, end_y, start_x, end_x
    """

    x, y, w, h = cv2.boundingRect(contour)
    return y, y + h, x, x + w

def generate_gabor_kernels(ksize=21, sigma=5.0, lambd=10.0, gamma=0.5, psi=0, num_angles=16):
    angles = np.linspace(0, np.pi, num_angles, endpoint=False)
    kernels = []
    for theta in angles:
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_64F)
        kernels.append(kernel)
    return angles, kernels

def directional_filtering(img, orientation_field, weights, block_size=16):
    rows = img.shape[0]
    columns = img.shape[1]
    dfi = np.zeros_like(img, dtype=np.float64)

    angles, kernels = generate_gabor_kernels()

    for y in range(0, rows, block_size):
        for x in range(0, columns, block_size):
            y_end = min(y + block_size, rows)
            x_end = min(x + block_size, columns)

            block_orientation = orientation_field[y:y_end, x:x_end]
            mean_theta = np.mean(block_orientation)

            block_weights = weights[y:y_end, x:x_end]
            mean_weight = np.mean(block_weights)

            if mean_weight < 0.1:
                # pomiń bardzo niestabilne obszary
                continue

            angle_diffs = np.abs(np.angle(np.exp(1j * (angles - mean_theta))))
            kernel_idx = np.argmin(angle_diffs)
            kernel = kernels[kernel_idx]

            block = img[y:y_end, x:x_end]

            k_half = kernel.shape[0] // 2
            padded_block = cv2.copyMakeBorder(block, k_half, k_half, k_half, k_half, borderType=cv2.BORDER_REFLECT)
            filtered = cv2.filter2D(padded_block, -1, kernel)
            filtered_block = filtered[k_half:k_half + y_end - y, k_half:k_half + x_end - x]

            # uwzględnij niezawodność: ważenie liniowe
            dfi[y:y_end, x:x_end] = filtered_block[..., np.newaxis] * mean_weight + dfi[y:y_end, x:x_end] * (1 - mean_weight)

    return dfi.squeeze()

def filter_clusters(coords: cv2.typing.MatLike, strengths: cv2.typing.MatLike = None, radius: int = 20) -> np.ndarray:
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

    return np.array(kept)

def filter_poincare_points(points_mask: cv2.typing.MatLike, 
                           points_scores: cv2.typing.MatLike,
                           reliability_map: cv2.typing.MatLike,
                           contour_border: np.ndarray | None = None,
                           min_reliability: float = 0.3,
                           border_size: int = 16,
                           near_radius: int = 20) -> cv2.typing.MatLike:

    filtered_points_mask = points_mask & (reliability_map > min_reliability)

    filtered_points_mask[:border_size, :] = False
    filtered_points_mask[-border_size:, :] = False
    filtered_points_mask[:, :border_size] = False
    filtered_points_mask[:, -border_size:] = False

    points_yx = np.argwhere(filtered_points_mask)
    best_points_score = np.min(points_scores)
    for py, px in points_yx:
        filtered_points_mask[py, px] = points_scores[py, px] == best_points_score

    points_yx = np.argwhere(filtered_points_mask)
    points_reliability = reliability_map[filtered_points_mask]
    filtered_points = filter_clusters(points_yx, strengths=points_reliability, radius=near_radius)
    for py, px in points_yx:
        if not any(y == py and x == px for y, x in filtered_points):
            filtered_points_mask[py, px] = False

    if contour_border is not None:
        points_yx = np.argwhere(filtered_points_mask)
        for py, px in points_yx:
            inside = cv2.pointPolygonTest(contour_border, (px.astype(float), py.astype(float)), False)
            filtered_points_mask[py, px] = inside >= 0

    return filtered_points_mask

def get_best_poincare_point(points_mask: cv2.typing.MatLike, reliability_map: cv2.typing.MatLike, 
                            get_score: typing.Callable[[float, np.ndarray, np.ndarray, np.ndarray], float], 
                            radius: int = 30) -> np.ndarray | None:
    """_summary_

    Args:
        points_mask (cv2.typing.MatLike): _description_
        reliability_map (cv2.typing.MatLike): _description_
        get_score (typing.Callable[[np.ndarray, np.ndarray, np.ndarray], float]): avg_reliability, position, center, image_size
        radius (int, optional): _description_. Defaults to 30.

    Returns:
        np.ndarray | None: _description_
    """
    h, w = reliability_map.shape
    center = np.array([w // 2, h // 2])

    best_score = -np.inf
    best_point = None

    for pt in np.argwhere(points_mask):
        y, x = pt

        y0 = max(0, y - radius)
        y1 = min(h, y + radius)
        x0 = max(0, x - radius)
        x1 = min(w, x + radius)
        region = reliability_map[y0:y1, x0:x1]
        avg_reliability = np.mean(region)
        
        score = get_score(avg_reliability, np.array([x, y]), center, np.array([w, h]))

        if score > best_score:
            best_score = score
            best_point = pt
    
    return best_point

def get_best_core(cores_mask: cv2.typing.MatLike, reliability_map: cv2.typing.MatLike, radius: int = 30) -> np.ndarray | None:
    def get_score(avg_reliability: float, pos: np.ndarray, center: np.ndarray, img_size: np.ndarray) -> float:
        dist_to_center = np.linalg.norm(pos - center)
        return avg_reliability - 0.01 * dist_to_center
    
    return get_best_poincare_point(cores_mask, reliability_map, get_score, radius)

def get_best_delta(deltas_mask: cv2.typing.MatLike, reliability_map: cv2.typing.MatLike, radius: int = 30) -> np.ndarray | None:
    def get_score(avg_reliability: float, pos: np.ndarray, center: np.ndarray, img_size: np.ndarray) -> float:
        vertical_bias = pos[1] / img_size[1]
        return avg_reliability + 0.2 * vertical_bias
    
    return get_best_poincare_point(deltas_mask, reliability_map, get_score, radius)

def poincare_index(orientation_field: cv2.typing.MatLike, reliability_map: cv2.typing.MatLike, 
                   contour_border: np.ndarray | None = None, min_reliability: float = 0.2, 
                   close_error: float = 0.5 * np.pi) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Returns:
        core_point, delta_point
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

    cores_mask = poincare_index_map > np.pi - close_error
    deltas_mask = poincare_index_map < -np.pi + close_error

    cores_scores = np.abs(np.pi - poincare_index_map)
    deltas_scores = np.abs(-np.pi + poincare_index_map)

    cores_mask = filter_poincare_points(cores_mask, cores_scores, reliability_map, contour_border, min_reliability, 16, 40)
    deltas_mask = filter_poincare_points(deltas_mask, deltas_scores, reliability_map, contour_border, min_reliability, 16, 40)

    core_point = get_best_core(cores_mask, reliability_map, 30)
    delta_point = get_best_delta(deltas_mask, reliability_map, 30)

    return core_point, delta_point

def build_polymonial_basis(Xs: np.typing.ArrayLike, Ys: np.typing.ArrayLike, degree: int = 4) -> cv2.typing.MatLike:
    x = Xs.flatten()
    y = Ys.flatten()

    terms = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            terms.append((x ** i) * (y ** j))

    return np.stack(terms, axis=1)

def weight_least_square(values: cv2.typing.MatLike, weights: cv2.typing.MatLike, polymonial_degree: int = 4) -> cv2.typing.MatLike:
    rows, columns = values.shape
    X, Y = np.meshgrid(np.arange(columns), np.arange(rows))

    v = values.flatten()
    w = weights.flatten()

    A = build_polymonial_basis(X, Y, polymonial_degree)
    
    W_sqrt = np.sqrt(w)
    A_weighted = A * W_sqrt[:, np.newaxis]
    V_weighted = v * W_sqrt
    
    coeffs, *_ = np.linalg.lstsq(A_weighted, V_weighted, rcond=None)
    return coeffs

def eval_polymonial(Xs, Ys, coeffs, degree):
    A = build_polymonial_basis(Xs, Ys, degree)
    result = A @ coeffs
    return result.reshape((-1,))

def polymonial_orientation_field(orientation_field: cv2.typing.MatLike, 
                                 weights: cv2.typing.MatLike,
                                 polymonial_degree: int = 4) -> tuple[cv2.typing.MatLike, cv2.typing.MatLike]:
    """
    Returns:
        PR, PI
    """
    O2 = 2 * orientation_field
    sin2O = np.sin(O2)
    cos2O = np.cos(O2)

    sin2O_coeffs = weight_least_square(sin2O, weights, polymonial_degree)
    cos2O_coeffs = weight_least_square(cos2O, weights, polymonial_degree)

    rows, columns = orientation_field.shape

    X, Y = np.meshgrid(np.arange(columns), np.arange(rows))
    X = X.flatten()
    Y = Y.flatten()

    PR = eval_polymonial(X, Y, sin2O_coeffs, polymonial_degree).reshape(rows, columns)
    PI = eval_polymonial(X, Y, cos2O_coeffs, polymonial_degree).reshape(rows, columns)
    return PR, PI

def get_point_influence(Xs, Ys, xc, yc, max_R):
    r = np.sqrt((Xs - xc)**2 + (Ys - yc)**2) + 1e-10
    return 1 - np.minimum(r, max_R) / max_R

def get_point_charge(Xs, Ys, orientation_field, PR, PI, weights, xc, yc, max_R):
    phi = get_point_influence(Xs, Ys, xc, yc, max_R)
    diffrence = np.pow(PR - np.cos(2 * orientation_field), 2) + np.pow(PI - np.sin(2 * orientation_field), 2)

    num = np.sum(weights * phi * diffrence)
    denom = np.sum(weights * phi**2)

    if denom == 0:
        return 0.0
    return num / denom

def get_points_charges(orientation_field, PR, PI, weights, points_mask, max_R):
    rows, columns = orientation_field.shape

    Xs, Ys = np.meshgrid(np.arange(columns), np.arange(rows))
    
    charges = []
    for xc, yc in np.argwhere(points_mask):
        circle_mask = (Xs - xc)**2 + (Ys - yc)**2 <= max_R**2
        
        indexes = np.nonzero(circle_mask)

        q = get_point_charge(Xs[indexes], Ys[indexes], 
                             orientation_field[indexes], 
                             PR[indexes],
                             PI[indexes], 
                             weights[indexes], xc, yc, max_R)
        charges.append(q)
    return charges

def calculate_point_charge(Xs: np.typing.ArrayLike,
                           Ys: np.typing.ArrayLike, 
                           rows: int,
                           columns: int, 
                           points_mask: cv2.typing.MatLike, 
                           points_electricity,
                           points_R: float, 
                           re_charge_func: typing.Callable[[float, float], float], 
                           im_charge_func: typing.Callable[[float, float], float]) -> tuple[list, list]:
    """
    Returns:
        charges, weights
    """    
    points_charges = []
    points_weights = []

    for px, py in np.argwhere(points_mask):
        dy = Ys - py
        dx = Xs - px
        r = np.sqrt(dx**2 + dy**2) + 1e-10

        mask = r <= points_R

        point_PC = np.zeros((rows, columns, 2))
        point_PC[..., 0] = np.where(mask, re_charge_func(dy, r) * points_electricity[len(points_charges)], 0)
        point_PC[..., 1] = np.where(mask, im_charge_func(dx, r) * points_electricity[len(points_charges)], 0)

        point_weight = 1 - np.minimum(r, points_R) / points_R

        points_charges.append(point_PC)
        points_weights.append(point_weight)

    return points_charges, points_weights

def point_charge_orientation_field(PR,
                                   PI,
                                   orientation_field: cv2.typing.MatLike,
                                   cores_mask: cv2.typing.MatLike,
                                   deltas_mask: cv2.typing.MatLike,
                                   cores_electricity,
                                   deltas_electricity,
                                   cores_R: int = 80, deltas_R: int = 40) -> cv2.typing.MatLike:
    
    PR_PI = np.stack((PR, PI), axis=-1)

    rows, columns = orientation_field.shape

    Ys, Xs = np.meshgrid(np.arange(rows), np.arange(columns), indexing='ij')

    cores_charges, cores_weights = calculate_point_charge(Ys, Xs, rows, columns, cores_mask, cores_electricity, cores_R, 
                                                          lambda dy, r: dy / r, lambda dx, r: dx / r)
    deltas_charges, deltas_weights = calculate_point_charge(Ys, Xs, rows, columns, deltas_mask, deltas_electricity, deltas_R, 
                                                            lambda dy, r: dy / r, lambda dx, r: -dx / r)

    points_charges = [*cores_charges, *deltas_charges]
    points_weights = [*cores_weights, *deltas_weights]

    model_weights = np.maximum(1 - np.sum(points_weights, axis=0), 0)

    weights_stack = np.stack(points_weights, axis=0)
    charges_stack = np.stack(points_charges, axis=0)

    combined = weights_stack[..., np.newaxis] * charges_stack
    U = model_weights[..., np.newaxis] * PR_PI + np.sum(combined, axis=0)

    return 0.5 * np.arctan2(U[..., 1], U[..., 0])

def get_point_mean_angle(point: np.ndarray | None, orientation_field: cv2.typing.MatLike, avg_window_size: int = 5) -> float:
    if point is None:
        return 0

    h, w = orientation_field.shape
    half = avg_window_size // 2

    x0, x1 = max(0, point[1] - half), min(w, point[1] + half + 1)
    y0, y1 = max(0, point[0] - half), min(h, point[0] + half + 1)
    region = orientation_field[y0:y1, x0:x1]

    cos_vals = np.cos(2 * region)
    sin_vals = np.sin(2 * region)

    mean_angle = 0.5 * np.arctan2(np.mean(sin_vals), np.mean(cos_vals))
    return mean_angle

MINUTIAE_ENDING = 'ending'
MINUTIAE_BIFURCATION = 'bifurcation'
MINUTIAE_POS = 'pos'
MINUTIAE_TYPE = 'type'
MINUTIAE_ANGLE = 'angle'
MINUTIAE_RELIABILITY = 'reliability'

def extract_minutiae(skeleton: cv2.typing.MatLike, reliability_map: cv2.typing.MatLike,
                     orientation_field: cv2.typing.MatLike, 
                     border_contour: np.ndarray | None = None, border: int = 10,
                     reliability_threshold: float = 0.5) -> list[dict]:
    """
    Ekstrahuje punkty minutiae (zakończenia i rozwidlenia) z binarnego obrazu szkieletowego.

    Parametry:
        skeleton (np.ndarray): Obraz szkieletu (wartości 0 lub 255)

    Zwraca:
        List[Dict]: Lista punktów minutiae w postaci słowników {'pos', 'type', 'orientation', 'reliability'}
    """

    # Upewnij się, że mamy wartości 0 i 1 (a nie 0 i 255)
    skel = (skeleton > 0).astype(np.uint8)

    # Definiujemy kernel do zliczania sąsiadów (bez środka)
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)

    # Zastosuj konwolucję w celu zliczenia aktywnych sąsiadów
    neighbor_count = cv2.filter2D(skel, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_CONSTANT)

    minutiae = []

    rows, cols = skel.shape
    for y in range(1, rows - 1):
        for x in range(1, cols - 1):
            if skel[y, x] != 1:
                continue

            # Filtracja krawędzi
            if x < border or x >= cols - border or y < border or y >= rows - border:
                continue
            
            # Filtracja reliability
            if reliability_map[y, x] < reliability_threshold:
                continue

            # Filtracja po konturze
            if border_contour is not None:
                inside = cv2.pointPolygonTest(border_contour, [x, y], False)
                if inside < 0:
                    continue

            count = neighbor_count[y, x]
            if count == 1:
                m_type = MINUTIAE_ENDING
            elif count == 3:
                m_type = MINUTIAE_BIFURCATION
            else:
                continue

            minutiae.append({
                MINUTIAE_POS: np.array([y, x]),
                MINUTIAE_ANGLE: get_point_mean_angle(np.array([y, x]), orientation_field, 5),
                MINUTIAE_RELIABILITY: float(reliability_map[y, x]),
                MINUTIAE_TYPE: m_type
            })

    return minutiae

def normalize_minutiae(minutiae: list[dict], core: np.ndarray, core_angle: float) -> list[dict]:
    normalized = []
    cos_a = np.cos(-core_angle)
    sin_a = np.sin(-core_angle)

    distances = [np.sqrt((m[MINUTIAE_POS][0] - core[0]) ** 2 + (m[MINUTIAE_POS][1] - core[1]) ** 2) for m in minutiae]
    mean_dist = np.mean(distances)
    scale = 100 / mean_dist if mean_dist > 0 else 1.0

    for m in minutiae:
        dp = m[MINUTIAE_POS] - core

        # Obrót
        x_new = dp[1] * cos_a - dp[0] * sin_a
        y_new = dp[1] * sin_a + dp[0] * cos_a
        
        # Skalowanie
        x_new *= scale
        y_new *= scale

        # Korekta kąta
        angle_new = (m[MINUTIAE_ANGLE] - core_angle) % np.pi

        normalized.append({
            MINUTIAE_POS: np.array([y_new, x_new]),
            MINUTIAE_ANGLE: angle_new,
            MINUTIAE_RELIABILITY: m[MINUTIAE_RELIABILITY],
            MINUTIAE_TYPE: m[MINUTIAE_TYPE]
        })
    
    return normalized

def draw_orientation_field(img: cv2.typing.MatLike, 
                           orientation_field: cv2.typing.MatLike,
                           weights: cv2.typing.MatLike,
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
    if img is None:
        img_color = np.ones((orientation_field.shape[0] * step, orientation_field.shape[1] * step))
    else:
        img_color = img.copy()
        if img_color.shape[2] == 1:
            img_color = np.repeat(img_color, 3, axis=2)

    rows, cols = orientation_field.shape
    for y in range(0, rows):
        for x in range(0, cols):
            if weights[y, x] == 0:
                continue

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

def get_fingerprint_data(img: cv2.typing.MatLike, 
                         binarize_box_size: int = 16, 
                         gradient_blur_size: int = 5, 
                         gradient_blur_power: float = 5,
                         poincare_index_min_reliability: float = 0.68,
                         poincare_close_error: float = 0,
                         show_debug_graphics: bool = False) -> tuple[cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike, np.ndarray | None, np.ndarray | None, float, list[dict]]:
    """
    Returns:
        core_point, delta_point, core_angle, minutae
    """
    
    # NORMALIZE FINGERPRINT
    normalized = normalize_img(img)

    # CALCULATE ORIENTATION FIELD
    orientation_field, reliability_map = gradient_orientation_field(normalized, 3, gradient_blur_size, gradient_blur_power)

    # GET LARGEST RELIABLE REGION
    contour = get_largest_reliable_region(reliability_map, 0.3)
    start_y, end_y, start_x, end_x = get_reliable_region_rectangle(contour)

    contour[..., 0] -= start_x
    contour[..., 1] -= start_y

    normalized = normalized[start_y:end_y, start_x:end_x]
    orientation_field = orientation_field[start_y:end_y, start_x:end_x]
    reliability_map = reliability_map[start_y:end_y, start_x:end_x]

    # BINARIZE FINGERPRINT
    binarized = binarize_img(normalized, binarize_box_size)

    # SKELETONIZE FINGERPRINT
    skeleton = skeletonize_img(binarized)

    mask = reliability_map > 0
    orientation_field = np.where(mask, orientation_field, 0)
    skeleton = np.where(mask, skeleton, 0)

    # POINCARE INDEX
    core_point, delta_point = poincare_index(orientation_field, reliability_map, get_reliable_region_border(contour, 40), poincare_index_min_reliability, poincare_close_error)

    # MINUTAE
    minutiae = extract_minutiae(skeleton, reliability_map, orientation_field, get_reliable_region_border(contour, 16), 16, 0.5)

    # NORMALIZE MINUTAE
    core_angle = get_point_mean_angle(core_point, orientation_field, 5)
    normalized_minutiae = normalize_minutiae(minutiae, core_point, core_angle)

    # POLYMONIAL MODEL OF ORIENTATION FIELD
    # PR, PI = polymonial_orientation_field(orientation_field, reliability_map, 4)

    # POINTS CHARGES
    # cores_charges = get_points_charges(orientation_field, PR, PI, reliability_map, cores_mask, 80)
    # deltas_charges = get_points_charges(orientation_field, PR, PI, reliability_map, deltas_mask, 40)

    # POINT CHARGE
    # final_O = point_charge_orientation_field(PR, PI, orientation_field, cores_mask, deltas_mask, cores_charges, deltas_charges, 80, 40)

    if show_debug_graphics:
        show_img(img, "Original")

        show_img(normalized, "Normalized")

        show_img(binarized, "Binarized")

        show_img(skeleton, "Skeleton")

        show_img(reliability_map, "Reliability")

        skeleton_color = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2RGB)
        cv2.drawContours(skeleton_color, [contour], -1, (0, 255, 0), thickness=1)
        show_img(skeleton_color, "Contour")

        small_orientation_field, small_weight = average_orientation_field(orientation_field, reliability_map, block_size=8)
        overlay = draw_orientation_field(None, small_orientation_field, small_weight, step=8, line_length=6)
        show_img(overlay, title="Orientation Field")

        endings_mask = [m[MINUTIAE_POS] for m in minutiae if m[MINUTIAE_TYPE] == MINUTIAE_ENDING]
        bifurcation_mask = [m[MINUTIAE_POS] for m in minutiae if m[MINUTIAE_TYPE] == MINUTIAE_BIFURCATION] 

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(skeleton, cmap='gray')
        if core_point is not None:
            plt.scatter(core_point[1], core_point[0], color='blue', label='Core', s=15)
        if delta_point is not None:
            plt.scatter(delta_point[1], delta_point[0], color='red', label='Delta', s=15)
        plt.legend()
        plt.title("Singular Points (Poincare Index)")
        # plt.tight_layout()
        # plt.axis("equal")

        plt.subplot(1, 2, 2)
        plt.imshow(skeleton, cmap='gray')
        plt.scatter(np.array(endings_mask)[..., 1], np.array(endings_mask)[..., 0], color='green', label='Ending', s=15)
        plt.scatter(np.array(bifurcation_mask)[..., 1], np.array(bifurcation_mask)[..., 0], color='violet', label='Bifurcation', s=15)
        plt.legend()
        plt.title("Minutiae")
        # plt.tight_layout()
        # plt.axis("equal")

        plt.show()

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return normalized_minutiae, core_point, delta_point, core_angle

TEMPLATE_MINUTIAE = 'minutiae'
TEMPLATE_CORE = 'core'
TEMPLATE_DELTA = 'delta'
TEMPLATE_ANGLE = 'angle'

def create_fingerprint_template(fingerprint_file_path: str) -> dict:
    minutiae, core_point, delta_point, core_angle = get_fingerprint_data(load_img(fingerprint_file_path), 16, 5, 5, 0.5, 0.4 * np.pi)
    
    template = {
        TEMPLATE_MINUTIAE: [
            {
                MINUTIAE_POS: np.round(m[MINUTIAE_POS], 2).tolist(),
                MINUTIAE_ANGLE: np.round(m[MINUTIAE_ANGLE], 4),
                MINUTIAE_RELIABILITY: np.round(m[MINUTIAE_RELIABILITY], 4),
                MINUTIAE_TYPE: m[MINUTIAE_TYPE]
            } for m in minutiae
        ]
    }
    
    if core_point is not None:
        template[TEMPLATE_CORE] = np.round(core_point, 2).tolist()
        
    if delta_point is not None:
        template[TEMPLATE_DELTA] = np.round(delta_point, 2).tolist()
        
    template[TEMPLATE_ANGLE] = np.round(core_angle, 4)
    
    return template

def get_data_from_fingerprint_template(template: dict) -> tuple[list[dict], np.ndarray | None, np.ndarray | None, float]:
    minutiae = [
        {
            MINUTIAE_POS: np.array(m[MINUTIAE_POS]),
            MINUTIAE_ANGLE: m[MINUTIAE_ANGLE],
            MINUTIAE_RELIABILITY: m[MINUTIAE_RELIABILITY],
            MINUTIAE_TYPE: m[MINUTIAE_TYPE]
        } for m in template[TEMPLATE_MINUTIAE]
    ]

    core_point = template.get(TEMPLATE_CORE)
    core_point = core_point if core_point is None else np.array(core_point)

    delta_point = template.get(TEMPLATE_DELTA)
    delta_point = delta_point if delta_point is None else np.array(delta_point)

    return minutiae, core_point, delta_point, template[TEMPLATE_ANGLE]

def create_fingerprint_templates_collection(fingerprints_paths: list[str]) -> list[dict]:
    templates = []
    
    for path in fingerprints_paths:            
        templates.append(create_fingerprint_template(path))
        
    return templates

def save_fingerprint_templates_collection(save_path: str, templates_collection: list[dict]) -> None:
    with open(save_path, 'w') as f:
        json.dump(templates_collection, f, indent=4)

def load_fingerprint_templates_collection(file_path: str) -> list[dict]:
    with open(file_path, 'r') as f:
        data = json.load(f)  
    return data

def create_and_save_fingerprint_templates_collection(fingerprints_paths: list[str], save_path: str) -> list[dict]:
    template = create_fingerprint_templates_collection(fingerprints_paths)
    save_fingerprint_templates_collection(save_path, template)
    return template
    
def compare_minutiae(minutiae_A: dict, minutiae_B: dict, dist_threshold: int = 15, angle_threshold: float = 0.26) -> bool:
    if minutiae_A[MINUTIAE_TYPE] != minutiae_B[MINUTIAE_TYPE]:
        return False
    
    dp = minutiae_A[MINUTIAE_POS] - minutiae_B[MINUTIAE_POS]
    dist = np.sqrt(dp[0] ** 2 + dp[1] ** 2)
    if dist > dist_threshold:
        return False
    
    dtheta = np.abs(minutiae_A[MINUTIAE_ANGLE] - minutiae_B[MINUTIAE_ANGLE]) % np.pi
    dtheta = min(dtheta, np.pi - dtheta)
    if dtheta > angle_threshold:
        return False
    
    return True

def compare_minutiae_sets(minutiae_set_A: list[dict], minutiae_set_B: list[dict], 
                          dist_threshold: int = 15, angle_threshold: float = 0.26) -> int:
    matched = 0
    used = set()

    for m_A in minutiae_set_A:
        for i, m_B in enumerate(minutiae_set_B):
            if i in used:
                continue
            
            if compare_minutiae(m_A, m_B, dist_threshold, angle_threshold):
                matched += 1
                used.add(i)
                break
    
    return matched