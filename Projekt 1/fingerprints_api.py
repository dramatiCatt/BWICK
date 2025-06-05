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

    cos2O = np.cos(2 * orientation_field)
    sin2O = np.sin(2 * orientation_field)

    cos2O = cv2.GaussianBlur(cos2O, (blur_kernel_size, blur_kernel_size), sigmaX=blur_power)
    sin2O = cv2.GaussianBlur(sin2O, (blur_kernel_size, blur_kernel_size), sigmaX=blur_power)

    # orientation_field = cv2.GaussianBlur(orientation_field, (blur_kernel_size, blur_kernel_size), sigmaX=blur_power)
    orientation_field = 0.5 * np.arctan2(sin2O, cos2O)

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
                                                          lambda dy, r: dy / r, lambda dx, r: -dx / r)
    deltas_charges, deltas_weights = calculate_point_charge(Ys, Xs, rows, columns, deltas_mask, deltas_electricity, deltas_R, 
                                                            lambda dy, r: -dy / r, lambda dx, r: -dx / r)

    points_charges = [*cores_charges, *deltas_charges]
    points_weights = [*cores_weights, *deltas_weights]

    model_weights = np.maximum(1 - np.sum(points_weights, axis=0), 0)

    show_img(model_weights, "Model Weights")

    weights_stack = np.stack(points_weights, axis=0)
    charges_stack = np.stack(points_charges, axis=0)

    combined = weights_stack[..., np.newaxis] * charges_stack
    U = model_weights[..., np.newaxis] * PR_PI + np.sum(combined, axis=0)

    show_img(U[..., 0], "U RE")
    show_img(U[..., 1], "U IM")

    return 0.5 * np.arctan2(U[..., 1], U[..., 0])

def extract_minutiae(skeleton):
    """
    Ekstrahuje punkty minutiae (zakończenia i rozwidlenia) z binarnego obrazu szkieletowego.

    Parametry:
        skeleton (np.ndarray): Obraz szkieletu (wartości 0 lub 255)

    Zwraca:
        List[Dict]: Lista punktów minutiae w postaci słowników {'x', 'y', 'type'}
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
            if skel[y, x] == 1:
                count = neighbor_count[y, x]
                if count == 1:
                    minutiae.append({'x': x, 'y': y, 'type': 'ending'})
                elif count == 3:
                    minutiae.append({'x': x, 'y': y, 'type': 'bifurcation'})

    return minutiae

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
                         poincare_index_min_weight: float = 0.68,
                         poincare_close_error: float = 0) -> tuple[cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike]:
    """
    Returns:
        binarized, skeleton, orientation_field, weight, cores_mask, deltas_mask
    """
    
    # NORMALIZE FINGERPRINT
    normalized = normalize_img(img)

    # BINARIZE FINGERPRINT
    binarized = binarize_img(normalized, binarize_box_size)

    # SKELETONIZE FINGERPRINT
    skeleton = skeletonize_img(binarized)

    # CALCULATE ORIENTATION FIELD
    orientation_field, weight = gradient_orientation_field(normalized, 3, gradient_blur_size, gradient_blur_power)

    mask = weight > 0
    orientation_field = np.where(mask, orientation_field, 0)
    skeleton = np.where(mask, skeleton, 0)

    # POINCARE INDEX
    cores_mask, deltas_mask = poincare_index(orientation_field, weight, weights_min_power=poincare_index_min_weight, close_error=poincare_close_error)

    # POLYMONIAL MODEL OF ORIENTATION FIELD
    # PR, PI = fpa.polymonial_orientation_field(fingerprint_O, fingerprint_W, 4)

    # fingerprint_PO = 0.5 * np.arctan2(PI, PR)

    # POINTS CHARGES
    # cores_charges = fpa.get_points_charges(fingerprint_O, PR, PI, fingerprint_W, cores_mask, 80)
    # deltas_charges = fpa.get_points_charges(fingerprint_O, PR, PI, fingerprint_W, deltas_mask, 40)

    # POINT CHARGE
    # final_O = fpa.point_charge_orientation_field(PR, PI, fingerprint_O, cores_mask, deltas_mask, cores_charges, deltas_charges, 80, 40)

    return binarized, skeleton, orientation_field, weight, cores_mask, deltas_mask