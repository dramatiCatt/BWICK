from timer import timer
import cv2
import numpy as np
from .math import weight_least_square, eval_polymonial
from .point_charge import calculate_point_charge

@timer
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

@timer
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

@timer
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

@timer
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

@timer
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