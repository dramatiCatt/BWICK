from timer import timer
import cv2
import numpy as np

@timer
def generate_gabor_kernels(ksize=21, sigma=5.0, lambd=10.0, gamma=0.5, psi=0, num_angles=16):
    angles = np.linspace(0, np.pi, num_angles, endpoint=False)
    kernels = []
    for theta in angles:
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_64F)
        kernels.append(kernel)
    return angles, kernels

@timer
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

@timer
def build_polymonial_basis(Xs: np.typing.ArrayLike, Ys: np.typing.ArrayLike, degree: int = 4) -> cv2.typing.MatLike:
    x = Xs.flatten()
    y = Ys.flatten()

    terms = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            terms.append((x ** i) * (y ** j))

    return np.stack(terms, axis=1)

@timer
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

@timer
def eval_polymonial(Xs, Ys, coeffs, degree):
    A = build_polymonial_basis(Xs, Ys, degree)
    result = A @ coeffs
    return result.reshape((-1,))

@timer
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
