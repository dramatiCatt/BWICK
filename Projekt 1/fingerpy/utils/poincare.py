from timer import timer
import numpy as np
import cv2
import typing
from .math import filter_clusters
import numba

@timer
def filter_poincare_points(points_mask: cv2.typing.MatLike, 
                           points_scores: cv2.typing.MatLike,
                           reliability_map: cv2.typing.MatLike,
                           contour_border: np.ndarray | None = None,
                           min_reliability: float = 0.3,
                           border_size: int = 16,
                           near_radius: int = 20,
                           points_score_offset: float = 0.0) -> cv2.typing.MatLike:

    filtered_points_mask = points_mask & (reliability_map > min_reliability)

    filtered_points_mask[:border_size, :] = False
    filtered_points_mask[-border_size:, :] = False
    filtered_points_mask[:, :border_size] = False
    filtered_points_mask[:, -border_size:] = False

    points_yx = np.argwhere(filtered_points_mask)
    best_points_score = np.min(points_scores[filtered_points_mask])
    for py, px in points_yx:
        filtered_points_mask[py, px] = points_scores[py, px] <= best_points_score + points_score_offset

    points_yx = np.argwhere(filtered_points_mask)
    points_reliability = reliability_map[filtered_points_mask]
    filtered_points = filter_clusters(points_yx, strengths=points_reliability, radius=near_radius)
    filtered_cluster_points_mask = np.zeros_like(filtered_points_mask, dtype=bool)
    for y, x in filtered_points:
        filtered_cluster_points_mask[int(y), int(x)] = True
    filtered_points_mask = filtered_cluster_points_mask

    if contour_border is not None:
        points_yx = np.argwhere(filtered_points_mask)
        for py, px in points_yx:
            inside = cv2.pointPolygonTest(contour_border, (px.astype(float), py.astype(float)), False)
            filtered_points_mask[py, px] = inside >= 0

    return filtered_points_mask

@timer
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

@timer
def get_best_core(cores_mask: cv2.typing.MatLike, reliability_map: cv2.typing.MatLike, radius: int = 30) -> np.ndarray | None:
    def get_score(avg_reliability: float, pos: np.ndarray, center: np.ndarray, img_size: np.ndarray) -> float:
        dist_to_center = np.linalg.norm(pos - center)
        return avg_reliability - 0.01 * dist_to_center
    
    return get_best_poincare_point(cores_mask, reliability_map, get_score, radius)

@timer
def get_best_delta(deltas_mask: cv2.typing.MatLike, reliability_map: cv2.typing.MatLike, radius: int = 30) -> np.ndarray | None:
    def get_score(avg_reliability: float, pos: np.ndarray, center: np.ndarray, img_size: np.ndarray) -> float:
        vertical_bias = pos[1] / img_size[1]
        return avg_reliability + 0.2 * vertical_bias
    
    return get_best_poincare_point(deltas_mask, reliability_map, get_score, radius)

@timer
@numba.njit
def get_poincare_index_map(orientation_field: cv2.typing.MatLike, offsets: np.ndarray) -> np.ndarray:
    poincare_index_map = np.zeros_like(orientation_field, dtype=np.float64)

    rows, columns = orientation_field.shape

    for y in range(1, rows - 1):
        for x in range(1, columns - 1):
            angles_local = np.empty(len(offsets), dtype=orientation_field.dtype)
            for i, (dy, dx) in enumerate(offsets):
                angles_local[i] = orientation_field[y + dy, x + dx]

            diffs = np.empty(8, dtype=angles_local.dtype)
            for i in range(8):
                diffs[i] = (angles_local[(i + 1) % 8] - angles_local[i] + np.pi * 0.5) % np.pi - np.pi * 0.5

            poincare_index_map[y, x] = np.sum(diffs)
    
    return poincare_index_map

@timer
def poincare_index(orientation_field: cv2.typing.MatLike, reliability_map: cv2.typing.MatLike, 
                   contour_border: np.ndarray | None = None, min_core_reliability: float = 0.5,
                   min_delta_reliability: float = 0.6, 
                   close_error: float = 0.5 * np.pi) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Returns:
        core_point, delta_point
    """
    offsets = np.array([
        [-1, -1], 
        [ 0, -1], 
        [ 1, -1], 
        [ 1,  0], 
        [ 1,  1], 
        [ 0,  1],
        [-1,  1], 
        [-1,  0]
    ])

    poincare_index_map = get_poincare_index_map(orientation_field, offsets)

    cores_mask = poincare_index_map > np.pi - close_error
    deltas_mask = poincare_index_map < -np.pi + close_error

    cores_scores = np.abs(np.pi - poincare_index_map)
    deltas_scores = np.abs(-np.pi + poincare_index_map)

    cores_mask = filter_poincare_points(cores_mask, cores_scores, reliability_map, contour_border, min_core_reliability, 16, 40, 0.1 * np.pi)
    deltas_mask = filter_poincare_points(deltas_mask, deltas_scores, reliability_map, contour_border, min_delta_reliability, 16, 40, 0.1 * np.pi)

    core_point = get_best_core(cores_mask, reliability_map, 30)
    delta_point = get_best_delta(deltas_mask, reliability_map, 30)

    return core_point, delta_point