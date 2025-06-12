from timer import timer
import numpy as np
import cv2
from typing import Callable
import numba

@timer
@numba.njit
def get_point_influence(Xs: np.ndarray, Ys: np.ndarray, xc: np.ndarray, yc: np.ndarray, max_R: float) -> np.ndarray:
    r = np.sqrt((Xs - xc)**2 + (Ys - yc)**2) + 1e-10
    return 1 - np.minimum(r, max_R) / max_R

@timer
@numba.njit
def get_point_charge(Xs: np.ndarray, Ys: np.ndarray, orientation_field: cv2.typing.MatLike, 
                     PR: cv2.typing.MatLike, PI: cv2.typing.MatLike, reliability_map: cv2.typing.MatLike, 
                     xc: np.ndarray, yc: np.ndarray, max_R: float) -> float:
    phi = get_point_influence(Xs, Ys, xc, yc, max_R)
    diffrence = (PR - np.cos(2 * orientation_field)) ** 2 + (PI - np.sin(2 * orientation_field)) ** 2

    num = np.sum(reliability_map * phi * diffrence)
    denom = np.sum(reliability_map * phi ** 2)

    if denom == 0:
        return 0.0
    return num / denom

@timer
def get_points_charges(orientation_field: cv2.typing.MatLike, PR: cv2.typing.MatLike, PI: cv2.typing.MatLike, 
                       reliability_map: cv2.typing.MatLike, points_mask: cv2.typing.MatLike, max_R: float) -> np.ndarray:
    rows, columns = orientation_field.shape

    Xs, Ys = np.meshgrid(np.arange(columns), np.arange(rows))
    
    points = np.argwhere(points_mask)

    charges = np.empty(len(points), dtype=np.float64)
    charge_idx = 0
    for xc, yc in points:
        circle_mask = (Xs - xc)**2 + (Ys - yc)**2 <= max_R**2
        
        indexes = np.nonzero(circle_mask)

        charges[charge_idx] = get_point_charge(Xs[indexes], Ys[indexes], 
                             orientation_field[indexes], 
                             PR[indexes],
                             PI[indexes], 
                             reliability_map[indexes], xc, yc, max_R)
        charge_idx += 1
    return charges

@timer
def calculate_point_charge(Xs: np.typing.ArrayLike,
                           Ys: np.typing.ArrayLike, 
                           rows: int,
                           columns: int, 
                           points_mask: cv2.typing.MatLike, 
                           points_electricity,
                           points_R: float, 
                           re_charge_func: Callable[[float, float], float], 
                           im_charge_func: Callable[[float, float], float]) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        charges, weights
    """
    points = np.argwhere(points_mask)

    points_charges = np.empty((len(points), rows, columns, 2), dtype=np.float64)
    points_weights = np.empty(len(points), dtype=np.float64)

    point_idx = 0
    for px, py in points:
        dy = Ys - py
        dx = Xs - px
        r = np.sqrt(dx**2 + dy**2) + 1e-10

        mask = r <= points_R

        point_PC = np.zeros((rows, columns, 2))
        point_PC[..., 0] = np.where(mask, re_charge_func(dy, r) * points_electricity[len(points_charges)], 0)
        point_PC[..., 1] = np.where(mask, im_charge_func(dx, r) * points_electricity[len(points_charges)], 0)

        point_weight = 1 - np.minimum(r, points_R) / points_R

        points_charges[point_idx] = point_PC
        points_weights[point_idx] = point_weight
        
        point_idx += 1

    return points_charges, points_weights
