from timer import timer
import numpy as np
import cv2
from typing import Callable

@timer
def get_point_influence(Xs, Ys, xc, yc, max_R):
    r = np.sqrt((Xs - xc)**2 + (Ys - yc)**2) + 1e-10
    return 1 - np.minimum(r, max_R) / max_R

@timer
def get_point_charge(Xs, Ys, orientation_field, PR, PI, weights, xc, yc, max_R):
    phi = get_point_influence(Xs, Ys, xc, yc, max_R)
    diffrence = np.pow(PR - np.cos(2 * orientation_field), 2) + np.pow(PI - np.sin(2 * orientation_field), 2)

    num = np.sum(weights * phi * diffrence)
    denom = np.sum(weights * phi**2)

    if denom == 0:
        return 0.0
    return num / denom

@timer
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

@timer
def calculate_point_charge(Xs: np.typing.ArrayLike,
                           Ys: np.typing.ArrayLike, 
                           rows: int,
                           columns: int, 
                           points_mask: cv2.typing.MatLike, 
                           points_electricity,
                           points_R: float, 
                           re_charge_func: Callable[[float, float], float], 
                           im_charge_func: Callable[[float, float], float]) -> tuple[list, list]:
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
