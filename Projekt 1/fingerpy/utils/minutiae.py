from timer import timer
import cv2
import numpy as np
from .math import get_point_mean_angle
from scipy.spatial import KDTree
from typing import Self
import numba

MINUTIAE_ENDING = 'ending'
MINUTIAE_BIFURCATION = 'bifurcation'
MINUTIAE_TYPES_LIST = [MINUTIAE_ENDING, MINUTIAE_BIFURCATION]

class Minutiae():
    MINUTIAE_POS = 'pos'
    MINUTIAE_ANGLE = 'angle'
    MINUTIAE_TYPE = 'type'

    @timer
    def __init__(self, pos: np.ndarray, angle: float, type_name: str) -> None:
        self._pos = pos
        self._angle = angle
        self._type_name = type_name

    @classmethod
    @timer
    def from_data(cls, data: dict) -> Self:
        return cls(
            np.array(data[cls.MINUTIAE_POS]),
            data[cls.MINUTIAE_ANGLE],
            data[cls.MINUTIAE_TYPE]
        )

    @timer
    def to_data(self) -> dict:
        return {
            self.MINUTIAE_POS: np.round(self._pos, 4).tolist(),
            self.MINUTIAE_ANGLE: np.round(self._angle, 4),
            self.MINUTIAE_TYPE: self._type_name
        }

    @staticmethod
    @timer
    @numba.njit
    def _normalized_numba(pos: np.ndarray, core: np.ndarray, angle: float, core_angle: float, 
                          cos_a: float, sin_a: float, scale: float) -> tuple[np.ndarray, float]:
        dp = pos - core
        new_pos = np.zeros_like(pos, dtype=np.float64)

        new_pos[1] = dp[1] * cos_a - dp[0] * sin_a
        new_pos[0] = dp[1] * sin_a + dp[0] * cos_a

        new_pos *= scale

        new_angle = (angle - core_angle) % np.pi

        return new_pos, new_angle

    @timer
    def normalized(self, core: np.ndarray, core_angle: float, cos_a: float, sin_a: float, scale: float) -> Self:
        new_pos, new_angle = self._normalized_numba(self.pos, core, self.angle, core_angle, cos_a, sin_a, scale)

        return Minutiae(
            new_pos,
            new_angle,
            self._type_name
        )

    @property
    @timer
    def pos(self) -> np.ndarray:
        return self._pos

    @property
    @timer
    def angle(self) -> float:
        return self._angle

    @property
    @timer
    def type_name(self):
        return self._type_name

@timer
def extract_minutiae(skeleton: cv2.typing.MatLike, reliability_map: cv2.typing.MatLike,
                     orientation_field: cv2.typing.MatLike, 
                     border_contour: np.ndarray | None = None, border: int = 10,
                     reliability_threshold: float = 0.5) -> list[Minutiae]:
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

            m_pos = np.array([y, x])
            minutiae.append(
                Minutiae(
                    m_pos, 
                    get_point_mean_angle(m_pos, orientation_field, 5),
                    m_type
                )
            )

    return minutiae

@timer
def normalize_minutiae(minutiae: list[Minutiae], core: np.ndarray, core_angle: float) -> list[Minutiae]:
    normalized = []
    cos_a = np.cos(-core_angle)
    sin_a = np.sin(-core_angle)

    distances = [np.sqrt((m.pos[0] - core[0]) ** 2 + (m.pos[1] - core[1]) ** 2) for m in minutiae]
    mean_dist = np.mean(distances)
    scale = 100 / mean_dist if mean_dist > 0 else 1.0

    for m in minutiae:
        normalized.append(m.normalized(core, core_angle, cos_a, sin_a, scale))
    
    return normalized

@timer
def compare_minutiae_sets(minutiae_set_A: list[Minutiae], minutiae_set_B: list[Minutiae], 
                          dist_threshold: int = 15, angle_threshold: float = 0.26) -> int:
    if not minutiae_set_A or not minutiae_set_B:
        return 0

    type_to_points_B = {}
    trees: dict[str, KDTree] = {}

    for t in MINUTIAE_TYPES_LIST:
        points = np.array([m.pos for m in minutiae_set_B if m.type_name == t])
        angles = np.array([m.angle for m in minutiae_set_B if m.type_name == t])

        if len(points) > 0:
            trees[t] = KDTree(points)
            type_to_points_B[t] = (points, angles)

    matched = 0
    used_indicies = {t: set() for t in MINUTIAE_TYPES_LIST}

    for m_A in minutiae_set_A:
        t = m_A.type_name
        if t not in trees:
            continue

        pos_A = m_A.pos
        angle_A = m_A.angle

        indicies = trees[t].query_ball_point(pos_A, dist_threshold)
        _, angles_B = type_to_points_B[t]

        for idx in indicies:
            if idx in used_indicies[t]:
                continue

            dtheta = abs(angle_A - angles_B[idx]) % np.pi
            dtheta = min(dtheta, np.pi - dtheta)

            if dtheta <= angle_threshold:
                matched += 1
                used_indicies[t].add(idx)
                break
    
    return matched

@timer
def compare_minutiae_sets_Stolarek(minutiae_set_A: list[Minutiae], minutiae_set_B: list[Minutiae], 
                                    dist_threshold: int = 15, angle_threshold: float = 0.26) -> int:
    pass