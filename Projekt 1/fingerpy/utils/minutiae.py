from timer import timer
import cv2
import numpy as np
from .math import get_point_mean_angle, farange, nearest_value_idx, euclidean_distance, angle_berween_points
from scipy.spatial import KDTree
from typing import Self
import numba
from collections import deque

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
        self._ridge_id = -1

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
    def x(self) -> float:
        return self._pos[1]
    
    @property
    @timer
    def y(self) -> float:
        return self._pos[0]

    @property
    @timer
    def angle(self) -> float:
        return self._angle

    @property
    @timer
    def type_name(self):
        return self._type_name

    @property
    @timer
    def ridge_id(self) -> int:
        return self._ridge_id

    @ridge_id.setter
    @timer
    def ridge_id(self, value: int) -> int:
        self._ridge_id = value
        return self._ridge_id

@timer
def count_neightbours(skeleton: cv2.typing.MatLike) -> np.ndarray:
    # Definiujemy kernel do zliczania sąsiadów (bez środka)
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)

    # Zastosuj konwolucję w celu zliczenia aktywnych sąsiadów
    return cv2.filter2D(skeleton, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_CONSTANT)

@timer
def get_neightbours_coords(skeleton: cv2.typing.MatLike) -> tuple[np.ndarray, np.ndarray]:
    NEIGHBOR_OFFSETS = [
        (-1, -1), (-1, 0), (-1, 1),
        ( 0, -1),          ( 0, 1),
        ( 1, -1), ( 1, 0), ( 1, 1)
    ]

    height, width = skeleton.shape
    Y_grid, X_grid = np.indices(skeleton.shape)

    all_potential_neightbors = np.full((height, width, len(NEIGHBOR_OFFSETS), 2), -1)

    for i, (dy, dx) in enumerate(NEIGHBOR_OFFSETS):
        ny_potential = Y_grid + dy
        nx_potential = X_grid + dx

        all_potential_neightbors[:, :, i, 0] = ny_potential
        all_potential_neightbors[:, :, i, 1] = nx_potential

    mask_y_valid = (all_potential_neightbors[:, :, :, 0] >= 0) & (all_potential_neightbors[:, :, :, 0] < height)
    mask_x_valid = (all_potential_neightbors[:, :, :, 1] >= 0) & (all_potential_neightbors[:, :, :, 1] < width)

    mask_valid_coords = mask_y_valid & mask_x_valid
    
    return all_potential_neightbors, mask_valid_coords

@timer
def trace_ridge(start_y: int, start_x: int, max_length: int,
                skeleton: cv2.typing.MatLike,
                neightbours_coords: np.ndarray, neightbours_count: np.ndarray,
                neightbours_valid_mask: np.ndarray) -> list[tuple[int, int]]:
    path = []
    current_y, current_x = start_y, start_x
    previous_y, previous_x = -1, -1

    if skeleton[current_y, current_x] == 0:
        return path

    if neightbours_count[current_y, current_x] == 0 and max_length > 0:
        path.append((current_y, current_x))
        return path

    for _ in range(max_length):
        path.append((current_y, current_x))
        found_next = False
        
        neightbours_for_pixel = neightbours_coords[current_y, current_x]
        mask_for_pixel = neightbours_valid_mask[current_y, current_x]

        for i in np.where(mask_for_pixel)[0]:
            ny, nx = neightbours_for_pixel[i]

            if skeleton[ny, nx] == 0:
                continue

            if ny == previous_y and nx == previous_x:
                continue

            if neightbours_count[ny, nx] > 2:
                continue

            previous_y, previous_x = current_y, current_x
            current_y, current_x = ny, nx
            found_next = True
            break
            
        if not found_next:
            break
    
    return path

@timer
def find_all_ridge_pixels(start_y: int, start_x: int, visited: np.ndarray, 
                          skeleton: cv2.typing.MatLike,
                          neightbours_coords: np.ndarray, 
                          neightbours_valid_mask: np.ndarray) -> list[tuple[int, int]]:
    q = deque([(start_y, start_x)])
    visited[start_y, start_x] = True
    ridge_pixels = [(start_y, start_x)]

    while q:
        cy, cx = q.popleft()
        
        for i in np.where(neightbours_valid_mask[cy, cx])[0]:
            ny, nx = neightbours_coords[cy, cx, i]
            
            if skeleton[ny, nx] == 0:
                continue

            if visited[ny, nx]:
                continue
            
            visited[ny, nx] = True
            q.append((ny, nx))
            ridge_pixels.append((ny, nx))
    return ridge_pixels

@timer
def fill_hole(skeleton: cv2.typing.MatLike, max_hole_size: int) -> tuple[cv2.typing.MatLike, list[tuple[int, int]]]:
    modified_skeleton = skeleton.copy()

    filled_pixels = []

    height, width = skeleton.shape

    visited_background = np.zeros_like(skeleton, dtype=bool)

    NEIGHBOR_OFFSETS = [
        (-1, 0), (0, -1), (0, 1), (1, 0)
    ]

    for r in range(height):
        for c in range(width):
            if modified_skeleton[r, c] != 0 or visited_background[r, c]:
                continue

            current_hole_pixels = []
            q = deque([(r, c)])
            visited_background[r, c] = True

            is_closed_hole = True

            while q:
                cy, cx = q.popleft()
                current_hole_pixels.append((cy, cx))

                for dy, dx in NEIGHBOR_OFFSETS:
                    ny, nx = cy + dy, cx + dx

                    if not (0 <= ny < height and 0 <= nx < width):
                        is_closed_hole = False
                        break

                    if modified_skeleton[ny, nx] != 0 or visited_background[ny, nx]:
                        continue

                    visited_background[ny, nx] = True
                    q.append((ny, nx))
                
                if not is_closed_hole:
                    break

            if not is_closed_hole and len(current_hole_pixels) > max_hole_size:
                continue

            for hy, hx in current_hole_pixels:
                modified_skeleton[hy, hx] = 1
                filled_pixels.append((hy, hx))

    return modified_skeleton, filled_pixels

@timer
def postprocess_minutiae(minutiae_list: list[Minutiae], skeleton: cv2.typing.MatLike, min_ridge_length: int = 15,
                         max_bridge_dist: int = 15, min_distance_between_minutiae: int = 15) -> list[Minutiae]:
    
    minutiae_to_remove = set()

    # 0. Ridges Map
    ridge_map = np.zeros_like(skeleton, dtype=int)
    current_ridge_id = 1

    neightbours_count = count_neightbours(skeleton)
    neightbours_coords, neightbours_valid_mask = get_neightbours_coords(skeleton)
    
    visited_pixels = np.zeros_like(skeleton, dtype=bool)

    height, width = skeleton.shape

    for r in range(height):
        for c in range(width):
            if skeleton[r, c] != 1 or visited_pixels[r, c]:
                continue

            ridge_pixels = find_all_ridge_pixels(r, c, visited_pixels, skeleton, neightbours_coords, neightbours_valid_mask)
                
            for py, px in ridge_pixels:
                ridge_map[py, px] = current_ridge_id
            
            current_ridge_id += 1
    
    for m in minutiae_list:
        m.ridge_id = ridge_map[int(m.y), int(m.x)]

    # 1. Spur Removal
    for m in minutiae_list:
        if m.type_name != MINUTIAE_ENDING:
            continue

        path = trace_ridge(int(m.y), int(m.x), min_ridge_length + 1, skeleton, neightbours_coords, neightbours_count, neightbours_valid_mask)

        if len(path) <= min_ridge_length:
            minutiae_to_remove.add(m)

        # Dodatkowo, jeśli punkt końcowy jest bardzo blisko innego punktu końcowego,
        # i kąty wskazują na przerwę w grzbiecie, można je połączyć.
        # To jest bardziej złożone i wymagałoby śledzenia kierunku grzbietu od obu punktów.
        # Na razie uproszczenie: jeśli jest to krótki grzbiet, usuwamy.

    # 2. Double Minutiae and Bridge Removal
    for i, m1 in enumerate(minutiae_list):
        if m1 in minutiae_to_remove:
            continue

        for j in range(i + 1, len(minutiae_list)):
            m2 = minutiae_list[j]

            if m2 in minutiae_to_remove:
                continue
            
            dist = euclidean_distance(m1.x, m1.y, m2.x, m2.y)

            if m1.type_name == m2.type_name and dist < min_distance_between_minutiae:
                minutiae_to_remove.add(m2)
                continue

            if dist >= max_bridge_dist:
                continue

            if m1.type_name == MINUTIAE_ENDING and m2.type_name == MINUTIAE_ENDING and \
                m1.ridge_id != -1 and m2.ridge_id != -1 and m1.ridge_id != m2.ridge_id:
                
                line_angle = angle_berween_points(m1.x, m1.y, m2.x, m2.y)

                angle_diff1 = abs(m1.angle - line_angle)
                angle_diff2 = abs(m2.angle - line_angle)

                angle_diff1 = min(angle_diff1, np.pi - angle_diff1)
                angle_diff2 = min(angle_diff2, np.pi - angle_diff2)

                angle_threshold_for_bridge = np.pi / 4

                if (abs(angle_diff1 - np.pi / 2) < angle_threshold_for_bridge and
                    abs(angle_diff2 - np.pi / 2) < angle_threshold_for_bridge):

                    minutiae_to_remove.add(m1)
                    minutiae_to_remove.add(m2)

    # New Minutiae List
    processed_minutiae = []
    for m in minutiae_list:
        if m in minutiae_to_remove:
            continue
        processed_minutiae.append(m)

    return processed_minutiae

@timer
def extract_minutiae(skeleton: cv2.typing.MatLike, reliability_map: cv2.typing.MatLike,
                     orientation_field: cv2.typing.MatLike, 
                     border_contour: np.ndarray | None = None, border: int = 10,
                     reliability_threshold: float = 0.5, max_hole_size: int = 10) -> list[Minutiae]:
    """
    Ekstrahuje punkty minutiae (zakończenia i rozwidlenia) z binarnego obrazu szkieletowego.

    Parametry:
        skeleton (np.ndarray): Obraz szkieletu (wartości 0 lub 255)

    Zwraca:
        List[Dict]: Lista punktów minutiae w postaci słowników {'pos', 'type', 'orientation', 'reliability'}
    """

    # Upewnij się, że mamy wartości 0 i 1 (a nie 0 i 255)
    skel = (skeleton > 0).astype(np.uint8)

    # skel, _ = fill_hole(skel, max_hole_size)

    neighbor_count = count_neightbours(skel)

    minutiae: list[Minutiae] = []

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

    return postprocess_minutiae(minutiae, skel, 15, 15, 15)

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
def _get_best_transform_Stolarek(m1_x_coords: np.ndarray, m1_y_coords: np.ndarray, m1_angles: np.ndarray,
                                           m2_x_coords: np.ndarray, m2_y_coords: np.ndarray, m2_angles: np.ndarray,
                                           angle_threshold: float,
                                           delta_x_array: np.ndarray, delta_y_array: np.ndarray, angle_array: np.ndarray,
                                           cos_angles_grid: np.ndarray, sin_angles_grid: np.ndarray) -> tuple[float, float, float]:
    A = np.zeros(shape=(len(delta_x_array), len(delta_y_array), len(angle_array)))

    offsets = np.array([-1, 0, 1], dtype=np.int64)

    ox, oy, oa = np.meshgrid(offsets, offsets, offsets, indexing='ij')

    ox_flat = ox.flatten()
    oy_flat = oy.flatten()
    oa_flat = oa.flatten()

    num_offsets = len(offsets)**3

    all_indices_to_increment_x = []
    all_indices_to_increment_y = []
    all_indices_to_increment_angle = []

    for idx1, m1_angle in enumerate(m1_angles):
        m1_x = m1_x_coords[idx1]
        m1_y = m1_y_coords[idx1]
        
        for idx2, m2_angle in enumerate(m2_angles):
            m2_x = m2_x_coords[idx2]
            m2_y = m2_y_coords[idx2]

            angle_diff_raw = np.abs(m2_angle + angle_array - m1_angle)
            angle_diff = np.minimum(angle_diff_raw, 2.0 * np.pi - angle_diff_raw)

            valid_angle_indices = np.where(angle_diff < angle_threshold)[0]

            if valid_angle_indices.size == 0:
                continue

            current_cos_thetas = cos_angles_grid[valid_angle_indices]
            current_sin_thetas = sin_angles_grid[valid_angle_indices]

            transformed_m2_x = current_cos_thetas * m2_x - current_sin_thetas * m2_y
            transformed_m2_y = current_sin_thetas * m2_x + current_cos_thetas * m2_y

            delta_xs = m1_x - transformed_m2_x
            delta_ys = m1_y - transformed_m2_y

            delta_x_idx_batch = np.array([nearest_value_idx(delta_x_array, dx) for dx in delta_xs])
            delta_y_idx_batch = np.array([nearest_value_idx(delta_y_array, dy) for dy in delta_ys])

            if delta_x_idx_batch.size > 0:
                for i in range(delta_x_idx_batch.size):
                    all_indices_to_increment_x.append(delta_x_idx_batch[i])
                    all_indices_to_increment_y.append(delta_y_idx_batch[i])
                    all_indices_to_increment_angle.append(valid_angle_indices[i])

                    for offset_idx in range(num_offsets):
                        n_dx = delta_x_idx_batch[i] + ox_flat[offset_idx]
                        n_dy = delta_y_idx_batch[i] + oy_flat[offset_idx]
                        n_angle = valid_angle_indices[i] + oa_flat[offset_idx]

                        if (0 <= n_dx < len(delta_x_array) and
                            0 <= n_dy < len(delta_y_array) and
                            0 <= n_angle < len(angle_array)):
                            all_indices_to_increment_x.append(n_dx)
                            all_indices_to_increment_y.append(n_dy)
                            all_indices_to_increment_angle.append(n_angle)

    np.add.at(A, (all_indices_to_increment_x, all_indices_to_increment_y, all_indices_to_increment_angle), 1)

    if A.sum() == 0:
        return np.nan, np.nan, np.nan
    
    best_transform_indices = np.unravel_index(np.argmax(A), A.shape)

    best_delta_x = delta_x_array[best_transform_indices[0]]
    best_delta_y = delta_y_array[best_transform_indices[1]]
    best_angle = angle_array[best_transform_indices[2]]

    return best_delta_x, best_delta_y, best_angle

@timer
def transform_minutiae_set_Stolarek(minutiae_set: list[Minutiae], target_minutiae_set: list[Minutiae], 
                                    angle_threshold: float = 0.24 * np.pi, 
                                    delta_x_limit: float = 5, delta_x_step: float = 1,
                                    delta_y_limit: float = 5, delta_y_step: float = 1,
                                    angle_limit: float = 0.08 * np.pi, angle_step: float = 0.006 * np.pi) -> list[Minutiae]:
    delta_x_array = farange(-delta_x_limit, delta_x_limit, delta_x_step)
    delta_y_array = farange(-delta_y_limit, delta_y_limit, delta_y_step)
    angle_array = farange(-np.abs(angle_limit), np.abs(angle_limit), angle_step)

    m1_x_coords = np.array([m.pos[1] for m in minutiae_set])
    m1_y_coords = np.array([m.pos[0] for m in minutiae_set])
    m1_angles = np.array([m.angle for m in minutiae_set])

    m2_x_coords = np.array([m.pos[1] for m in target_minutiae_set])
    m2_y_coords = np.array([m.pos[0] for m in target_minutiae_set])
    m2_angles = np.array([m.angle for m in target_minutiae_set])

    cos_angles_grid = np.cos(angle_array)
    sin_angles_grid = np.sin(angle_array)

    best_delta_x, best_delta_y, best_angle = _get_best_transform_Stolarek(
        m1_x_coords, m1_y_coords, m1_angles,
        m2_x_coords, m2_y_coords, m2_angles,
        angle_threshold,
        delta_x_array, delta_y_array, angle_array,
        cos_angles_grid, sin_angles_grid
    )

    if np.isnan(best_delta_x):
        print("Brak pasujących transformacji. Zwracam kopię oryginalnego zestawu")
        return list(minutiae_set)

    transformed_set = []
    for m in minutiae_set:
        new_x = m.pos[1] + best_delta_x
        new_y = m.pos[0] + best_delta_y
        new_angle = m.angle + best_angle

        transformed_set.append(
            Minutiae(
                np.array([new_y, new_x]),
                new_angle,
                m.type_name
            )
        )
    
    return transformed_set

    # for m1 in minutiae_set:
    #     for m2 in target_minutiae_set:
    #         for angle_idx, angle in enumerate(angle_array):
    #             angle_diff = abs(m2.angle + angle - m1.angle)
    #             angle_diff = min(angle_diff, 2.0 * np.pi - angle_diff)
                
    #             if angle_diff >= angle_threshold:
    #                 continue

    #             cos_theta = np.cos(angle)
    #             sin_theta = np.sin(angle)

    #             delta_x = m1.pos[1] - (cos_theta * m2.pos[1] - sin_theta * m2.pos[0])
    #             delta_y = m1.pos[0] - (sin_theta * m2.pos[1] + cos_theta * m2.pos[0])

    #             near_delta_x = nearest_value(delta_x_array, delta_x)
    #             near_delta_y = nearest_value(delta_y_array, delta_y)

    #             delta_x_idx = np.where(delta_x_array == near_delta_x)[0]
    #             delta_y_idx = np.where(delta_y_array == near_delta_y)[0]

    #             A[delta_x_idx, delta_y_idx, angle_idx] += 1
    #             for i in range(-1, 2, 1):
    #                 for j in range(-1, 2, 1):
    #                     for k in range(-1, 2, 1):
    #                         i_idx = delta_x_idx + i
    #                         if i_idx < 0 or i_idx >= len(delta_x_array):
    #                             continue

    #                         j_idx = delta_y_idx + j
    #                         if j_idx < 0 or j_idx >= len(delta_y_array):
    #                             continue

    #                         k_idx = angle_idx + k
    #                         if k_idx < 0 or k_idx >= len(angle_array):
    #                             continue
                            
    #                         A[i_idx, j_idx, k_idx] += 1
    
    # best_transform = np.argmax(A, keepdims=True)

    # best_delta_x = delta_x_array[best_transform[0]]
    # best_delta_y = delta_y_array[best_transform[1]]
    # best_angle = angle_array[best_transform[2]]

    # transformed_set = []

    # for m in minutiae_set:
    #     new_x = m.pos[1] + best_delta_x
    #     new_y = m.pos[0] + best_delta_y
    #     new_angle = m.angle + best_angle

    #     transformed_set.append(
    #         Minutiae(
    #             np.ndarray([new_y, new_x]),
    #             new_angle,
    #             m.type_name
    #         )
    #     )

    # return transformed_set