from __future__ import annotations
import cv2
import matplotlib.pyplot as plt
import numpy as np
from typing import Self
import json
from timer import timer
from .utils import imgprocessing as imgp, orientation as orient, contour as ctour, poincare, minutiae as minuti, math, point_charge

class FingerprintTemplate():
    TEMPLATE_MINUTIAE = 'minutiae'
    TEMPLATE_CORE = 'core'
    TEMPLATE_DELTA = 'delta'
    TEMPLATE_ANGLE = 'angle'

    @timer
    def __init__(self, minutiae: list[minuti.Minutiae], core_point: np.ndarray | None, delta_point: np.ndarray | None, core_angle: float) -> None:
        self._minutiae = minutiae
        self._core_point = core_point
        self._delta_point = delta_point
        self._core_angle = core_angle

    @classmethod
    @timer
    def from_img(cls, img: cv2.typing.MatLike) -> Self:
        return cls.from_fingerprint(Fingerprint(img))

    @classmethod
    @timer
    def from_img_file(cls, img_path: str) -> Self:
        return cls.from_img(imgp.load_img(img_path))

    @classmethod
    @timer
    def from_fingerprint(cls, fingerprint: 'Fingerprint') -> Self:
        return fingerprint.generate_template()

    @classmethod
    @timer
    def from_data(cls, data: dict) -> Self:
        minutiae = [minuti.Minutiae.from_data(m) for m in data[cls.TEMPLATE_MINUTIAE]]

        core_point = data.get(cls.TEMPLATE_CORE)
        core_point = core_point if core_point is None else np.array(core_point)

        delta_point = data.get(cls.TEMPLATE_DELTA)
        delta_point = delta_point if delta_point is None else np.array(delta_point)

        return cls(
            minutiae,
            core_point,
            delta_point,
            data[cls.TEMPLATE_ANGLE]
        )

    @classmethod
    @timer
    def from_data_file(cls, data_path: str) -> Self:
        with open(data_path, 'r') as f:
            return cls.from_data(json.load(f))

    @timer
    def to_data(self) -> dict:
        return {
            self.TEMPLATE_MINUTIAE: [m.to_data() for m in self._minutiae],
            self.TEMPLATE_CORE: np.round(self._core_point, 4).tolist() if self._core_point is not None else None,
            self.TEMPLATE_DELTA: np.round(self._delta_point, 4).tolist() if self._delta_point is not None else None,
            self.TEMPLATE_ANGLE: np.round(self._core_angle, 4)
        }

    @property
    @timer
    def minutiae(self) -> list[minuti.Minutiae]:
        return self._minutiae
    
    @property
    @timer
    def core(self) -> np.ndarray | None:
        return self._core_point
    
    @property
    @timer
    def delta(self) -> np.ndarray | None:
        return self._delta_point
    
    @property
    @timer
    def angle(self) -> float:
        return self._core_angle

    @timer
    def show_points(self, img: cv2.typing.MatLike | None = None, stop: bool = True) -> None:
        endings_mask = [m.pos for m in self._minutiae if m.type_name == minuti.MINUTIAE_ENDING]
        bifurcation_mask = [m.pos for m in self._minutiae if m.type_name == minuti.MINUTIAE_BIFURCATION]

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        if img is not None:
            plt.imshow(img, cmap='gray')
        if self._core_point is not None:
            plt.scatter(self._core_point[1], self._core_point[0], color='blue', label='Core', s=15)
        if self._delta_point is not None:
            plt.scatter(self._delta_point[1], self._delta_point[0], color='red', label='Delta', s=15)
        plt.legend()
        plt.title("Singular Points (Poincare Index)")

        plt.subplot(1, 2, 2)
        # if img is not None:
        #     plt.imshow(img, cmap='gray')
        plt.scatter(np.array(endings_mask)[..., 1], np.array(endings_mask)[..., 0], color='green', label='Ending', s=15)
        plt.scatter(np.array(bifurcation_mask)[..., 1], np.array(bifurcation_mask)[..., 0], color='violet', label='Bifurcation', s=15)
        plt.legend()
        plt.title("Minutiae")

        plt.show(block=stop)

class Fingerprint():
    @timer
    def __init__(self, img: cv2.typing.MatLike) -> None:
        self._original_img = img
        self._preprocess()

    @classmethod
    @timer
    def from_file(cls, img_path: str) -> Self:
        return cls(imgp.load_img(img_path))
    
    @timer
    def _preprocess(self) -> None:
        # NORMALIZE FINGERPRINT
        self._normalized = imgp.normalize_img(self._original_img)
        normalized_u8 = (self._normalized * 255).astype(np.uint8)
        normalized_u8 = imgp.median_filter(normalized_u8, 3)
        normalized_u8 = imgp.gaussian_blur(normalized_u8, 5, 0.5)
        self._normalized = (normalized_u8 / 255.0).astype(np.float64)

        # CALCULATE ORIENTATION FIELD
        self._orientation_field, self._reliability_map = orient.gradient_orientation_field(self._normalized, 3, 5, 5)
        self._orientation_field[self._reliability_map <= 0] = 0

        # GET LARGEST RELIABLE REGION
        self._contour = ctour.get_largest_reliable_region(self._reliability_map, 0.3)
        start_y, end_y, start_x, end_x = ctour.get_reliable_region_rectangle(self._contour)

        # MOVE CONTOUR
        self._contour[..., 0] -= start_x
        self._contour[..., 1] -= start_y

        # CROP IMAGES
        self._normalized = self._normalized[start_y:end_y, start_x:end_x]
        self._orientation_field = self._orientation_field[start_y:end_y, start_x:end_x]
        self._reliability_map = self._reliability_map[start_y:end_y, start_x:end_x]

        # BINARIZE FINGERPRINT
        self._binarized = imgp.binarize_img(self._normalized, 16)

        # SKELETONIZE FINGERPRINT
        self._skeleton = imgp.skeletonize_img(self._binarized)
        self._skeleton[self._reliability_map <= 0] = 0

    @timer
    def generate_template(self) -> FingerprintTemplate:
        contour40 = ctour.get_reliable_region_border(self._contour, 40)
        contour16 = ctour.get_reliable_region_border(self._contour, 16)

        # POINCARE INDEX
        core_point, delta_point = poincare.poincare_index(self._orientation_field, self._reliability_map, contour40, 0.5, 0.6, 0.2 * np.pi)

        # MINUTAE
        minutiae = minuti.extract_minutiae(self._skeleton, self._reliability_map, self._orientation_field, contour16, 16, 0.5)

        # NORMALIZE MINUTAE
        core_angle = math.get_point_mean_angle(core_point, self._orientation_field, 5)
        normalized_minutiae = minuti.normalize_minutiae(minutiae, core_point, core_angle)

        return FingerprintTemplate(normalized_minutiae, core_point, delta_point, core_angle)
    
    @timer
    def _generate_polymonial(self) -> None:
        contour40 = ctour.get_reliable_region_border(self._contour, 40)

        # POINCARE INDEX
        core_point, delta_point = poincare.poincare_index(self._orientation_field, self._reliability_map, contour40, 0.4, 0.6, 0.2 * np.pi)

        # POLYMONIAL MODEL OF ORIENTATION FIELD
        PR, PI = orient.polymonial_orientation_field(self._orientation_field, self._reliability_map, 4)

        core_mask = np.zeros_like(PR, dtype=bool)
        core_mask[core_point] = True

        delta_mask = np.zeros_like(PR, dtype=bool)
        delta_mask[delta_point] = True

        # POINTS CHARGES
        cores_charges = point_charge.get_points_charges(self._orientation_field, PR, PI, self._reliability_map, core_mask, 80)
        deltas_charges = point_charge.get_points_charges(self._orientation_field, PR, PI, self._reliability_map, delta_mask, 40)

        # POINT CHARGE
        self._polymonial_orientation_field = orient.point_charge_orientation_field(PR, PI, self._orientation_field, core_mask, delta_mask, cores_charges, deltas_charges, 80, 40)

        imgp.show_img(self._polymonial_orientation_field, title="Polymonial")

        small_orientation_field, small_weight = orient.average_orientation_field(self._polymonial_orientation_field, self.reliability_map, block_size=8)
        overlay = orient.draw_orientation_field(None, small_orientation_field, small_weight, step=8, line_length=6)
        imgp.show_img(overlay, title="Polymonial Orientation Field")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @property
    @timer
    def original_img(self) -> cv2.typing.MatLike:
        return self._original_img

    @property
    @timer
    def normalized(self) -> cv2.typing.MatLike:
        return self._normalized
    
    @property
    @timer
    def binarized(self) -> cv2.typing.MatLike:
        return self._binarized
    
    @property
    @timer
    def skeleton(self) -> cv2.typing.MatLike:
        return self._skeleton
    
    @property
    @timer
    def orientation_field(self) -> cv2.typing.MatLike:
        return self._orientation_field
    
    @property
    @timer
    def reliability_map(self) -> cv2.typing.MatLike:
        return self._reliability_map
    
    @property
    @timer
    def contour(self) -> cv2.typing.MatLike:
        return self._contour

    @timer
    def show_original(self, stop: bool=True) -> None:
        imgp.show_img(self._original_img, "Original")

        if stop:
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    @timer
    def show_normalized(self, stop: bool=True) -> None:
        imgp.show_img(self.normalized, "Normalized")

        if stop:
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    @timer
    def show_binarized(self, stop: bool=True) -> None:
        imgp.show_img(self.binarized, "Binarized")

        if stop:
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    @timer
    def show_skeleton(self, stop: bool=True) -> None:
        imgp.show_img(self.skeleton, "Skeleton")

        if stop:
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    @timer
    def show_orientation_field(self, stop: bool=True) -> None:
        small_orientation_field, small_weight = orient.average_orientation_field(self.orientation_field, self.reliability_map, block_size=8)
        overlay = orient.draw_orientation_field(None, small_orientation_field, small_weight, step=8, line_length=6)
        imgp.show_img(overlay, title="Orientation Field")

        if stop:
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    @timer
    def show_reliability(self, stop: bool=True) -> None:
        imgp.show_img(self.reliability_map, "Reliability")
        
        if stop:
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    @timer
    def show_contour(self, stop: bool=True) -> None:
        skeleton_color = cv2.cvtColor(self.skeleton, cv2.COLOR_GRAY2RGB)
        cv2.drawContours(skeleton_color, [self.contour], -1, (0, 255, 0), thickness=1)
        imgp.show_img(skeleton_color, "Contour")
        
        if stop:
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    @timer
    def show_all_steps(self, stop: bool=True) -> None:
        self.show_original(False)
        self.show_normalized(False)
        self.show_binarized(False)
        self.show_orientation_field(False)
        self.show_reliability(False)
        self.show_skeleton(False)
        self.show_contour(False)

        if stop:
            cv2.waitKey(0)
            cv2.destroyAllWindows()

@timer
def create_templates(file_paths: list[str]) -> list[FingerprintTemplate]:
    templates = []
    
    for path in file_paths:            
        templates.append(FingerprintTemplate.from_img_file(path))
        
    return templates

@timer
def save_templates(save_path: str, templates: list[FingerprintTemplate]) -> None:
    templates_data = [template.to_data() for template in templates]
    with open(save_path, 'w') as f:
        json.dump(templates_data, f, indent=4)

@timer
def load_templates(file_path: str) -> list[FingerprintTemplate]:
    with open(file_path, 'r') as f:
        data = json.load(f)
    return [FingerprintTemplate.from_data(template) for template in data]

@timer
def create_and_save_templates(save_path: str, fingerprints_paths: list[str]) -> list[FingerprintTemplate]:
    templates = create_templates(fingerprints_paths)
    save_templates(save_path, templates)
    return templates

@timer
def authenticate(fingerprint_minutiae: list[minuti.Minutiae], fingerprint_templates: list[FingerprintTemplate], auth_threashold: float | int) -> bool:
    best_match = 0
    for template in fingerprint_templates:
        transformed_minutiae = minuti.transform_minutiae_set_Stolarek(fingerprint_minutiae, template.minutiae, 
                                                                      0.24 * np.pi, 15, 5, 15, 5, 0.5 * np.pi, 0.25 * np.pi)
        matched = minuti.compare_minutiae_sets(transformed_minutiae, template.minutiae)
        best_match = max(best_match, matched)

    if isinstance(auth_threashold, float):
        match_percent = 0.0
        if len(fingerprint_minutiae) != 0:
            match_percent = best_match / len(fingerprint_minutiae)

        return match_percent >= auth_threashold
    elif isinstance(auth_threashold, int):
        return best_match >= auth_threashold