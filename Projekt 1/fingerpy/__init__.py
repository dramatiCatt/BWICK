import cv2
import numpy as np
from typing import Self
from __future__ import annotations
import json
from timer import timer
from .utils import imgprocessing as img, orientation as orient, contour as ctour, poincare, minutiae as minuti, math

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

    @timer
    @classmethod
    def from_img(cls, img: cv2.typing.MatLike) -> Self:
        return cls.from_fingerprint(Fingerprint(img))

    @timer
    @classmethod
    def from_img_file(cls, img_path: str) -> Self:
        return cls.from_img(img.load_img(img_path))

    @timer
    @classmethod
    def from_fingerprint(cls, fingerprint: 'Fingerprint') -> Self:
        return fingerprint.generate_template()

    @timer
    @classmethod
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

    @timer
    @classmethod
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

    @timer
    @property
    def minutiae(self) -> list[minuti.Minutiae]:
        return self._minutiae
    
    @timer
    @property
    def core(self) -> np.ndarray | None:
        return self._core_point
    
    @timer
    @property
    def delta(self) -> np.ndarray | None:
        return self._delta_point
    
    @timer
    @property
    def angle(self) -> float:
        return self._core_angle

    @timer
    def show_points(self) -> None:
        # endings_mask = [m[MINUTIAE_POS] for m in minutiae if m[MINUTIAE_TYPE] == MINUTIAE_ENDING]
        # bifurcation_mask = [m[MINUTIAE_POS] for m in minutiae if m[MINUTIAE_TYPE] == MINUTIAE_BIFURCATION] 

        # plt.figure(figsize=(10, 5))

        # plt.subplot(1, 2, 1)
        # plt.imshow(skeleton, cmap='gray')
        # if core_point is not None:
        #     plt.scatter(core_point[1], core_point[0], color='blue', label='Core', s=15)
        # if delta_point is not None:
        #     plt.scatter(delta_point[1], delta_point[0], color='red', label='Delta', s=15)
        # plt.legend()
        # plt.title("Singular Points (Poincare Index)")
        # # plt.tight_layout()
        # # plt.axis("equal")

        # plt.subplot(1, 2, 2)
        # plt.imshow(skeleton, cmap='gray')
        # plt.scatter(np.array(endings_mask)[..., 1], np.array(endings_mask)[..., 0], color='green', label='Ending', s=15)
        # plt.scatter(np.array(bifurcation_mask)[..., 1], np.array(bifurcation_mask)[..., 0], color='violet', label='Bifurcation', s=15)
        # plt.legend()
        # plt.title("Minutiae")
        # # plt.tight_layout()
        # # plt.axis("equal")

        # plt.show()
        pass

class Fingerprint():
    @timer
    def __init__(self, img: cv2.typing.MatLike) -> None:
        self._original_img = img
        self._preprocess()

    @timer
    @classmethod
    def from_file(cls, img_path: str) -> Self:
        return cls(img.load_img(img_path))
    
    @timer
    def _preprocess(self) -> None:
        # NORMALIZE FINGERPRINT
        self._normalized = img.normalize_img(self._original_img)

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
        self._binarized = img.binarize_img(self._normalized, 16)

        # SKELETONIZE FINGERPRINT
        self._skeleton = img.skeletonize_img(self._binarized)
        self._skeleton[self._reliability_map <= 0] = 0

    @timer
    def generate_template(self) -> FingerprintTemplate:
        contour40 = ctour.get_reliable_region_border(self._contour, 40)
        contour16 = ctour.get_reliable_region_border(self._contour, 16)

        # POINCARE INDEX
        core_point, delta_point = poincare.poincare_index(self._orientation_field, self._reliability_map, contour40, 0.5, 0.4 * np.pi)

        # MINUTAE
        minutiae = minuti.extract_minutiae(self._skeleton, self._reliability_map, self._orientation_field, contour16, 16, 0.5)

        # NORMALIZE MINUTAE
        core_angle = math.get_point_mean_angle(core_point, self._orientation_field, 5)
        normalized_minutiae = minuti.normalize_minutiae(minutiae, core_point, core_angle)

        return FingerprintTemplate(normalized_minutiae, core_point, delta_point, core_angle)
    
    @timer
    def _generate_polymonial(self) -> None:
        # POLYMONIAL MODEL OF ORIENTATION FIELD
        # PR, PI = polymonial_orientation_field(orientation_field, reliability_map, 4)

        # POINTS CHARGES
        # cores_charges = get_points_charges(orientation_field, PR, PI, reliability_map, cores_mask, 80)
        # deltas_charges = get_points_charges(orientation_field, PR, PI, reliability_map, deltas_mask, 40)

        # POINT CHARGE
        # final_O = point_charge_orientation_field(PR, PI, orientation_field, cores_mask, deltas_mask, cores_charges, deltas_charges, 80, 40)
        pass

    @timer
    @property
    def original_img(self) -> cv2.typing.MatLike:
        return self._original_img

    @timer
    @property
    def normalized(self) -> cv2.typing.MatLike:
        return self._normalized
    
    @timer
    @property
    def binarized(self) -> cv2.typing.MatLike:
        return self._binarized
    
    @timer
    @property
    def skeleton(self) -> cv2.typing.MatLike:
        return self._skeleton
    
    @timer
    @property
    def orientation_field(self) -> cv2.typing.MatLike:
        return self._orientation_field
    
    @timer
    @property
    def reliability_map(self) -> cv2.typing.MatLike:
        return self._reliability_map
    
    @timer
    @property
    def contour(self) -> cv2.typing.MatLike:
        return self._contour

    @timer
    def show_original(self) -> None:
        img.show_img(self._original_img, "Original")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @timer
    def show_normalized(self) -> None:
        img.show_img(self.normalized, "Normalized")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @timer
    def show_binarized(self) -> None:
        img.show_img(self.binarized, "Binarized")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @timer
    def show_skeleton(self) -> None:
        img.show_img(self.skeleton, "Skeleton")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @timer
    def show_orientation_field(self) -> None:
        small_orientation_field, small_weight = orient.average_orientation_field(self.orientation_field, self.reliability_map, block_size=8)
        overlay = orient.draw_orientation_field(None, small_orientation_field, small_weight, step=8, line_length=6)
        img.show_img(overlay, title="Orientation Field")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @timer
    def show_reliability(self) -> None:
        img.show_img(self.reliability_map, "Reliability")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @timer
    def show_contour(self) -> None:
        skeleton_color = cv2.cvtColor(self.skeleton, cv2.COLOR_GRAY2RGB)
        cv2.drawContours(skeleton_color, [self.contour], -1, (0, 255, 0), thickness=1)
        img.show_img(skeleton_color, "Contour")
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