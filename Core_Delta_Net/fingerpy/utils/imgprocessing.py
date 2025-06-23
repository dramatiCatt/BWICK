from timer import timer
import cv2
import numpy as np
from .math import generate_gabor_kernels
import numba

@timer
def load_img(img_path: str) -> cv2.typing.MatLike:
    return cv2.imread(img_path)

@timer
def get_img_size(img: cv2.typing.MatLike) -> tuple[int, int]:
    '''
    rows, columns
    '''
    return img.shape[0], img.shape[1]

@timer
def show_img(img: cv2.typing.MatLike, title: str = "Image", scaleX: float = 1, scaleY: float = 1) -> None:
    cv2.imshow(title, cv2.resize(img, None, fx=scaleX, fy=scaleY))

@timer
def grayscale_img(img: cv2.typing.MatLike) -> cv2.typing.MatLike:
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[..., np.newaxis]

@timer
def normalize_img(img: cv2.typing.MatLike) -> cv2.typing.MatLike:
    intensity = grayscale_img(img)
    maxI = np.max(intensity)
    minI = np.min(intensity)
    return ((intensity - minI) / (maxI - minI))

@timer
@numba.njit
def binarize_img(img: cv2.typing.MatLike, box_size: int | None = 5) -> cv2.typing.MatLike:
    rows = img.shape[0]
    columns = img.shape[1]

    if box_size is not None:
        binarized = np.zeros_like(img)

        for y in range(0, rows, box_size):
            y_end = min(y + box_size, rows)

            for x in range(0, columns):
                x_end = min(x + box_size, columns)

                # Get box values
                box = img[y:y_end, x:x_end]

                local_threshold = np.mean(box)
                box_mask = box >= local_threshold
                binarized_box = np.where(box_mask, 1.0, 0.0)

                binarized[y:y_end, x:x_end] = binarized_box
    else:
        binarized = np.where(img >= np.mean(img), 1.0, 0.0).astype(img.dtype)

    return binarized

@timer
def skeletonize_img(img: cv2.typing.MatLike) -> cv2.typing.MatLike:
    img_u8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.ximgproc.thinning(img_u8, thinningType=cv2.ximgproc.THINNING_GUOHALL)

@timer
def crop_img(img: cv2.typing.MatLike, left: int, right: int, top: int, bottom: int) -> cv2.typing.MatLike:
    w, h = get_img_size(img)
    
    start_x = max(0, left)
    end_x = min(right, w - right)
    start_y = max(0, top)
    end_y = min(bottom, h - bottom)

    return img[start_y:end_y, start_x:end_x]

@timer
def gaussian_blur(img: cv2.typing.MatLike, kernel_size: int, power: float) -> cv2.typing.MatLike:
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigmaX=power)

@timer
def median_filter(img: cv2.typing.MatLike, kernel_size: int = 3) -> cv2.typing.MatLike:
    return cv2.medianBlur(img, kernel_size)

@timer
def directional_filtering(img: cv2.typing.MatLike, orientation_field: cv2.typing.MatLike, reliability_map: cv2.typing.MatLike, block_size: int=16):
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

            block_reliability = reliability_map[y:y_end, x:x_end]
            mean_reliability = np.mean(block_reliability)

            if mean_reliability < 0.1:
                continue

            angle_diffs = np.abs(np.angle(np.exp(1j * (angles - mean_theta))))
            kernel_idx = np.argmin(angle_diffs)
            kernel = kernels[kernel_idx]

            block = img[y:y_end, x:x_end]

            k_half = kernel.shape[0] // 2
            padded_block = cv2.copyMakeBorder(block, k_half, k_half, k_half, k_half, borderType=cv2.BORDER_REFLECT)
            filtered = cv2.filter2D(padded_block, -1, kernel)
            filtered_block = filtered[k_half:k_half + y_end - y, k_half:k_half + x_end - x]

            dfi[y:y_end, x:x_end] = filtered_block[..., np.newaxis] * mean_reliability + dfi[y:y_end, x:x_end] * (1 - mean_reliability)

    return dfi.squeeze()