from timer import timer
import cv2
import numpy as np

@timer
def get_largest_reliable_region(reliability_map: cv2.typing.MatLike, threshold: float = 0.3) -> np.ndarray:
    """
    Returns:
        array of contour points
    """

    mask = (reliability_map > threshold).astype(np.uint8)

    # Denoising: otwarcie morfologiczne
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # kontur
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("Brak obszarów powyżej progu wiarygodności")
    
    # Największy kontur
    return max(contours, key=cv2.contourArea).squeeze()

@timer
def get_reliable_region_border(contour: np.ndarray, border_size: int = 16) -> np.ndarray:
    center = np.array([contour[..., 0].mean(), contour[..., 1].mean()])

    border_contour = np.zeros_like(contour)
    for i in range(len(contour)):
        direction = center - contour[i]
        direction /= np.sqrt(direction[0] ** 2 + direction[1] ** 2)

        border_contour[i] = contour[i] + direction * border_size
    return border_contour

@timer
def get_reliable_region_rectangle(contour: np.ndarray) -> tuple[int, int, int, int]:
    """
    Returns:
        start_y, end_y, start_x, end_x
    """

    x, y, w, h = cv2.boundingRect(contour)
    return y, y + h, x, x + w
