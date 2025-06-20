from torch.utils.data import Dataset
from PIL import Image
import json
import torch
import numpy as np # Albumentations często pracuje z NumPy arrays
import albumentations as A
from albumentations.pytorch import ToTensorV2 # Specjalna transformacja dla PyTorch
import cv2

class FingerprintDataset(Dataset[tuple[np.ndarray[tuple[int, ...]], torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, split_file_path: str, split_type: str, transform: A.Compose | None = None, target_image_size: tuple[int, int] = (224, 224)):
        """
        Args:
            split_file_path (str): Ścieżka do pliku JSON z podziałem danych (np. "dataset_split.json").
            split_type (str): Typ podziału ('train', 'val', 'test').
            transform (albumentations.Compose, optional): Opcjonalne transformacje z Albumentations.
            target_image_size (Tuple[int, int]): Rozmiar do którego obrazy zostaną przeskalowane (szerokość, wysokość).
        """
        with open(split_file_path, 'r', encoding='utf-8') as f:
            self.data: list[dict[str, str | int]] = json.load(f)[split_type]
        self.transform: A.Compose | None = transform
        self.target_image_size: tuple[int, int] = target_image_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[np.ndarray[tuple[int, ...]], torch.Tensor, torch.Tensor, torch.Tensor]:
        sample: dict[str, str | int] = self.data[idx]
        image_path: str | int = sample["image_path"]
        assert isinstance(image_path, str)

        image: Image.Image = Image.open(image_path).convert("RGB")
        image_np: np.ndarray[tuple[int, ...]] = np.array(image)

        original_width, original_height = image.size

        json_path: str | int = sample["json_path"]
        assert isinstance(json_path, str)
        image_point_data: dict[str, list[int]] = {}
        with open(json_path, 'r') as f:
            image_point_data = json.load(f)

        has_delta = image_point_data.get('delta') is not None
        
        breakpoint()
        transform_params: dict[str, np.ndarray[tuple[int, ...]] | list[list[int]]]= {
            'image': image_np,
            'keypoints': [image_point_data['core']]
        }

        if has_delta:
            transform_params['keypoints'].append(image_point_data['delta'])

        # Zastosuj transformacje
        if self.transform:
            augmented: dict[str, list[int]] = self.transform(**transform_params)
            image_transformed = augmented['image']
            core_transformed = augmented['keypoints'][0]
            delta_transformed = None if len(augmented['keypoints']) == 1 else augmented['keypoints'][1]
        else:
            image_transformed = A.Compose([
                A.Resize(self.target_image_size[1], self.target_image_size[0]), # height, width
                ToTensorV2()
            ])(image=image_np)['image']
            
            scale_x: float = self.target_image_size[0] / original_width
            scale_y: float = self.target_image_size[1] / original_height
            
            core = image_point_data['core']
            core_transformed = [core[0] * scale_x, core[1] * scale_y]

            delta = image_point_data.get('delta')
            delta_transformed = None if delta is None else [delta[0] * scale_x, delta[1] * scale_y]

        has_delta = delta_transformed is not None

        height_transformed, width_transformed = image_transformed.shape[1], image_transformed.shape[2] # C, H, W

        core_coords_transformed = torch.tensor(core_transformed, dtype=torch.float32)
        # Normalizacja
        core_coords_normalized = torch.tensor([
            core_coords_transformed[0] / width_transformed,
            core_coords_transformed[1] / height_transformed
        ], dtype=torch.float32)

        delta_coords_normalized = torch.tensor([-1.0, -1.0], dtype=torch.float32) # Domyślnie brak delty
        
        if has_delta:
            delta_coords_transformed = torch.tensor(delta_transformed, dtype=torch.float32)
            delta_coords_normalized = torch.tensor([
                delta_coords_transformed[0] / width_transformed,
                delta_coords_transformed[1] / height_transformed
            ], dtype=torch.float32)

        delta_existence_label: torch.Tensor = torch.tensor(1.0 if has_delta else 0.0, dtype=torch.float32)

        return image_transformed, core_coords_normalized, delta_coords_normalized, delta_existence_label
    
def get_fingerprint_transforms(
    image_size: tuple[int, int] = (224, 224),
    apply_rotate: bool = True,
    rotate_limit: int = 15, # stopnie
    apply_shift_scale_rotate: bool = False, # Lepsza opcja dla obrotu, przesunięcia i skalowania
    shift_limit: float = 0.05, # % obrazu
    scale_limit: float = 0.1,  # % zmiany skali
    apply_horizontal_flip: bool = True,
    apply_vertical_flip: bool = False, # Ostrożnie z odwracaniem pionowym dla odcisków
    apply_brightness_contrast: bool = True,
    brightness_limit: float = 0.2,
    contrast_limit: float = 0.2,
    apply_gaussian_noise: bool = True,
    noise_limit_std: float = 0.05,
    normalize_image: bool = True # Normalizacja wartości pikseli (0-1, potem mean/std)
) -> A.Compose:
    """
    Tworzy złożone transformacje Albumentations z opcjami włączania/wyłączania.

    Args:
        image_size (Tuple[int, int]): Docelowy rozmiar obrazu (szerokość, wysokość).
        apply_rotate (bool): Czy zastosować losowy obrót.
        rotate_limit (int): Limit stopni dla obrotu (-rotate_limit do +rotate_limit).
        apply_shift_scale_rotate (bool): Czy zastosować A.ShiftScaleRotate (obejmuje obrót, przesunięcie, skalowanie).
                                        Jeśli True, apply_rotate, shift_limit, scale_limit są używane w tej transformacji.
                                        Preferowane zamiast oddzielnych obrotów, przesunięć, skalowań.
        shift_limit (float): Limit przesunięcia jako ułamek wymiarów obrazu.
        scale_limit (float): Limit skalowania jako ułamek.
        apply_horizontal_flip (bool): Czy zastosować losowe odbicie poziome.
        apply_vertical_flip (bool): Czy zastosować losowe odbicie pionowe.
        apply_brightness_contrast (bool): Czy zastosować zmianę jasności/kontrastu.
        brightness_limit (float): Limit zmiany jasności.
        contrast_limit (float): Limit zmiany kontrastu.
        apply_gaussian_noise (bool): Czy zastosować szum Gaussa.
        noise_limit_std (float): Limit odchylenia standardowego dla szumu Gaussa.
        normalize_image (bool): Czy znormalizować obraz do standardowych wartości dla ImageNet.
        
    Returns:
        albumentations.Compose: Złożona transformacja Albumentations.
    """
    transforms_list = []

    # Zawsze skaluj obraz do docelowego rozmiaru na początku.
    # Upewnij się, że Resize jest przed wszelkimi geometrycznymi transformacjami.
    transforms_list.append(A.Resize(image_size[1], image_size[0])) # height, width

    if apply_shift_scale_rotate:
        transforms_list.append(
            A.Affine(
                scale=(1 - scale_limit, 1 + scale_limit), # Zakres skalowania
                translate_percent={'x': (-shift_limit, shift_limit), 'y': (-shift_limit, shift_limit)}, # Zakres przesunięcia jako procent
                rotate=(-rotate_limit if apply_rotate else 0, rotate_limit if apply_rotate else 0), # Zakres obrotu
                shear=(0, 0), # Jeśli nie potrzebujesz ścinania, ustaw na None lub (0, 0)
                p=0.8, # Prawdopodobieństwo zastosowania tej transformacji
                interpolation=cv2.INTER_LINEAR, # Metoda interpolacji dla obrazu
                mask_interpolation=cv2.INTER_NEAREST, # Metoda interpolacji dla maski (jeśli masz maski, w twoim przypadku nie)
                cval=0, # Wartość wypełnienia dla pikseli poza obrazem (czarny)
                cval_mask=0, # Wartość wypełnienia dla maski
                mode=cv2.BORDER_REFLECT_101 # Sposób obsługi krawędzi obrazu
            )
        )
    else: # Oddzielne transformacje jeśli nie używasz ShiftScaleRotate
        if apply_rotate:
            transforms_list.append(A.Rotate(limit=rotate_limit, p=0.8)) # p=0.8 dla losowości

        # Albumentations nie ma oddzielnych A.Shift i A.Scale dla keypointów tak łatwo jak ShiftScaleRotate.
        # A.ShiftScaleRotate jest zalecanym sposobem na te trzy transformacje jednocześnie.
        # Jeśli chcesz oddzielnie, musiałbyś stworzyć własne kombinacje, co jest bardziej złożone
        # lub użyć tylko Rotate i Flipping.
        # Jeśli apply_shift_scale_rotate jest False, to przesunięcie i skalowanie
        # nie będą używane, chyba że zastosujesz je jako część A.Affine (bardziej zaawansowane).
        print("Ostrzeżenie: Bez apply_shift_scale_rotate=True, przesunięcie i skalowanie nie będą stosowane oddzielnie na punktach.")


    if apply_horizontal_flip:
        transforms_list.append(A.HorizontalFlip(p=0.5))

    if apply_vertical_flip:
        transforms_list.append(A.VerticalFlip(p=0.5)) # Prawdopodobieństwo 0.5

    if apply_brightness_contrast:
        transforms_list.append(A.RandomBrightnessContrast(
            brightness_limit=brightness_limit,
            contrast_limit=contrast_limit,
            p=0.5
        ))

    if apply_gaussian_noise:
        transforms_list.append(A.GaussNoise(std_range=(0.0, noise_limit_std), p=0.5))

    # Normalizacja obrazu po wszystkich transformacjach
    if normalize_image:
        transforms_list.append(A.Normalize(
            mean=(0.485, 0.456, 0.406), # Standardowe wartości dla ImageNet
            std=(0.229, 0.224, 0.225)
        ))

    # Ostatnia transformacja: konwersja do tensora PyTorch
    transforms_list.append(ToTensorV2())

    # Zwróć złożoną transformację. keypoint_params jest KLUCZOWE
    # by Albumentations wiedziało, jak transformować punkty.
    return A.Compose(
        transforms_list,
        keypoint_params=A.KeypointParams(format='xy', label_fields=[]) # 'xy' dla pikseli (x,y)
    )