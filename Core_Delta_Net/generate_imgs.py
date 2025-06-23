if __name__ == "__main__":
    import pathlib
    import fingerpy as fp
    import cv2
    import numpy as np
    import json

    data_dir = pathlib.Path("./Data")

    img_type = "normalized"

    for item in data_dir.iterdir():
        if not item.name.endswith(".tif"):
            continue

        finger = fp.Fingerprint.from_file(str(item))

        pngPath = f"./Data/{item.name[:item.name.rfind('.')]}"

        normalizedPngPath = f"{pngPath}_normalized.png"
        orientationFieldPngPath = f"{pngPath}_orientation_field.png"
        binarizedPngPath = f"{pngPath}_binarized.png"
        skeletonPngPath = f"{pngPath}_skeleton.png"

        cv2.imwrite(normalizedPngPath, (finger.normalized * 255).astype(np.uint8))

        cv2.imwrite(orientationFieldPngPath, (finger.orientation_field * 255).astype(np.uint8))

        cv2.imwrite(binarizedPngPath, (finger.binarized * 255).astype(np.uint8))

        cv2.imwrite(skeletonPngPath, finger.skeleton)

        jsonPath = f"{pngPath}.json"
        cropJsonPath = f"{pngPath}_crop.json"
        with open(jsonPath, 'r') as f:
            data = json.load(f)

        core = data['core']
        core[0] -= finger._crop_start_x
        core[1] -= finger._crop_start_y
        data['core'] = core

        delta = data.get('delta')
        if delta is not None:
            delta[0] -= finger._crop_start_x
            delta[1] -= finger._crop_start_y
            data['delta'] = delta
        
        with open(cropJsonPath, 'w') as f:
            json.dump(data, f, indent=4)