if __name__ == "__main__":
    import pathlib
    import matplotlib.pyplot as plt
    import os
    import json
    import cv2

    data_dir = pathlib.Path("./Data")

    for item in data_dir.iterdir():
        if not item.name.endswith(".tif"):
            continue

        img = cv2.imread(str(item))

        jsonPath: str = f"./Data/{item.name[:item.name.rfind('.')]}.json"
        jsonData : dict[str, list[int]] = {}
        if os.path.exists(jsonPath):
            with open(jsonPath, 'r') as f:
                jsonData = json.load(f)
            
        plt.figure(figsize=(5, 5))
        plt.imshow(img, cmap='gray')

        if jsonData.get('core') is not None:
            core_point: list[int] = jsonData['core']
            plt.scatter(core_point[0], core_point[1], color='blue', label='Core', s=15)
        
        if jsonData.get('delta') is not None:
            delta_point: list[int] = jsonData['delta']
            plt.scatter(delta_point[0], delta_point[1], color='red', label='Delta', s=15)
        
        plt.legend()
        plt.title(f"Singular Points ({item})")

        plt.show()