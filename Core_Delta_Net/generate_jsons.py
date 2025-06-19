if __name__ == "__main__":
    import pathlib
    import os
    import json

    data_dir = pathlib.Path("./Data")

    for item in data_dir.iterdir():
        if not item.name.endswith(".tif"):
            continue

        jsonPath = f"./Data/{item.name[:item.name.rfind('.')]}.json"

        if not os.path.exists(jsonPath):
            with open(jsonPath, 'w') as f:
                json.dump({"core": [], "delta": []}, f, indent=4)