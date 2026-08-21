import os
import shutil
from typing import Dict, List, Any
import yaml

def init_yolo_directories(output_dir: str, overwrite: bool = False) -> bool:
    if os.path.exists(output_dir):
        if not overwrite:
            # check if empty
            if os.listdir(output_dir):
                return False
        else:
            shutil.rmtree(output_dir)

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    os.makedirs(os.path.join(output_dir, "manifests"), exist_ok=True)
    return True

def write_pothole_yaml(output_dir: str):
    data = {
        "path": os.path.abspath(output_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {
            0: "pothole"
        }
    }
    with open(os.path.join(output_dir, "pothole.yaml"), "w") as f:
        yaml.dump(data, f, sort_keys=False)

def write_manifest_csv(output_dir: str, filename: str, headers: List[str], rows: List[List[Any]]):
    import csv
    with open(os.path.join(output_dir, "manifests", filename), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
