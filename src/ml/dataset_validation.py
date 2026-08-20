import os
import yaml
import math
import re
from typing import List, Dict, Tuple

def validate_dataset_item(xml_w: int, xml_h: int, decoded_width: int, decoded_height: int, boxes: List[Dict[str, float]]) -> Tuple[bool, str]:
    if xml_w != decoded_width or xml_h != decoded_height:
        return False, f"Size mismatch: XML({xml_w}x{xml_h}) vs Image({decoded_width}x{decoded_height})"

    if xml_w <= 0 or xml_h <= 0:
        return False, "Dimensions must be positive"

    for i, b in enumerate(boxes):
        xmin = b["xmin"]
        ymin = b["ymin"]
        xmax = b["xmax"]
        ymax = b["ymax"]

        if not (math.isfinite(xmin) and math.isfinite(ymin) and math.isfinite(xmax) and math.isfinite(ymax)):
            return False, f"Non-finite box at index {i}"

        if xmin >= xmax or ymin >= ymax:
            return False, f"Zero or negative area box at index {i}"

        if xmin < 0 or ymin < 0 or xmax > decoded_width or ymax > decoded_height:
            return False, f"Out of bounds box at index {i} ({xmin}, {ymin}, {xmax}, {ymax})"

    return True, ""

def validate_prepared_yolo_dataset(dataset_dir: str) -> Tuple[bool, str]:
    try:
        abs_dataset_dir = os.path.abspath(dataset_dir)
        pothole_yaml = os.path.join(abs_dataset_dir, "pothole.yaml")
        if not os.path.exists(pothole_yaml):
            return False, "pothole.yaml not found"

        with open(pothole_yaml, "r") as f:
            data = yaml.safe_load(f)

        if not data or not isinstance(data, dict):
            return False, "Malformed pothole.yaml"

        if data.get("names") != {0: "pothole"}:
            return False, "pothole.yaml must define exactly names: {0: 'pothole'}"

        if data.get("path") != abs_dataset_dir:
            return False, f"YAML path {data.get('path')} does not match dataset dir {abs_dataset_dir}"

        if data.get("train") != "images/train" or data.get("val") != "images/val" or data.get("test") != "images/test":
            return False, "YAML splits must map exactly to images/<split>"

        for split in ["train", "val", "test"]:
            img_dir = os.path.join(abs_dataset_dir, "images", split)
            lbl_dir = os.path.join(abs_dataset_dir, "labels", split)

            if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
                return False, f"Missing {split} image or label directory"

            images = set()
            for root, _, files in os.walk(img_dir):
                for f in files:
                    if f.lower().endswith(".jpg"):
                        images.add(os.path.relpath(os.path.join(root, os.path.splitext(f)[0]), img_dir))

            labels = set()
            for root, _, files in os.walk(lbl_dir):
                for f in files:
                    if f.endswith(".txt"):
                        labels.add(os.path.relpath(os.path.join(root, os.path.splitext(f)[0]), lbl_dir))

            if not images:
                return False, f"Split {split} has no images"

            if images != labels:
                return False, f"Split {split} has orphan images or labels"

            potholes_found = False

            for lbl in labels:
                with open(os.path.join(lbl_dir, f"{lbl}.txt"), "r") as f:
                    lines = [l.strip() for l in f.readlines() if l.strip()]

                for line in lines:
                    parts = line.split()
                    if len(parts) != 5:
                        return False, f"Label {lbl}.txt has invalid line (must be 5 parts)"

                    try:
                        cls_id = int(parts[0])
                    except ValueError:
                        return False, f"Label {lbl}.txt has non-integer class id"

                    if cls_id != 0:
                        return False, f"Label {lbl}.txt has invalid class id {cls_id}"

                    try:
                        x_norm = float(parts[1])
                        y_norm = float(parts[2])
                        w_norm = float(parts[3])
                        h_norm = float(parts[4])
                    except ValueError:
                        return False, f"Label {lbl}.txt has non-float coordinates"

                    if not (math.isfinite(x_norm) and math.isfinite(y_norm) and math.isfinite(w_norm) and math.isfinite(h_norm)):
                        return False, f"Label {lbl}.txt has non-finite coordinate"

                    if w_norm <= 0 or h_norm <= 0:
                        return False, f"Label {lbl}.txt has zero/negative width/height"

                    if x_norm - w_norm/2 < 0 or x_norm + w_norm/2 > 1 or y_norm - h_norm/2 < 0 or y_norm + h_norm/2 > 1:
                        return False, f"Label {lbl}.txt has out of bounds coordinates"

                    potholes_found = True

            if not potholes_found:
                return False, f"Split {split} has no pothole instances"

        manifests_dir = os.path.join(abs_dataset_dir, "manifests")
        meta_path = os.path.join(manifests_dir, "preparation_metadata.json")
        if not os.path.exists(meta_path):
            return False, "preparation_metadata.json is missing"

        import json
        with open(meta_path, "r") as f:
            meta = json.load(f)
            fp = meta.get("dataset_fingerprint", "")
            if not isinstance(fp, str) or not re.match(r"^[0-9a-f]{64}$", fp):
                return False, "preparation_metadata.json missing valid 64-char hex dataset_fingerprint"

        return True, ""
    except Exception as e:
        return False, f"Validation crashed: {str(e)}"
