import subprocess
import sys
import os
import tempfile
import json
from src.ml.metrics import extract_detection_metrics

def test_evaluate_pothole_metrics_extraction():
    # Pure metric extraction
    d = {
        'metrics/precision(B)': 0.8,
        'metrics/mAP50(B)': 0.9,
    }
    res = extract_detection_metrics(d)
    assert res["precision"] == 0.8
    assert res["mAP50"] == 0.9
    assert "recall" not in res

def test_metadata_fingerprint_changes():
    with tempfile.TemporaryDirectory() as td:
        rdd_root = os.path.join(td, "mock_rdd")
        img_dir = os.path.join(rdd_root, "India", "train", "images")
        xml_dir = os.path.join(rdd_root, "India", "train", "annotations", "xmls")
        os.makedirs(img_dir)
        os.makedirs(xml_dir)

        # 3 groups
        import csv
        from PIL import Image
        def create_synthetic_image(path, idx=0):
            img = Image.new('RGB', (10, 10), color=(idx, idx, idx))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            img.save(path, format='JPEG')

        for i in range(3):
            create_synthetic_image(os.path.join(img_dir, f"{i}.jpg"), i)
            with open(os.path.join(xml_dir, f"{i}.xml"), "w") as f:
                f.write(f"<annotation><size><width>10</width><height>10</height></size><object><name>D40</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>2</xmax><ymax>2</ymax></bndbox></object></annotation>")

        groups_csv = os.path.join(td, "groups.csv")
        with open(groups_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_relpath", "group_id"])
            for i in range(3): writer.writerow([f"{i}.jpg", f"g{i}"])

        # Run 1
        out_dir1 = os.path.join(td, "processed1")
        prep_script = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "prepare_rdd2022_potholes.py")
        subprocess.run([sys.executable, prep_script, "--rdd-root", rdd_root, "--groups-csv", groups_csv, "--output-dir", out_dir1])

        with open(os.path.join(out_dir1, "manifests", "preparation_metadata.json")) as f:
            fp1 = json.load(f)["dataset_fingerprint"]

        # Write sidecar metadata
        meta_path = os.path.join(td, "groups_metadata.json")
        with open(meta_path, "w") as f:
            json.dump({"grouping_quality": "fake"}, f)

        # Run 2
        out_dir2 = os.path.join(td, "processed2")
        subprocess.run([sys.executable, prep_script, "--rdd-root", rdd_root, "--groups-csv", groups_csv, "--output-dir", out_dir2])

        with open(os.path.join(out_dir2, "manifests", "preparation_metadata.json")) as f:
            fp2 = json.load(f)["dataset_fingerprint"]

        assert fp1 != fp2, "Fingerprint did not change when sidecar metadata was added"
