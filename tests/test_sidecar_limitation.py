import os
import subprocess
import sys
import tempfile
import csv
import json

def test_preparation_sidecar_limitation():
    with tempfile.TemporaryDirectory() as td:
        rdd_root = os.path.join(td, "mock_rdd")
        img_dir = os.path.join(rdd_root, "India", "train", "images")
        xml_dir = os.path.join(rdd_root, "India", "train", "annotations", "xmls")
        os.makedirs(img_dir)
        os.makedirs(xml_dir)

        from PIL import Image
        def create_synthetic_image(path, idx=0):
            img = Image.new('RGB', (10, 10), color=(idx, idx, idx))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            img.save(path, format='JPEG')

        for i in range(3):
            create_synthetic_image(os.path.join(img_dir, f"{i}.jpg"), i)
            with open(os.path.join(xml_dir, f"{i}.xml"), "w") as f:
                f.write("<annotation><size><width>10</width><height>10</height></size><object><name>D40</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>2</xmax><ymax>2</ymax></bndbox></object></annotation>")

        groups_csv = os.path.join(td, "groups.csv")
        with open(groups_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_relpath", "group_id"])
            for i in range(3):
                writer.writerow([f"{i}.jpg", f"g{i}"])

        meta_json = os.path.join(td, "groups_metadata.json")
        with open(meta_json, "w") as f:
            json.dump({
                "grouping_quality": "sequence_proxy_not_verified_route",
                "limitation": "my_limitation_string"
            }, f)

        out_dir = os.path.join(td, "processed")
        prep_script = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "prepare_rdd2022_potholes.py")
        res = subprocess.run([sys.executable, prep_script, "--rdd-root", rdd_root, "--groups-csv", groups_csv, "--output-dir", out_dir], capture_output=True, text=True)
        assert res.returncode == 0

        with open(os.path.join(out_dir, "manifests", "preparation_metadata.json")) as f:
            meta = json.load(f)

        assert meta["grouping_method"] == "sequence_proxy_not_verified_route"
        assert meta["residual_leakage_limitation"] == "my_limitation_string"
