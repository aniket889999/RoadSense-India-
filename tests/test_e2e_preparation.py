import subprocess
import sys
import os
import tempfile
import csv
from PIL import Image

def create_synthetic_image(path, idx=0):
    img = Image.new('RGB', (100, 100), color=(idx, idx, idx))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path, format='JPEG')

def write_xml(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("""<annotation>
            <size><width>100</width><height>100</height></size>
            <object><name>D40</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>20</xmax><ymax>20</ymax></bndbox></object>
        </annotation>""")

def test_e2e_preparation_nested_paths_and_audit():
    with tempfile.TemporaryDirectory() as td:
        rdd_root = os.path.join(td, "mock_rdd")
        img_dir = os.path.join(rdd_root, "India", "train", "images")
        xml_dir = os.path.join(rdd_root, "India", "train", "annotations", "xmls")

        # Test collision files
        create_synthetic_image(os.path.join(img_dir, "a_b", "p.jpg"), 1)
        write_xml(os.path.join(xml_dir, "a_b", "p.xml"))

        create_synthetic_image(os.path.join(img_dir, "a", "b_p.jpg"), 2)
        write_xml(os.path.join(xml_dir, "a", "b_p.xml"))

        # We need at least 3 positive groups, so add another one
        create_synthetic_image(os.path.join(img_dir, "c.jpg"), 3)
        write_xml(os.path.join(xml_dir, "c.xml"))

        groups_csv = os.path.join(td, "groups.csv")
        with open(groups_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_relpath", "group_id"])
            writer.writerow(["a_b/p.jpg", "g1"])
            writer.writerow(["a/b_p.jpg", "g2"])
            writer.writerow(["c.jpg", "g3"])

        out_dir = os.path.join(td, "processed")
        prep_script = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "prepare_rdd2022_potholes.py")
        result = subprocess.run([sys.executable, prep_script,
                                 "--rdd-root", rdd_root,
                                 "--groups-csv", groups_csv,
                                 "--output-dir", out_dir,
                                 "--audit-near-duplicates"],
                                 capture_output=True, text=True)

        assert result.returncode == 0, f"Prep failed: {result.stderr}\n{result.stdout}"

        # Verify collision files both exist
        images_found = []
        labels_found = []
        for root, _, files in os.walk(os.path.join(out_dir, "images")):
            for f in files:
                if f.endswith(".jpg"): images_found.append(f)
        for root, _, files in os.walk(os.path.join(out_dir, "labels")):
            for f in files:
                if f.endswith(".txt"): labels_found.append(f)

        assert len(images_found) == 3
        assert len(labels_found) == 3

        assert "p.jpg" in images_found
        assert "b_p.jpg" in images_found

        # Check audit manifest exists
        audit_path = os.path.join(out_dir, "manifests", "near_duplicates_audit.csv")
        assert os.path.exists(audit_path)
