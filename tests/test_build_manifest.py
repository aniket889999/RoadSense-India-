import subprocess
import sys
import os
import tempfile

def test_build_group_manifest_empty():
    with tempfile.TemporaryDirectory() as td:
        rdd_root = os.path.join(td, "mock_rdd")
        img_dir = os.path.join(rdd_root, "India", "train", "images")
        os.makedirs(img_dir)

        # Add non-jpg files
        with open(os.path.join(img_dir, "fake.txt"), "w") as f:
            f.write("text")

        out_csv = os.path.join(td, "out.csv")
        script = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "build_group_manifest.py")

        res = subprocess.run([sys.executable, script, "--rdd-root", rdd_root, "--output", out_csv], capture_output=True, text=True)
        assert res.returncode == 1
        assert "contains no supported .jpg images" in res.stdout
        assert not os.path.exists(out_csv)
