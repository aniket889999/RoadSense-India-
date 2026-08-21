import subprocess
import sys

def test_cli_help():
    scripts = [
        "scripts/build_group_manifest.py",
        "scripts/prepare_rdd2022_potholes.py",
        "scripts/validate_yolo_dataset.py",
        "scripts/train_pothole.py",
        "scripts/evaluate_pothole.py"
    ]

    for script in scripts:
        result = subprocess.run([sys.executable, script, "--help"], capture_output=True, text=True)
        assert result.returncode == 0, f"{script} failed to run --help natively"
