from src.ml.dataset_validation import validate_dataset_item, validate_prepared_yolo_dataset
import os
import tempfile
import yaml

def test_validation_size_mismatch():
    is_valid, err = validate_dataset_item(100, 100, 200, 200, [])
    assert not is_valid
    assert "Size mismatch" in err

def test_validation_out_of_bounds():
    boxes = [{"xmin": -10.0, "ymin": 0.0, "xmax": 50.0, "ymax": 50.0}]
    is_valid, err = validate_dataset_item(100, 100, 100, 100, boxes)
    assert not is_valid
    assert "Out of bounds" in err

def test_validate_prepared_yolo_dataset_bad_geometry():
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "manifests"))
        with open(os.path.join(td, "manifests", "preparation_metadata.json"), "w") as f:
            f.write('{"dataset_fingerprint": "' + '0'*64 + '"}')

        with open(os.path.join(td, "pothole.yaml"), "w") as f:
            yaml.dump({
                "names": {0: "pothole"},
                "path": td,
                "train": "images/train",
                "val": "images/val",
                "test": "images/test"
            }, f)

        for split in ["train", "val", "test"]:
            os.makedirs(os.path.join(td, "images", split))
            os.makedirs(os.path.join(td, "labels", split))

            with open(os.path.join(td, "images", split, "1.jpg"), "w") as f: f.write("fake")

        # Zero width
        with open(os.path.join(td, "labels", "train", "1.txt"), "w") as f: f.write("0 0.5 0.5 0.0 0.1")
        with open(os.path.join(td, "labels", "val", "1.txt"), "w") as f: f.write("0 0.5 0.5 0.1 0.1")
        with open(os.path.join(td, "labels", "test", "1.txt"), "w") as f: f.write("0 0.5 0.5 0.1 0.1")

        is_valid, err = validate_prepared_yolo_dataset(td)
        assert not is_valid
        assert "zero/negative width/height" in err

        # Out of bounds left edge
        with open(os.path.join(td, "labels", "train", "1.txt"), "w") as f: f.write("0 0.1 0.5 0.5 0.1")
        is_valid, err = validate_prepared_yolo_dataset(td)
        assert not is_valid
        assert "out of bounds coordinates" in err

        # Out of bounds right edge
        with open(os.path.join(td, "labels", "train", "1.txt"), "w") as f: f.write("0 0.9 0.5 0.5 0.1")
        is_valid, err = validate_prepared_yolo_dataset(td)
        assert not is_valid
        assert "out of bounds coordinates" in err

        # Valid
        with open(os.path.join(td, "labels", "train", "1.txt"), "w") as f: f.write("0 0.5 0.5 0.1 0.1")
        is_valid, err = validate_prepared_yolo_dataset(td)
        assert is_valid, err
