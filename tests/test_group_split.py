from src.ml.group_split import generate_safe_split, parse_group_csv, verify_no_leakage
import tempfile
import os

def test_generate_safe_split_exact_proportions():
    # 100 images
    images = []
    group_mapping = {}
    pos_counts = {}

    # g1: 70 images
    for i in range(70):
        img = f"g1_{i}.jpg"
        images.append(img)
        group_mapping[img] = "g1"
        pos_counts[img] = 1

    # g2: 15 images
    for i in range(15):
        img = f"g2_{i}.jpg"
        images.append(img)
        group_mapping[img] = "g2"
        pos_counts[img] = 1

    # g3: 15 images
    for i in range(15):
        img = f"g3_{i}.jpg"
        images.append(img)
        group_mapping[img] = "g3"
        pos_counts[img] = 1

    splits, errors = generate_safe_split(images, group_mapping, pos_counts, seed=42)
    assert not errors

    counts = {"train": 0, "val": 0, "test": 0}
    for s in splits.values():
        counts[s] += 1

    # Must produce exactly 70, 15, 15
    assert counts["train"] == 70
    assert counts["val"] == 15
    assert counts["test"] == 15

def test_parse_group_csv_strict():
    csv_content = "image_relpath,group_id\nimg_0.jpg,g1\nimg_0.jpg,g2\n"
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        f.write(csv_content)
        f_name = f.name
    try:
        mapping, errors = parse_group_csv(f_name, set(["img_0.jpg"]))
        assert errors
        assert "duplicate path" in errors[0]
    finally:
        os.unlink(f_name)
