from src.ml.group_split import generate_safe_split
import time

def test_generate_safe_split_scalability():
    # 5000 images in 5000 groups
    images = []
    group_mapping = {}
    pos_counts = {}

    for i in range(5000):
        img = f"img_{i}.jpg"
        images.append(img)
        group_mapping[img] = f"g{i}"
        pos_counts[img] = 1 if i < 100 else 0

    start_time = time.time()
    splits, errors = generate_safe_split(images, group_mapping, pos_counts, seed=42)
    end_time = time.time()

    assert not errors
    assert len(splits) == 5000
    assert end_time - start_time < 5.0 # Should be very fast, well under 1 second
