# RDD2022 India Data Card

## Source and Attribution
The dataset originates from the Global Road Damage Detection Challenge (CRDDC 2022).
**Important:** You must verify the license terms from the official source before commercial or public usage.

```text
Arya et al. (2024), “RDD2022: A multi-national image dataset for automatic road damage detection,”
Geoscience Data Journal, 11(4), 846–862. DOI: 10.1002/gdj3.260
Dataset DOI: 10.6084/m9.figshare.21431547.v1
```

## Data Mapping
- **D40 (Pothole)**: Successfully mapped to class `0` (`pothole`).
- **D00, D10, D20 (Cracks, etc.)**: These are strictly ignored.
- Images without `D40` objects but with valid annotations are treated as negative examples (background).

## Exclusions
The pipeline will automatically quarantine:
- XML files without matching images (and vice-versa).
- Bounding boxes with zero area or out-of-bounds coordinates.
- Corrupted images or malformed XML syntax.

## Data Splits
Splits (Train: 70%, Val: 15%, Test: 15%) are strictly partitioned by `group_id` provided in a manual manifest. Splitting purely randomly across images is prohibited to prevent data leakage from consecutive video frames representing the same physical potholes.
