import csv
import re
from io import StringIO
from typing import List, Tuple
from src.contracts import ManualAnnotationRow

def parse_manual_csv(file_bytes: bytes, video_width: int, video_height: int, allowed_frames: List[int]) -> Tuple[List[ManualAnnotationRow], List[str]]:
    try:
        csv_str = file_bytes.decode('utf-8')
    except UnicodeDecodeError:
        return [], ["CSV file is not valid UTF-8 encoding."]

    errors = []
    rows = []

    expected_columns_list = [
        "incident_id",
        "frame_index",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
        "label",
        "note",
    ]

    try:
        reader = csv.reader(StringIO(csv_str), strict=True)
        rows_data = list(reader)
    except csv.Error as e:
        return [], [f"CSV formatting error: {str(e)}"]

    if not rows_data:
        return [], ["CSV file is empty or lacks headers."]

    header = rows_data[0]
    if header != expected_columns_list:
        return [], [f"Invalid headers. Expected exactly: {','.join(expected_columns_list)}"]

    if len(rows_data) < 2:
        return [], ["CSV file has no data rows."]

    incident_id_pattern = re.compile(r"^[A-Za-z0-9-]+$")

    for i, row in enumerate(rows_data[1:]):
        line_num = i + 2
        # Skip truly empty rows that don't even have an empty string
        if not row:
            continue

        if len(row) > len(expected_columns_list):
            errors.append(f"Line {line_num}: Extra values found.")
            continue
        if len(row) < len(expected_columns_list):
            errors.append(f"Line {line_num}: Missing values found.")
            continue

        try:
            # DO NOT strip incident_id or label before validation
            incident_id = row[0]
            if not incident_id:
                errors.append(f"Line {line_num}: incident_id is required and cannot be empty.")
                continue

            if not incident_id_pattern.match(incident_id):
                errors.append(f"Line {line_num}: incident_id '{incident_id}' is invalid. Use only alphanumeric characters and dashes.")

            frame_index = int(row[1])
            if frame_index not in allowed_frames:
                errors.append(f"Line {line_num}: frame_index {frame_index} is not in the sampled frames manifest.")

            x_min = int(row[2])
            y_min = int(row[3])
            x_max = int(row[4])
            y_max = int(row[5])

            label = row[6]
            if label != "pothole":
                errors.append(f"Line {line_num}: label must be exactly 'pothole'. Found '{label}'.")

            # note can be trimmed after validity
            note = row[7].strip()

            if not (0 <= x_min < x_max <= video_width):
                errors.append(f"Line {line_num}: x coordinates invalid or out of bounds (0-{video_width}).")

            if not (0 <= y_min < y_max <= video_height):
                errors.append(f"Line {line_num}: y coordinates invalid or out of bounds (0-{video_height}).")

            rows.append(ManualAnnotationRow(
                incident_id=incident_id,
                frame_index=frame_index,
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
                label=label,
                note=note
            ))

        except (ValueError, TypeError) as e:
            errors.append(f"Line {line_num}: invalid data format. {str(e)}")

    if errors:
        return [], errors

    if not rows:
        return [], ["CSV file has no data rows."]

    return rows, []
