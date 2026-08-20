import xml.etree.ElementTree as ET
import math
from typing import List, Tuple, Dict, Any

def parse_rdd_xml(xml_path: str) -> Tuple[int, int, List[Dict[str, float]], str]:
    """
    Parses RDD2022 XML files.
    Returns: (xml_width, xml_height, list of D40 bounding boxes, error_string)
    """
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return 0, 0, [], "Malformed XML"

    root = tree.getroot()

    size_node = root.find("size")
    if size_node is None:
        return 0, 0, [], "Missing size node"

    try:
        xml_width = int(size_node.find("width").text) # type: ignore
        xml_height = int(size_node.find("height").text) # type: ignore
    except (AttributeError, ValueError, TypeError):
        return 0, 0, [], "Invalid or missing image dimensions"

    if xml_width <= 0 or xml_height <= 0:
        return 0, 0, [], "Dimensions must be positive"

    boxes = []

    for obj in root.findall("object"):
        name_node = obj.find("name")
        if name_node is None or not name_node.text:
            return xml_width, xml_height, [], "Missing class label"

        label = name_node.text.strip()

        # D00, D10, D20 are valid but negative samples
        if label in ["D00", "D10", "D20"]:
            continue

        if label == "D40":
            bndbox = obj.find("bndbox")
            if bndbox is None:
                return xml_width, xml_height, [], "D40 object missing bounding box"

            try:
                xmin = float(bndbox.find("xmin").text) # type: ignore
                ymin = float(bndbox.find("ymin").text) # type: ignore
                xmax = float(bndbox.find("xmax").text) # type: ignore
                ymax = float(bndbox.find("ymax").text) # type: ignore
            except (AttributeError, ValueError, TypeError):
                return xml_width, xml_height, [], "D40 bounding box coordinates missing or non-numeric"

            if not (math.isfinite(xmin) and math.isfinite(ymin) and math.isfinite(xmax) and math.isfinite(ymax)):
                return xml_width, xml_height, [], "D40 bounding box coordinates are NaN or infinite"

            boxes.append({
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax
            })
        else:
            return xml_width, xml_height, [], f"Unknown or invalid label: {label}"

    return xml_width, xml_height, boxes, ""

def convert_to_yolo(boxes: List[Dict[str, float]], width: int, height: int) -> List[str]:
    """
    Converts to YOLO normalized coordinates.
    Throws ValueError if out of bounds or invalid geometry. No clamping.
    """
    yolo_lines = []
    for b in boxes:
        dw = 1.0 / width
        dh = 1.0 / height

        xmin = b["xmin"]
        xmax = b["xmax"]
        ymin = b["ymin"]
        ymax = b["ymax"]

        if xmin >= xmax or ymin >= ymax:
            raise ValueError("Invalid bounding box area")

        if xmin < 0 or ymin < 0 or xmax > width or ymax > height:
            raise ValueError("Bounding box coordinates out of image bounds")

        x_center = (xmin + xmax) / 2.0
        y_center = (ymin + ymax) / 2.0
        w = xmax - xmin
        h = ymax - ymin

        x_norm = x_center * dw
        w_norm = w * dw
        y_norm = y_center * dh
        h_norm = h * dh

        if not (math.isfinite(x_norm) and math.isfinite(y_norm) and math.isfinite(w_norm) and math.isfinite(h_norm)):
            raise ValueError(f"Normalized coordinate is non-finite: x={x_norm}, y={y_norm}, w={w_norm}, h={h_norm}")

        if x_norm < 0 or x_norm > 1 or y_norm < 0 or y_norm > 1 or w_norm <= 0 or w_norm > 1 or h_norm <= 0 or h_norm > 1:
            raise ValueError(f"Normalized coordinate out of bounds: x={x_norm}, y={y_norm}, w={w_norm}, h={h_norm}")

        yolo_lines.append(f"0 {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}")

    return yolo_lines
