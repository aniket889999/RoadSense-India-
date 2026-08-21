from src.ml.rdd2022 import parse_rdd_xml, convert_to_yolo
import tempfile
import pytest
import os

def test_parse_rdd_xml_d40():
    xml_content = """<annotation>
        <size><width>600</width><height>400</height></size>
        <object><name>D40</name><bndbox><xmin>100</xmin><ymin>50</ymin><xmax>200</xmax><ymax>150</ymax></bndbox></object>
        <object><name>D10</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>20</xmax><ymax>20</ymax></bndbox></object>
    </annotation>"""
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        f.write(xml_content)
        f_name = f.name
    try:
        xml_w, xml_h, boxes, err = parse_rdd_xml(f_name)
        assert not err
        assert len(boxes) == 1
        assert boxes[0]["xmin"] == 100.0
    finally:
        os.unlink(f_name)

def test_parse_rdd_xml_no_d40():
    xml_content = """<annotation>
        <size><width>600</width><height>400</height></size>
        <object><name>D00</name><bndbox><xmin>100</xmin><ymin>50</ymin><xmax>200</xmax><ymax>150</ymax></bndbox></object>
    </annotation>"""
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        f.write(xml_content)
        f_name = f.name
    try:
        xml_w, xml_h, boxes, err = parse_rdd_xml(f_name)
        assert not err
        assert len(boxes) == 0 # viable negative
    finally:
        os.unlink(f_name)

def test_parse_rdd_xml_unknown_label():
    xml_content = """<annotation>
        <size><width>600</width><height>400</height></size>
        <object><name>D99</name><bndbox><xmin>100</xmin><ymin>50</ymin><xmax>200</xmax><ymax>150</ymax></bndbox></object>
    </annotation>"""
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        f.write(xml_content)
        f_name = f.name
    try:
        xml_w, xml_h, boxes, err = parse_rdd_xml(f_name)
        assert "Unknown or invalid label" in err
    finally:
        os.unlink(f_name)

def test_parse_rdd_xml_nan_coords():
    xml_content = """<annotation>
        <size><width>600</width><height>400</height></size>
        <object><name>D40</name><bndbox><xmin>NaN</xmin><ymin>50</ymin><xmax>200</xmax><ymax>150</ymax></bndbox></object>
    </annotation>"""
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        f.write(xml_content)
        f_name = f.name
    try:
        xml_w, xml_h, boxes, err = parse_rdd_xml(f_name)
        assert "NaN or infinite" in err
    finally:
        os.unlink(f_name)

def test_convert_to_yolo_out_of_bounds():
    boxes = [{"class": 0, "xmin": 1500, "ymin": 1500, "xmax": 2000, "ymax": 2000}]
    with pytest.raises(ValueError, match="out of image bounds"):
        convert_to_yolo(boxes, 1000, 1000)
