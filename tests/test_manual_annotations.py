from src.manual_annotations import parse_manual_csv

def test_parse_manual_csv_valid():
    csv_data = b"incident_id,frame_index,x_min,y_min,x_max,y_max,label,note\nPOT-001,5,10,10,50,50,pothole,\n"
    rows, errors = parse_manual_csv(csv_data, 100, 100, allowed_frames=[5])
    assert not errors
    assert len(rows) == 1

def test_unsampled_frame_index_rejected():
    csv_data = b"incident_id,frame_index,x_min,y_min,x_max,y_max,label,note\nPOT-001,6,10,10,50,50,pothole,\n"
    rows, errors = parse_manual_csv(csv_data, 100, 100, allowed_frames=[5])
    assert errors
    assert "not in the sampled frames manifest" in errors[0]

def test_empty_id_rejected():
    csv_data = b"incident_id,frame_index,x_min,y_min,x_max,y_max,label,note\n,5,10,10,50,50,pothole,\n"
    rows, errors = parse_manual_csv(csv_data, 100, 100, allowed_frames=[5])
    assert errors
    assert "incident_id is required" in errors[0]

def test_header_only_rejected():
    csv_data = b"incident_id,frame_index,x_min,y_min,x_max,y_max,label,note\n"
    rows, errors = parse_manual_csv(csv_data, 100, 100, allowed_frames=[5])
    assert errors
    assert "no data rows" in errors[0]

def test_header_plus_blank_line():
    csv_data = b"incident_id,frame_index,x_min,y_min,x_max,y_max,label,note\n\n"
    rows, errors = parse_manual_csv(csv_data, 100, 100, allowed_frames=[5])
    assert errors
    assert "no data rows" in errors[0]

def test_malformed_utf8_rejected():
    csv_data = b"\xff\xfeincident_id"
    rows, errors = parse_manual_csv(csv_data, 100, 100, allowed_frames=[5])
    assert errors
    assert "not valid UTF-8 encoding" in errors[0]

def test_extra_columns_rejected():
    csv_data = b"incident_id,frame_index,x_min,y_min,x_max,y_max,label,note\nPOT-001,5,10,10,50,50,pothole,,0.9\n"
    rows, errors = parse_manual_csv(csv_data, 100, 100, allowed_frames=[5])
    assert errors
    assert "Extra values found" in errors[0]

def test_missing_note_rejected():
    csv_data = b"incident_id,frame_index,x_min,y_min,x_max,y_max,label,note\nPOT-001,5,10,10,50,50,pothole\n"
    rows, errors = parse_manual_csv(csv_data, 100, 100, allowed_frames=[5])
    assert errors
    assert "Missing values found" in errors[0]

def test_duplicate_headers_rejected():
    csv_data = b"incident_id,frame_index,x_min,y_min,x_max,y_max,label,note,note\nPOT-001,5,10,10,50,50,pothole,,\n"
    rows, errors = parse_manual_csv(csv_data, 100, 100, allowed_frames=[5])
    assert errors
    assert "Invalid headers" in errors[0]

def test_unclosed_quoted_field():
    # The 'strict=True' param will catch unclosed quotes or newlines in unquoted fields
    csv_data = b"incident_id,frame_index,x_min,y_min,x_max,y_max,label,note\n\"POT-001,5,10,10,50,50,pothole,\n"
    rows, errors = parse_manual_csv(csv_data, 100, 100, allowed_frames=[5])
    assert errors
    assert "CSV formatting error" in errors[0]

def test_invalid_text_after_closing_quote():
    csv_data = b"incident_id,frame_index,x_min,y_min,x_max,y_max,label,note\n\"POT-001\"bad,5,10,10,50,50,pothole,\n"
    rows, errors = parse_manual_csv(csv_data, 100, 100, allowed_frames=[5])
    assert errors
    assert "CSV formatting error" in errors[0]

def test_oversized_note_field():
    import csv
    old_limit = csv.field_size_limit()
    csv.field_size_limit(100) # Temporarily lower limit to trigger exception for testing
    try:
        large_note = "A" * 200
        csv_data = f"incident_id,frame_index,x_min,y_min,x_max,y_max,label,note\nPOT-001,5,10,10,50,50,pothole,{large_note}\n".encode('utf-8')
        rows, errors = parse_manual_csv(csv_data, 100, 100, allowed_frames=[5])
        assert errors
        assert "CSV formatting error" in errors[0]
    finally:
        csv.field_size_limit(old_limit)

def test_leading_trailing_spaces_incident_id():
    csv_data = b"incident_id,frame_index,x_min,y_min,x_max,y_max,label,note\n POT-001 ,5,10,10,50,50,pothole,\n"
    rows, errors = parse_manual_csv(csv_data, 100, 100, allowed_frames=[5])
    assert errors
    assert "incident_id ' POT-001 ' is invalid" in errors[0]

def test_leading_trailing_spaces_label():
    csv_data = b"incident_id,frame_index,x_min,y_min,x_max,y_max,label,note\nPOT-001,5,10,10,50,50, pothole ,\n"
    rows, errors = parse_manual_csv(csv_data, 100, 100, allowed_frames=[5])
    assert errors
    assert "must be exactly 'pothole'. Found ' pothole '" in errors[0]
