"""
Unit tests for DocVLM utils
"""

from src.utils import extract_json_from_response, compute_key_value_f1


def test_extract_json_valid_direct():
    raw = '{"vendor_name": "TechCorp", "total": "$100.00"}'
    is_valid, data, _ = extract_json_from_response(raw)
    assert is_valid is True
    assert data["vendor_name"] == "TechCorp"


def test_extract_json_markdown_block():
    raw = 'Here is the extracted information:\n```json\n{"vendor_name": "TechCorp", "total": "$100.00"}\n```\nDone.'
    is_valid, data, _ = extract_json_from_response(raw)
    assert is_valid is True
    assert data["vendor_name"] == "TechCorp"


def test_compute_key_value_f1():
    pred = {"vendor": "Acme", "total": "$50"}
    true = {"vendor": "Acme", "total": "$50"}
    metrics = compute_key_value_f1(pred, true)
    assert metrics["f1"] == 100.0
    assert metrics["exact_match"] == 100.0
