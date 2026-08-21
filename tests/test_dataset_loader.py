"""
Unit tests for DocumentDatasetLoader
"""

import os
from src.dataset_loader import DocumentDatasetLoader


def test_generate_synthetic_document(tmp_path):
    out_img = os.path.join(tmp_path, "test_doc.png")
    gt = DocumentDatasetLoader.generate_synthetic_document(
        out_img,
        vendor="Test Corp",
        subtotal=100.0,
        tax=10.0,
        total=110.0
    )
    assert os.path.exists(out_img)
    assert gt["vendor_name"] == "Test Corp"
    assert gt["total_amount"] == "$110.00"
