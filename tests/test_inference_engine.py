"""
Unit tests for DocVLMInferenceEngine
"""

import os
from src.inference_engine import DocVLMInferenceEngine
from src.dataset_loader import DocumentDatasetLoader


def test_inference_engine_structured_output(tmp_path):
    img_path = os.path.join(tmp_path, "sample_invoice.png")
    DocumentDatasetLoader.generate_synthetic_document(img_path)

    engine = DocVLMInferenceEngine()
    result = engine.extract_structured_json(img_path)

    assert result["is_valid_json"] is True
    assert "structured_data" in result
    assert "latency_sec" in result
    assert "tokens_per_sec" in result
