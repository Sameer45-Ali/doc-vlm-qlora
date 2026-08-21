"""
DocVLM Utilities Module
High-resolution image preprocessing, JSON schema validation, and key-value extraction metrics.
"""

import os
import re
import json
from typing import Tuple, Dict, Any, Optional
from PIL import Image, ImageOps


def load_and_preprocess_image(image_path: str, max_size: int = 1024) -> Image.Image:
    """
    Loads an image, corrects EXIF orientation, converts to RGB, and resizes if needed.
    
    Args:
        image_path: Path to the image file.
        max_size: Maximum width/height dimension for memory-efficient VLM encoding.
        
    Returns:
        Preprocessed PIL Image.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")

    img = Image.open(image_path)
    # Fix orientation from camera metadata
    img = ImageOps.exif_transpose(img)
    # Ensure standard RGB 3-channel
    img = img.convert("RGB")

    # Resize proportionally if exceeding max dimension
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / float(max(w, h))
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    return img


def extract_json_from_response(raw_response: str) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """
    Extracts and parses JSON object from model generation text with robust regex matching.
    
    Returns:
        Tuple of (is_valid_json: bool, parsed_dict: Optional[dict], raw_or_cleaned_string: str)
    """
    if not raw_response or not raw_response.strip():
        return False, None, "Empty response"

    text = raw_response.strip()

    # Attempt 1: Direct JSON parsing
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return True, data, text
    except json.JSONDecodeError:
        pass

    # Attempt 2: Extract code block ```json ... ```
    json_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_block_match:
        try:
            data = json.loads(json_block_match.group(1))
            if isinstance(data, dict):
                return True, data, json_block_match.group(1)
        except json.JSONDecodeError:
            pass

    # Attempt 3: Match outermost curly braces { ... }
    brace_match = re.search(r"(\{.*\})", text, re.DOTALL)
    if brace_match:
        try:
            data = json.loads(brace_match.group(1))
            if isinstance(data, dict):
                return True, data, brace_match.group(1)
        except json.JSONDecodeError:
            pass

    return False, None, text


def compute_key_value_f1(pred_dict: Dict[str, Any], true_dict: Dict[str, Any]) -> Dict[str, float]:
    """
    Computes precision, recall, and F1-score across extracted key-value pairs.
    """
    if not pred_dict or not true_dict:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "exact_match": 0.0}

    pred_flat = {str(k).lower().strip(): str(v).lower().strip() for k, v in pred_dict.items() if not isinstance(v, (dict, list))}
    true_flat = {str(k).lower().strip(): str(v).lower().strip() for k, v in true_dict.items() if not isinstance(v, (dict, list))}

    if not true_flat:
        return {"precision": 1.0 if not pred_flat else 0.0, "recall": 1.0, "f1": 1.0, "exact_match": 1.0}

    correct_keys = 0
    correct_values = 0

    for k, v in pred_flat.items():
        if k in true_flat:
            correct_keys += 1
            if true_flat[k] == v:
                correct_values += 1

    precision = correct_values / max(len(pred_flat), 1)
    recall = correct_values / max(len(true_flat), 1)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    exact_match = 1.0 if pred_flat == true_flat else 0.0

    return {
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1": round(f1 * 100, 2),
        "exact_match": round(exact_match * 100, 2)
    }


if __name__ == "__main__":
    print("DocVLM utils module loaded successfully.")
