"""
DocVLM Dataset Loader Module
Handles visual document dataset formatting, image-text tokenization, and sample document synthesis.
"""

import os
import json
import random
from typing import List, Dict, Any, Optional
from PIL import Image, ImageDraw, ImageFont


# Default System Instruction for Structured Extraction
DEFAULT_PROMPT = """You are DocVLM, an expert document intelligence AI. Analyze the uploaded document image and extract all structured data into a valid JSON object containing:
- vendor_name (string)
- invoice_number (string)
- invoice_date (string, YYYY-MM-DD format)
- currency (string)
- subtotal (string)
- tax_amount (string)
- total_amount (string)
- payment_method (string)
- line_items (list of objects with description, quantity, unit_price, total)

Output ONLY the JSON object, nothing else."""


class DocumentDatasetLoader:
    """Manages loading and formatting multimodal document extraction datasets."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir

    def format_conversation(self, image_path: str, ground_truth_json: Dict[str, Any], prompt: str = DEFAULT_PROMPT) -> Dict[str, Any]:
        """
        Formats an image and JSON pair into standard VLM conversational format.
        """
        json_str = json.dumps(ground_truth_json, indent=2)
        return {
            "image_path": image_path,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": prompt}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": json_str}
                    ]
                }
            ]
        }

    @staticmethod
    def generate_synthetic_document(
        output_image_path: str,
        vendor: str = "TechNova Solutions",
        inv_no: str = "INV-2026-8941",
        date: str = "2026-08-15",
        subtotal: float = 450.00,
        tax: float = 40.50,
        total: float = 490.50,
        items: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Generates a clean synthetic visual document invoice image with ground-truth JSON metadata.
        """
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        width, height = 800, 1050
        img = Image.new("RGB", (width, height), color=(250, 250, 252))
        draw = ImageDraw.Draw(img)

        # Header Box
        draw.rectangle([(40, 40), (width - 40, 140)], fill=(30, 41, 59))
        draw.text((60, 60), "INVOICE / RECEIPT", fill=(255, 255, 255))
        draw.text((60, 95), f"Vendor: {vendor}", fill=(148, 163, 184))
        draw.text((width - 250, 65), f"Invoice #: {inv_no}", fill=(255, 255, 255))
        draw.text((width - 250, 95), f"Date: {date}", fill=(148, 163, 184))

        # Table Header
        draw.rectangle([(40, 180), (width - 40, 220)], fill=(226, 232, 240))
        draw.text((60, 195), "Description", fill=(30, 41, 59))
        draw.text((450, 195), "Qty", fill=(30, 41, 59))
        draw.text((550, 195), "Unit Price", fill=(30, 41, 59))
        draw.text((680, 195), "Total", fill=(30, 41, 59))

        if items is None:
            items = [
                {"description": "Cloud GPU Compute Hours (A100)", "quantity": 10, "unit_price": "$35.00", "total": "$350.00"},
                {"description": "Vector Database Storage (100GB)", "quantity": 1, "unit_price": "$100.00", "total": "$100.00"}
            ]

        y = 240
        for item in items:
            draw.text((60, y), str(item["description"]), fill=(51, 65, 85))
            draw.text((450, y), str(item["quantity"]), fill=(51, 65, 85))
            draw.text((550, y), str(item["unit_price"]), fill=(51, 65, 85))
            draw.text((680, y), str(item["total"]), fill=(51, 65, 85))
            draw.line([(40, y + 30), (width - 40, y + 30)], fill=(241, 245, 249), width=1)
            y += 45

        # Summary Totals Box
        draw.rectangle([(450, y + 40), (width - 40, y + 160)], fill=(248, 250, 252), outline=(203, 213, 225))
        draw.text((470, y + 55), "Subtotal:", fill=(100, 116, 139))
        draw.text((680, y + 55), f"${subtotal:.2f}", fill=(30, 41, 59))

        draw.text((470, y + 85), "Tax (9%):", fill=(100, 116, 139))
        draw.text((680, y + 85), f"${tax:.2f}", fill=(30, 41, 59))

        draw.text((470, y + 120), "TOTAL AMOUNT:", fill=(15, 23, 42))
        draw.text((680, y + 120), f"${total:.2f}", fill=(37, 99, 235))

        # Footer
        draw.text((60, height - 60), "Thank you for your business! Payment Method: Corporate Credit Card", fill=(148, 163, 184))

        img.save(output_image_path, quality=95)

        ground_truth = {
            "vendor_name": vendor,
            "invoice_number": inv_no,
            "invoice_date": date,
            "currency": "USD",
            "subtotal": f"${subtotal:.2f}",
            "tax_amount": f"${tax:.2f}",
            "total_amount": f"${total:.2f}",
            "payment_method": "Corporate Credit Card",
            "line_items": items
        }

        json_path = output_image_path.replace(".png", ".json").replace(".jpg", ".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(ground_truth, f, indent=2)

        return ground_truth


def create_sample_dataset_suite(output_dir: str = "samples", num_samples: int = 5):
    """Generates a suite of sample visual invoices for testing and fine-tuning demo."""
    os.makedirs(output_dir, exist_ok=True)
    vendors = ["CloudScale Systems", "Vertex AI Labs", "Apex Data Solutions", "Quantum Robotics", "NeuralCore Technologies"]
    
    generated = []
    for i in range(num_samples):
        v = vendors[i % len(vendors)]
        img_file = os.path.join(output_dir, f"sample_invoice_{i+1:02d}.png")
        sub = round(random.uniform(150.0, 950.0), 2)
        tx = round(sub * 0.09, 2)
        tot = round(sub + tx, 2)
        gt = DocumentDatasetLoader.generate_synthetic_document(
            img_file,
            vendor=v,
            inv_no=f"INV-2026-{1000 + i*47}",
            date=f"2026-08-{10 + (i % 18):02d}",
            subtotal=sub,
            tax=tx,
            total=tot
        )
        generated.append({"image": img_file, "ground_truth": gt})
        
    print(f"Generated {len(generated)} sample document pairs in '{output_dir}/'.")
    return generated


if __name__ == "__main__":
    create_sample_dataset_suite()
