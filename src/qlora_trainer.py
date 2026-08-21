"""
DocVLM QLoRA Fine-Tuning Engine
Implements 4-bit NF4 Quantization and Low-Rank Adaptation (PEFT) on Vision-Language Models.
"""

import os
import argparse
import torch
from typing import Dict, Any, Optional

try:
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        TrainingArguments,
        AutoModelForVision2Seq
    )
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        TaskType
    )
except ImportError:
    pass


class QLoRATrainer:
    """Orchestrates 4-bit QLoRA fine-tuning for Vision-Language Models."""

    def __init__(
        self,
        base_model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        output_dir: str = "lora_adapters"
    ):
        self.base_model_id = base_model_id
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.output_dir = output_dir

    def get_bnb_config(self) -> Any:
        """Configures 4-Bit NormalFloat (NF4) Quantization."""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        )

    def get_lora_config(self) -> Any:
        """Configures LoRA rank adapters targeting cross-attention and projection layers."""
        return LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            task_type="CAUSAL_LM"
        )

    def print_trainable_parameters(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Calculates and displays the exact parameter reduction achieved by QLoRA."""
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        pct = 100 * trainable_params / all_param if all_param > 0 else 0.0
        stats = {
            "trainable_params": trainable_params,
            "all_params": all_param,
            "trainable_percentage": round(pct, 4)
        }
        print(f"📊 [QLoRA Stats] Trainable: {trainable_params:,} | Total: {all_param:,} | Ratio: {pct:.2f}%")
        return stats

    def prepare_training_pipeline(self):
        """Prepares model and processor for quantized parameter-efficient training."""
        print(f"Loading Base VLM: {self.base_model_id} in 4-bit NF4 precision...")
        bnb_config = self.get_bnb_config()
        lora_config = self.get_lora_config()

        # In CPU-only environments, fall back gracefully to float32 model loading
        device_map = "auto" if torch.cuda.is_available() else "cpu"

        try:
            model = AutoModelForVision2Seq.from_pretrained(
                self.base_model_id,
                quantization_config=bnb_config if torch.cuda.is_available() else None,
                device_map=device_map,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True
            )
            processor = AutoProcessor.from_pretrained(self.base_model_id, trust_remote_code=True)
        except Exception as e:
            print(f"Note on base model initialization: {e}")
            return None, None

        if torch.cuda.is_available():
            model = prepare_model_for_kbit_training(model)

        model = get_peft_model(model, lora_config)
        self.print_trainable_parameters(model)

        return model, processor


def parse_args():
    parser = argparse.ArgumentParser(description="DocVLM QLoRA Fine-Tuning")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="lora_adapters")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trainer = QLoRATrainer(lora_r=args.lora_r, output_dir=args.output_dir)
    print(f"QLoRA Trainer initialized for {trainer.base_model_id}.")
