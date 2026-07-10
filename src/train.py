"""QLoRA fine-tune on the fantasy draft dataset.

Sized for a 16GB RTX 5060 Ti (Blackwell): 4-bit NF4 base weights via
bitsandbytes, LoRA adapters on attention + MLP projections, bf16 compute.
Gemma-3-4B-IT in 4-bit + adapters + optimizer states fits comfortably.

Run:
    python src/train.py
    python src/train.py --base google/gemma-3-1b-it --epochs 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "train.jsonl"
OUT = ROOT / "checkpoints" / "fantasy-draft-lora"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="google/gemma-3-4b-it")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-seq-len", type=int, default=1024)
    args = p.parse_args()

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",  # recommended for Gemma
    )
    model.config.use_cache = False

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    ds = load_dataset("json", data_files=str(DATA), split="train")
    split = ds.train_test_split(test_size=0.03, seed=42)

    cfg = SFTConfig(
        output_dir=str(OUT),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,   # effective batch 16
        gradient_checkpointing=True,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="epoch",
        max_length=args.max_seq_len,
        packing=False,
        report_to="none",
        optim="paged_adamw_8bit",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=cfg,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        peft_config=lora,
    )

    trainer.train()
    trainer.save_model(str(OUT / "final"))
    tokenizer.save_pretrained(str(OUT / "final"))
    print(f"Adapter saved to {OUT / 'final'}")


if __name__ == "__main__":
    main()
