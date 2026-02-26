#!/usr/bin/env python3
"""
Merge LoRA adapter into base model and export to GGUF for Ollama.

Prerequisites:
  pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

Usage:
  python3 merge_and_export.py [quantization]

Quantization options (default: q4_k_m):
  q4_k_m   — 4-bit, good balance of quality/size (~4.5GB)
  q5_k_m   — 5-bit, better quality (~5.5GB)
  q8_0     — 8-bit, near-lossless (~7.5GB)
  f16      — full float16, no quantization (~14GB)

After export:
  ollama create gerbil-qwen -f Modelfile
  ollama run gerbil-qwen
"""

import sys
import os

ADAPTER_DIR = os.path.join(os.path.dirname(__file__), "gerbil-lora-output")
GGUF_DIR = os.path.join(os.path.dirname(__file__), "gerbil-qwen-gguf")
MAX_SEQ_LENGTH = 4096
DEFAULT_QUANT = "q4_k_m"


def main():
    from unsloth import FastLanguageModel

    quant = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUANT
    valid_quants = ["q4_k_m", "q5_k_m", "q8_0", "f16"]
    if quant not in valid_quants:
        print(f"Invalid quantization: {quant}")
        print(f"Valid options: {', '.join(valid_quants)}")
        sys.exit(1)

    # ── Load fine-tuned model ───────────────────────────────────────
    print(f"Loading adapter from {ADAPTER_DIR} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ADAPTER_DIR,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    # ── Export to GGUF ──────────────────────────────────────────────
    print(f"Exporting to GGUF ({quant}) ...")
    os.makedirs(GGUF_DIR, exist_ok=True)

    model.save_pretrained_gguf(
        GGUF_DIR,
        tokenizer,
        quantization_method=quant,
    )

    # Find the output file
    gguf_files = [f for f in os.listdir(GGUF_DIR) if f.endswith(".gguf")]
    if gguf_files:
        gguf_path = os.path.join(GGUF_DIR, gguf_files[0])
        size_mb = os.path.getsize(gguf_path) / 1024 / 1024
        print(f"\nGGUF saved: {gguf_path} ({size_mb:.0f} MB)")
    else:
        print(f"\nGGUF files saved to {GGUF_DIR}/")

    print(f"\nNext steps:")
    print(f"  ollama create gerbil-qwen -f Modelfile")
    print(f"  ollama run gerbil-qwen 'How do I parse JSON in Gerbil?'")


if __name__ == "__main__":
    main()
