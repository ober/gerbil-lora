#!/bin/bash
# Option C: Train on a rented RunPod/Vast.ai GPU, then upload to Together AI.
#
# Usage:
#   1. Rent a GPU (RunPod A100 ~$1.50/hr or Vast.ai 4090 ~$0.30/hr)
#   2. SSH into the instance
#   3. scp this entire directory to the instance
#   4. Run this script
#   5. Download the adapter or upload to Together AI
#   6. Terminate the instance
#
# Total cost: ~$1-3 for training

set -euo pipefail

echo "=== Gerbil Scheme LoRA Training (RunPod/Vast.ai) ==="
echo ""

# ── Install dependencies ────────────────────────────────────────────
echo "Installing Unsloth and dependencies ..."
pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install -q --no-deps xformers trl peft accelerate bitsandbytes triton
pip install -q datasets

# ── Train ───────────────────────────────────────────────────────────
echo ""
echo "Starting training ..."
python3 train_unsloth.py

# ── Export to GGUF ──────────────────────────────────────────────────
echo ""
echo "Exporting to GGUF ..."
python3 merge_and_export.py q4_k_m

# ── Summary ─────────────────────────────────────────────────────────
echo ""
echo "=== Done! ==="
echo ""
echo "Your files:"
echo "  LoRA adapter: ./gerbil-lora-output/"
echo "  GGUF model:   ./gerbil-qwen-gguf/"
echo ""
echo "Next steps (pick one):"
echo ""
echo "  A) Download GGUF and use with Ollama locally:"
echo "     scp instance:~/gerbil-lora/gerbil-qwen-gguf/*.gguf ."
echo "     ollama create gerbil-qwen -f Modelfile"
echo ""
echo "  B) Upload adapter to Together AI for hosted inference:"
echo "     pip install together"
echo "     export TOGETHER_API_KEY=your-key"
echo "     python3 train_together.py upload-adapter  # (implement if needed)"
echo ""
echo "  C) Upload merged model to HuggingFace:"
echo "     huggingface-cli login"
echo "     python3 -c \""
echo "from unsloth import FastLanguageModel"
echo "model, tok = FastLanguageModel.from_pretrained('./gerbil-lora-output', max_seq_length=4096, load_in_4bit=True)"
echo "model.push_to_hub_merged('your-user/gerbil-qwen', tok)"
echo "\""
echo ""
echo "Remember to TERMINATE your GPU instance after downloading!"
