# Gerbil Scheme LoRA Training — Complete Guide

## Status

- [x] Generate training data from cookbooks, docs, source code
- [ ] Choose hosting path (Together AI vs Local Ollama)
- [ ] Train the LoRA
- [ ] Deploy and connect to OpenCode

---

## Training Data (DONE)

Generated **5,970 training entries** in `~/mine/gerbil-lora/`:

| File | Format | Size |
|------|--------|------|
| `training_data.jsonl` | ChatML/ShareGPT | 8.9 MB |
| `training_data_alpaca.jsonl` | Alpaca JSONL | 6.2 MB |
| `training_data_alpaca.json` | Alpaca JSON array | 6.3 MB |

Regenerate anytime: `python3 convert_training_data.py`

---

## Option A: Together AI (Recommended — no GPU needed)

### Cost
- Fine-tuning: ~$2-5 one-time
- Inference: $0.20/M input + $0.20/M output tokens (~$0.20/heavy coding day)
- No idle costs

### Steps

#### 1. Create account and get API key
- Sign up at https://together.ai
- Get API key from dashboard

#### 2. Install CLI
```bash
pip install together
export TOGETHER_API_KEY="your-key-here"
```

#### 3. Upload training data
```bash
# Together AI expects ChatML/ShareGPT format
together files upload ~/mine/gerbil-lora/training_data.jsonl
# Note the file ID returned (e.g., file-abc123)
```

#### 4. Start fine-tuning
```bash
together fine-tuning create \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --training-file file-abc123 \
  --n-epochs 3 \
  --learning-rate 1e-5 \
  --batch-size 4 \
  --lora \
  --lora-r 16 \
  --lora-alpha 32
```

#### 5. Monitor training
```bash
together fine-tuning list
together fine-tuning retrieve ft-job-xxxxx
# Takes ~30-60 minutes for this dataset size
```

#### 6. Test the model
```bash
together chat completions create \
  --model your-org/gerbil-qwen-lora \
  --message "How do I parse JSON in Gerbil Scheme?"
```

#### 7. Configure OpenCode
In your OpenCode config (usually `~/.opencode/config.json` or similar):
```json
{
  "provider": "openai-compatible",
  "base_url": "https://api.together.xyz/v1",
  "api_key": "your-together-api-key",
  "model": "your-org/gerbil-qwen-lora"
}
```

---

## Option B: Unsloth + Ollama (Free — needs GPU)

### Requirements
- GPU with 16GB+ VRAM (RTX 4090, 3090, A100, etc.)
- OR Mac with 32GB+ unified memory
- ~20GB disk space

### Steps

#### 1. Install Unsloth
```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps xformers trl peft accelerate bitsandbytes triton
```

#### 2. Run training script
Save this as `~/mine/gerbil-lora/train_unsloth.py`:

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import json

# ── Config ──
MODEL_NAME = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"
OUTPUT_DIR = "./gerbil-lora-output"
MAX_SEQ_LENGTH = 4096
EPOCHS = 3

# ── Load model ──
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,  # auto-detect
    load_in_4bit=True,
)

# ── Apply LoRA ──
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# ── Load training data ──
def format_chatml(example):
    """Convert our ChatML format to the tokenizer's chat template."""
    messages = example["conversations"]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}

dataset = load_dataset("json", data_files="training_data.jsonl", split="train")
dataset = dataset.map(format_chatml)

# ── Train ──
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=2,
    packing=True,
    args=TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=EPOCHS,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        optim="adamw_8bit",
        save_strategy="epoch",
        seed=42,
    ),
)

trainer.train()

# ── Save LoRA adapter ──
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nLoRA adapter saved to {OUTPUT_DIR}/")
```

Run it:
```bash
cd ~/mine/gerbil-lora
python3 train_unsloth.py
# Takes ~20-40 minutes on a single A100/4090
```

#### 3. Merge and convert to GGUF for Ollama
```python
# Save this as merge_and_export.py
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./gerbil-lora-output",
    max_seq_length=4096,
    dtype=None,
    load_in_4bit=True,
)

# Export to GGUF (Ollama format)
# Choose quantization: q4_k_m is good balance of quality/size (~4.5GB)
model.save_pretrained_gguf(
    "./gerbil-qwen-gguf",
    tokenizer,
    quantization_method="q4_k_m",
)
print("GGUF model saved to ./gerbil-qwen-gguf/")
```

```bash
python3 merge_and_export.py
```

#### 4. Import into Ollama
Create `~/mine/gerbil-lora/Modelfile`:
```
FROM ./gerbil-qwen-gguf/unsloth.Q4_K_M.gguf

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""

SYSTEM "You are an expert in Gerbil Scheme, a dialect of Scheme built on Gambit. You provide accurate, idiomatic Gerbil code with correct imports, function names, and arities."

PARAMETER temperature 0.2
PARAMETER num_ctx 4096
```

```bash
ollama create gerbil-qwen -f Modelfile
ollama run gerbil-qwen "How do I parse JSON in Gerbil?"
```

#### 5. Configure OpenCode
```json
{
  "provider": "ollama",
  "base_url": "http://localhost:11434/v1",
  "model": "gerbil-qwen"
}
```

---

## Option C: Rent a GPU for Training Only, Then Use Together AI

Best of both worlds — cheap training, zero-infra hosting.

#### 1. Rent a GPU on RunPod or Vast.ai
- RunPod: A100 40GB ~$1.50/hr → ~$1 total for training
- Vast.ai: 4090 24GB ~$0.30/hr → ~$0.20 total for training

#### 2. SSH in, run the Unsloth script (Option B steps 1-2)

#### 3. Upload the LoRA adapter to Together AI
```bash
together fine-tuning upload-model \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --adapter-path ./gerbil-lora-output
```

#### 4. Configure OpenCode to use Together AI (Option A step 7)

#### 5. Terminate the rented GPU

Total cost: ~$1-3 one-time training + $0.20/M tokens ongoing.

---

## Comparison

| | Together AI | Unsloth + Ollama | Rent GPU + Together |
|---|---|---|---|
| GPU needed? | No | Yes (16GB+) | Rented (~$1) |
| Training cost | ~$2-5 | $0 (your GPU) | ~$1 (rented) |
| Inference cost | $0.20/M tokens | $0 (local) | $0.20/M tokens |
| Monthly cost (moderate use) | ~$5-10 | $0 + electricity | ~$5-10 |
| Setup difficulty | Easy | Medium | Medium |
| Latency | ~1-2s | ~0.1-0.5s (local) | ~1-2s |
| Privacy | Data sent to API | Fully local | Data sent to API |

---

## Verifying the Fine-Tune Worked

Test these prompts — a base Qwen model will struggle with Gerbil-specific answers:

```
1. "How do I iterate over a hash table in Gerbil Scheme?"
   Expected: mentions (import :std/iter), uses (for ((values k v) (in-hash ht)) ...)

2. "What's the difference between hash-get and hash-ref in Gerbil?"
   Expected: hash-get is strictly 2-arity, returns #f; hash-ref throws on missing key

3. "Show me how to define a custom error class in Gerbil"
   Expected: uses deferror-class, mentions the defraise/context gotcha

4. "How do I build a static executable in Gerbil?"
   Expected: mentions build.ss with (exe: "main"), GERBIL_LOADPATH, gerbil build

5. "What's wrong with passing u8vector to a (pointer void) FFI parameter?"
   Expected: mentions Gambit passes raw Scheme object header, recommends scheme-object type
```

---

## Next Steps

1. Pick Option A or B (or C)
2. Run the training
3. Verify with test prompts above
4. Configure OpenCode
5. Iterate — add more training data from your sessions, retrain
