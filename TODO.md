# Gerbil Scheme LoRA Training — Together AI

## Status

- [x] Generate training data (5,970 entries from cookbooks, docs, source code)
- [x] Upload training data to Together AI
- [x] Start fine-tuning job (ft-5f979336-8831)
- [x] Training completed (~7 min, 63 steps, 3 epochs, 2.1M tokens)
- [ ] Verify model with test prompts
- [ ] Download adapter for local/portable use
- [ ] Deploy and connect to OpenCode

---

## Training Data

Generated **5,970 training entries** in `~/mine/gerbil-lora/`:

| File | Format | Size |
|------|--------|------|
| `training_data_together.jsonl` | Together AI (messages) | 8.5 MB |
| `training_data.jsonl` | ChatML/ShareGPT | 8.9 MB |
| `training_data_alpaca.jsonl` | Alpaca JSONL | 6.2 MB |

Regenerate: `python3 convert_training_data.py`

---

## Step 1: Setup (DONE)

```bash
pip install together
export TOGETHER_API_KEY="your-key-here"
```

## Step 2: Upload (DONE)

```bash
together files upload ~/mine/gerbil-lora/training_data_together.jsonl
# Returns file ID: file-f2144ecd-69ce-4dbb-bfda-e6a2a01ab2ed
```

## Step 3: Train (DONE — running)

```bash
python3 train_together.py train
# Or manually:
together fine-tuning create \
  --model Qwen/Qwen2.5-7B-Instruct \
  --training-file file-f2144ecd-69ce-4dbb-bfda-e6a2a01ab2ed \
  --n-epochs 3 \
  --learning-rate 1e-5 \
  --batch-size 8 \
  --lora
```

Job ID: `ft-5f979336-8831`
Model: `jaimef_2515/Qwen2.5-7B-Instruct-74900ead`

Training completed in ~7 minutes: 63 steps, 3 epochs, 2.1M tokens trained.

## Step 4: Monitor

```bash
python3 train_together.py status
# Or: together fine-tuning retrieve ft-5f979336-8831
# Takes ~30-60 minutes
```

## Step 5: Test on Together AI

```bash
python3 train_together.py test
# Or:
python3 verify_model.py \
  --base-url https://api.together.xyz/v1 \
  --model jaimef_2515/Qwen2.5-7B-Instruct-74900ead \
  --api-key $TOGETHER_API_KEY -v
```

## Step 6: Use with OpenCode (hosted on Together AI)

Point OpenCode at Together AI's API:

```
Base URL: https://api.together.xyz/v1
API Key:  $TOGETHER_API_KEY
Model:    jaimef_2515/Qwen2.5-7B-Instruct-74900ead
```

Pay-per-token, no idle costs (~$0.20/M tokens).

---

## Step 7: Download for Local/Portable Use (optional)

The trained LoRA adapter is yours. Download it to use anywhere:

```bash
together fine-tuning download ft-5f979336-8831 --output ./together-adapter
# Adapter also available at:
# s3://together-dev/finetune/.../ft-5f979336-8831_adapter-2026-02-26-02-46-54
```

### Run locally with Ollama

Merge the adapter with the base model and convert to GGUF:

```bash
# Needs a GPU or lots of RAM for the merge step
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

python3 merge_and_export.py \
  --adapter ./together-adapter \
  --base Qwen/Qwen2.5-7B-Instruct \
  --quant q4_k_m

ollama create gerbil-qwen -f Modelfile
ollama run gerbil-qwen "How do I parse JSON in Gerbil?"
```

VRAM requirements for running locally:

| Quantization | VRAM | Speed | File Size |
|-------------|------|-------|-----------|
| Q4_K_M | ~5 GB | ~30-40 tok/s | ~4.5 GB |
| Q5_K_M | ~6 GB | ~25-35 tok/s | ~5.5 GB |
| Q8_0 | ~8 GB | ~20-25 tok/s | ~7.5 GB |
| F16 | ~14 GB | ~15-20 tok/s | ~14 GB |

Even an 8GB GPU (RTX 3060/4060) runs Q4 comfortably. CPU-only works too (~5-10 tok/s). Mac with 8GB+ handles Q4.

OpenCode config for local Ollama:
```
Base URL: http://localhost:11434/v1
Model:    gerbil-qwen
```

### Host anywhere else

The downloaded adapter is standard PEFT/HuggingFace format. After merging, deploy to:

- **HuggingFace Inference Endpoints** — push merged model, get API
- **RunPod / Vast.ai** — deploy vLLM container
- **Fireworks AI** — upload model, get API endpoint
- **Any OpenAI-compatible server** — vLLM, llama.cpp, TGI

Qwen is Apache 2.0 licensed — no restrictions on commercial use.

---

## Verifying the Fine-Tune

Test prompts (a base Qwen will struggle with these):

```
1. "How do I iterate over a hash table in Gerbil Scheme?"
   Expected: (import :std/iter), (for ((values k v) (in-hash ht)) ...)

2. "What's the difference between hash-get and hash-ref in Gerbil?"
   Expected: hash-get is 2-arity returns #f; hash-ref throws on missing key

3. "Show me how to define a custom error class in Gerbil"
   Expected: deferror-class from :std/error, mentions defraise/context gotcha

4. "How do I build a static executable in Gerbil?"
   Expected: build.ss with (exe: "main"), GERBIL_LOADPATH, gerbil build

5. "What's wrong with passing u8vector to (pointer void) in FFI?"
   Expected: Gambit passes raw Scheme object header, use scheme-object type
```

Run all 10 test prompts automatically:
```bash
python3 verify_model.py --base-url <url> --model <model> --api-key <key> -v
```

---

## Iterating

To improve the model with more data:

1. Add recipes to the gerbil-mcp cookbook (`gerbil_howto_add`)
2. Regenerate: `python3 convert_training_data.py`
3. Re-upload: `python3 train_together.py upload`
4. Retrain: `python3 train_together.py train`
