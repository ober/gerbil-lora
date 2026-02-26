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

## Step 5: Create a Dedicated Endpoint

Fine-tuned models on Together AI are NOT serverless — they need a dedicated endpoint.

Create one via the UI or CLI:
```bash
# Via UI (check pricing first):
# https://api.together.ai/models/jaimef_2515/Qwen2.5-7B-Instruct-74900ead

# Via CLI:
together endpoints create \
  --model jaimef_2515/Qwen2.5-7B-Instruct-74900ead \
  --display-name gerbil-qwen \
  --hardware 2x_nvidia_h100_80gb_sxm
```

**Warning:** The only available hardware is 2x H100 at **$0.11/min ($6.60/hr)**.
This is expensive for casual use. Strongly consider downloading the adapter
and running locally with Ollama instead (see Step 8).

Stop the endpoint immediately when not testing:
```bash
together endpoints delete <endpoint-id>
```

## Step 6: Test on Together AI

```bash
python3 train_together.py test
# Or:
python3 verify_model.py \
  --base-url https://api.together.xyz/v1 \
  --model jaimef_2515/Qwen2.5-7B-Instruct-74900ead \
  --api-key $TOGETHER_API_KEY -v
```

## Step 7: Use with OpenCode (hosted on Together AI)

Point OpenCode at Together AI's API:

```
Base URL: https://api.together.xyz/v1
API Key:  $TOGETHER_API_KEY
Model:    jaimef_2515/Qwen2.5-7B-Instruct-74900ead
```

Costs per hour while endpoint is running. Stop when not in use.

---

## Step 6: Download and Run Locally with Ollama (recommended)

Together AI's dedicated endpoints cost $6.60/hr. Running locally is free.

### 6a. Download the LoRA adapter from Together AI

```bash
cd ~/mine/gerbil-lora
together fine-tuning download ft-5f979336-8831 --output ./together-adapter
```

This gives you the LoRA adapter weights (~50-200MB) — the diff that makes
the base Qwen model know Gerbil Scheme.

### 6b. Install merge dependencies

You need Python packages to merge the adapter into the base model and
convert to GGUF format. This requires a GPU with 16GB+ VRAM, OR ~32GB RAM
for CPU-only merge.

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps xformers trl peft accelerate bitsandbytes triton
```

### 6c. Merge adapter + base model → GGUF

This downloads the base model (~14GB), applies the LoRA adapter on top,
and quantizes to a single GGUF file that Ollama can run:

```bash
python3 merge_and_export.py \
  --adapter ./together-adapter \
  --base Qwen/Qwen2.5-7B-Instruct \
  --quant q4_k_m
```

The `--base` flag downloads Qwen/Qwen2.5-7B-Instruct from HuggingFace
automatically (one-time ~14GB download, cached for future use).

Quantization options:

| Quantization | GGUF Size | VRAM to Run | Speed | Quality |
|-------------|-----------|-------------|-------|---------|
| `q4_k_m` | ~4.5 GB | ~5 GB | ~30-40 tok/s | Good enough |
| `q5_k_m` | ~5.5 GB | ~6 GB | ~25-35 tok/s | Better |
| `q8_0` | ~7.5 GB | ~8 GB | ~20-25 tok/s | Near-lossless |
| `f16` | ~14 GB | ~14 GB | ~15-20 tok/s | Perfect |

### 6d. Install Ollama (if not already installed)

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 6e. Import the GGUF into Ollama

```bash
cd ~/mine/gerbil-lora
ollama create gerbil-qwen -f Modelfile
```

The `Modelfile` in this repo points to `./gerbil-qwen-gguf/unsloth.Q4_K_M.gguf`
and sets up the Qwen ChatML template and system prompt.

### 6f. Test it

```bash
ollama run gerbil-qwen "How do I parse JSON in Gerbil Scheme?"
```

Or run the full verification suite:
```bash
python3 verify_model.py \
  --base-url http://localhost:11434/v1 \
  --model gerbil-qwen \
  -v
```

### 6g. Configure OpenCode

Point OpenCode at your local Ollama:

```
Base URL: http://localhost:11434/v1
Model:    gerbil-qwen
```

Free forever. No API costs, no idle charges, fully private.

### Hardware requirements to run

| Hardware | Works? | Speed |
|----------|--------|-------|
| RTX 3060/4060 (8GB) | Yes, Q4 | ~30-40 tok/s |
| RTX 3090/4090 (24GB) | Yes, any quant | ~40+ tok/s |
| Mac M1/M2 8GB | Yes, Q4 | ~20-30 tok/s |
| Mac M1/M2 16GB+ | Yes, Q8 | ~25-35 tok/s |
| CPU only, 16GB+ RAM | Yes, Q4 | ~5-10 tok/s |

---

## Alternative: Host Anywhere Else

The downloaded adapter is standard PEFT/HuggingFace format. After merging,
deploy to any platform:

- **HuggingFace Inference Endpoints** — push merged model, get API
- **RunPod / Vast.ai** — deploy vLLM container (~$0.15-0.30/hr)
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
python3 verify_model.py --base-url <url> --model <model> -v
```

---

## Iterating

To improve the model with more data:

1. Add recipes to the gerbil-mcp cookbook (`gerbil_howto_add`)
2. Regenerate: `python3 convert_training_data.py`
3. Re-upload: `python3 train_together.py upload`
4. Retrain: `python3 train_together.py train`
5. Re-download and merge: steps 6a-6e above
