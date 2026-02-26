# Gerbil Scheme LoRA Training — Together AI

## Status

- [x] Generate training data (5,970 entries from cookbooks, docs, source code)
- [x] Upload training data to Together AI
- [x] Start fine-tuning job (ft-5f979336-8831)
- [x] Training completed (~7 min, 63 steps, 3 epochs, 2.1M tokens)
- [ ] Download adapter and convert to GGUF
- [ ] Deploy locally with Ollama
- [ ] Verify model with test prompts
- [ ] Connect to OpenCode

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

## Step 3: Train (DONE)

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

## Step 4: Monitor (DONE)

```bash
python3 train_together.py status
# Or: together fine-tuning retrieve ft-5f979336-8831
```

---

## Step 5: Download and Run Locally with Ollama (recommended)

Together AI's dedicated endpoints cost $6.60/hr. Running locally is free.

**No merge required!** Ollama supports LoRA adapters natively. We download the
small adapter (~50-200MB), convert it to GGUF, and Ollama applies it on the
fly over the base model. No GPU or 32GB RAM needed for the conversion step.

### One-command setup

```bash
./download_and_convert.sh
```

This runs all the steps below automatically.

### 5a. Download the LoRA adapter from Together AI

```bash
together fine-tuning download ft-5f979336-8831 \
  --checkpoint-type adapter \
  --output_dir ./together-adapter
```

This gives you the LoRA adapter weights (~50-200MB) — the diff that makes
the base Qwen model know Gerbil Scheme.

### 5b. Install lightweight converter dependencies

No Unsloth, no GPU, no 32GB RAM needed. Just some Python packages
and llama.cpp's converter script:

```bash
python3 -m pip install --break-system-packages gguf transformers safetensors
python3 -m pip install --break-system-packages torch --index-url https://download.pytorch.org/whl/cpu
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/ggml-org/llama.cpp.git
cd llama.cpp && git sparse-checkout set --skip-checks convert_lora_to_gguf.py gguf-py scripts && cd ..
```

### 5c. Convert adapter to GGUF

```bash
python3 llama.cpp/convert_lora_to_gguf.py \
  --outfile ./gerbil-lora-adapter.gguf \
  ./together-adapter
```

This only processes the small adapter weights — no need to download or
load the full 14GB base model.

### 5d. Install Ollama (if not already installed)

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 5e. Pull base model and create gerbil-qwen

```bash
# Pull the base model (one-time ~4.7GB download)
ollama pull qwen2.5:7b-instruct

# Create the model with the LoRA adapter applied
ollama create gerbil-qwen -f Modelfile
```

The `Modelfile` uses `FROM qwen2.5:7b-instruct` and applies the adapter
via the `ADAPTER` instruction. No merge needed — Ollama handles it at runtime.

### 5f. Test it

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

### 5g. Configure OpenCode

Point OpenCode at your local Ollama:

```
Base URL: http://localhost:11434/v1
Model:    gerbil-qwen
```

Free forever. No API costs, no idle charges, fully private.

### Hardware requirements to run

| Hardware | Works? | Speed |
|----------|--------|-------|
| RTX 3060/4060 (8GB) | Yes | ~30-40 tok/s |
| RTX 3090/4090 (24GB) | Yes | ~40+ tok/s |
| Mac M1/M2 8GB | Yes | ~20-30 tok/s |
| Mac M1/M2 16GB+ | Yes | ~25-35 tok/s |
| CPU only, 16GB+ RAM | Yes | ~5-10 tok/s |

---

## Alternative: Together AI Hosted (expensive)

Fine-tuned Qwen 7B is NOT available for serverless LoRA on Together AI
(only Qwen 72B and Llama models are supported). The only hosted option
is a dedicated endpoint at **$6.60/hr** (2x H100), which is impractical.

If you want serverless pay-per-token inference, you'd need to retrain
on a supported base model like `Meta-Llama/Meta-Llama-3.1-8B-Instruct-Reference`.

```bash
# Retrain on Llama 3.1 8B (supported for serverless LoRA):
# 1. Update BASE_MODEL in train_together.py
# 2. python3 train_together.py upload
# 3. python3 train_together.py train
# Then use serverless inference at per-token prices — no endpoint needed.
```

## Alternative: Cloud Merge (if you need a merged GGUF)

If for some reason you need a single merged GGUF file (not separate
base + adapter), you can:

1. **HuggingFace GGUF-my-LoRA Space** — upload adapter, it merges in the cloud
2. **Google Colab** — free T4 GPU with ~12.7GB RAM, enough for a 7B merge
3. **Together AI merged download** — `together fine-tuning download ft-5f979336-8831 --checkpoint-type merged`

## Alternative: Host Anywhere Else

The downloaded adapter is standard PEFT/HuggingFace format. Deploy to any platform:

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
5. Re-download and convert: `./download_and_convert.sh`
