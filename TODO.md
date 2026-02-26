# Gerbil Scheme LoRA Training — Together AI

## Status

- [x] Generate training data (5,985 entries from cookbooks, docs, source code)
- [x] Upload training data to Together AI
- [x] Start fine-tuning job (ft-5f979336-8831)
- [x] Training completed (~7 min, 63 steps, 3 epochs, 2.1M tokens)
- [x] Download adapter and convert to GGUF
- [x] Deploy locally with Ollama
- [x] Push to Ollama registry (`ollama push jaimef/gerbil-qwen`)
- [ ] Deploy to RunPod serverless (hosted, scale-to-zero)
- [ ] Verify model with test prompts
- [ ] Connect to OpenCode

---

## Training Data

Generated **5,985 training entries** in `~/mine/gerbil-lora/`:

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
```

Job ID: `ft-5f979336-8831`
Model: `jaimef_2515/Qwen2.5-7B-Instruct-74900ead`

Training completed in ~7 minutes: 63 steps, 3 epochs, 2.1M tokens trained.

---

## Step 4: Deploy — Choose Your Option

### Option A: Local Ollama (free, needs GPU for good speed)

**No merge required!** Ollama supports LoRA adapters natively.

```bash
./download_and_convert.sh
```

Or pull from the registry:
```bash
ollama pull jaimef/gerbil-qwen
```

Configure OpenCode (`~/.config/opencode/opencode.json`):
```json
{
  "provider": {
    "ollama": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Ollama (local)",
      "options": {
        "baseURL": "http://localhost:11434/v1"
      },
      "models": {
        "gerbil-qwen": {
          "name": "Gerbil Qwen"
        }
      }
    }
  }
}
```

| Hardware | Speed |
|----------|-------|
| GPU (8GB+ VRAM) | ~30-40 tok/s |
| Mac M-series | ~20-35 tok/s |
| CPU only | ~5-10 tok/s (painful for OpenCode) |

### Option B: RunPod Serverless (recommended if no local GPU)

Scale-to-zero — **$0 when idle**, ~$0.34/hr only while generating.

```bash
# Prerequisites
pip install together huggingface_hub
export TOGETHER_API_KEY="your-key"
hf auth login   # needs write token from https://huggingface.co/settings/tokens

# Download merged model from Together AI and upload to HuggingFace
./deploy_runpod.sh jaimef/gerbil-qwen-7b
```

This downloads the pre-merged model from Together AI (~14GB, merged server-side — no local GPU needed), uploads it to HuggingFace, then prints RunPod setup instructions.

#### Create RunPod endpoint

1. Go to https://www.runpod.io/console/serverless
2. Click **New Endpoint**
3. Search for **vLLM** in RunPod Hub, click **Deploy**
4. Configure:
   - **Model**: `jaimef/gerbil-qwen-7b`
   - **GPU**: 16GB+ (RTX 4000 SFF Ada is cheapest for 7B)
   - **Min Workers**: 0 (scale to zero when idle)
   - **Max Workers**: 1 (cap spend at one GPU)
   - **Idle Timeout**: 60 seconds
5. Click **Deploy**

Your endpoint URL: `https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1`

Get your API key: https://www.runpod.io/console/user/settings

#### Configure OpenCode

Add to `~/.config/opencode/opencode.json`:

```json
{
  "provider": {
    "runpod": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "RunPod (serverless)",
      "options": {
        "baseURL": "https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1",
        "apiKey": "<RUNPOD_API_KEY>"
      },
      "models": {
        "gerbil-qwen": {
          "name": "Gerbil Qwen 7B"
        }
      }
    }
  }
}
```

#### Estimated monthly costs

| Usage | Hours/month | Cost/month |
|-------|-------------|------------|
| Idle (scale-to-zero) | 0 | **$0** |
| Light (1hr/day) | ~30 | **~$10** |
| Moderate (3hr/day) | ~90 | **~$31** |
| Heavy (8hr/day) | ~240 | **~$82** |
| Always on 24/7 | 720 | **~$245** |

### Option C: Together AI Dedicated Endpoint (expensive, not recommended)

Fine-tuned Qwen 7B is NOT available for Together AI serverless LoRA
(only Qwen 72B and Llama models are supported). Dedicated endpoints
cost **$6.60/hr** (2x H100). Not practical for a 7B model.

---

## Step 5: Verify

```bash
# Local Ollama
python3 verify_model.py \
  --base-url http://localhost:11434/v1 \
  --model gerbil-qwen -v

# RunPod
python3 verify_model.py \
  --base-url https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1 \
  --model jaimef/gerbil-qwen-7b \
  --api-key <RUNPOD_API_KEY> -v
```

Test prompts (a base Qwen will struggle with these):

```
1. "How do I iterate over a hash table in Gerbil Scheme?"
   Expected: (import :std/iter), (for ((values k v) (in-hash ht)) ...)

2. "What's the difference between hash-get and hash-ref in Gerbil?"
   Expected: hash-get returns #f on missing; hash-ref throws

3. "Show me how to define a custom error class in Gerbil"
   Expected: deferror-class from :std/error

4. "How do I build a static executable in Gerbil?"
   Expected: build.ss with (exe: "main"), gerbil build

5. "What's wrong with passing u8vector to (pointer void) in FFI?"
   Expected: Gambit passes raw Scheme object header, use scheme-object
```

---

## Iterating

To improve the model with more data:

1. Add recipes to the gerbil-mcp cookbook (`gerbil_howto_add`)
2. Regenerate: `python3 convert_training_data.py`
3. Re-upload: `python3 train_together.py upload`
4. Retrain: `python3 train_together.py train`
5. Re-deploy: `./download_and_convert.sh` (local) or `./deploy_runpod.sh` (hosted)
