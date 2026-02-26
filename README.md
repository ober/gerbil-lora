# Gerbil Scheme LoRA

A fine-tuned Qwen 2.5 7B model that knows Gerbil Scheme. Run locally with Ollama, host on RunPod, or pull from the registry.

## Quick Start (Local)

```bash
ollama pull jaimef/gerbil-qwen
ollama run jaimef/gerbil-qwen "How do I parse JSON in Gerbil Scheme?"
```

## Deployment Options

| Option | Cost | Speed | Setup |
|--------|------|-------|-------|
| **Local Ollama (GPU)** | Free | 30-40 tok/s | `ollama pull jaimef/gerbil-qwen` |
| **Local Ollama (CPU)** | Free | 5-10 tok/s (slow) | Same as above |
| **RunPod Serverless** | $0 idle, ~$0.34/hr active | 30-40 tok/s | `./deploy_runpod.sh` |
| **Together AI Endpoint** | $6.60/hr always-on | Fast | Not recommended |

### Estimated RunPod monthly costs

| Usage | Hours/month | Cost/month |
|-------|-------------|------------|
| Idle (scale-to-zero) | 0 | **$0** |
| Light (1hr/day) | ~30 | **~$10** |
| Moderate (3hr/day) | ~90 | **~$31** |
| Heavy (8hr/day) | ~240 | **~$82** |

## Use with OpenCode

### Option A: Local Ollama

Add to `~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
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

### Option B: RunPod Serverless (recommended if no local GPU)

```json
{
  "$schema": "https://opencode.ai/config.json",
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

Replace `<ENDPOINT_ID>` and `<RUNPOD_API_KEY>` with your values from the RunPod console.

## Deploy to RunPod

Upload the merged model to HuggingFace and deploy on RunPod serverless (scale-to-zero):

```bash
# Prerequisites
pip install together huggingface_hub
export TOGETHER_API_KEY="your-key"
hf auth login   # needs write token

# Deploy (downloads merged model from Together AI, uploads to HuggingFace)
./deploy_runpod.sh jaimef/gerbil-qwen-7b
```

The script downloads the pre-merged model from Together AI (~14GB, merged server-side — no local GPU or 32GB RAM needed), uploads to HuggingFace, then prints RunPod setup instructions.

### RunPod endpoint setup

1. Go to https://www.runpod.io/console/serverless
2. Click **New Endpoint**, search for **vLLM**, click Deploy
3. Set: Model = `jaimef/gerbil-qwen-7b`, GPU = 16GB+, Min Workers = 0, Max Workers = 1
4. Your endpoint URL: `https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1`

## Build from Source

If you want to retrain or customize the model yourself:

### 1. Generate training data

```bash
git clone https://github.com/mighty-gerbils/gerbil.git ~/mine/gerbil
git clone https://github.com/gambit/gambit.git ~/mine/gambit
git clone https://github.com/ober/gerbil-mcp.git ~/mine/gerbil-mcp

python3 convert_training_data.py   # 5,985 entries, ~8.5MB
```

### 2. Train on Together AI (~$3, ~7 minutes)

```bash
pip install together
export TOGETHER_API_KEY="your-key"

python3 train_together.py upload
python3 train_together.py train
python3 train_together.py status
```

### 3. Deploy

**Local (with GPU or slow CPU):**
```bash
./download_and_convert.sh
```

**Hosted (RunPod serverless):**
```bash
hf auth login
./deploy_runpod.sh YOUR_USERNAME/gerbil-qwen-7b
```

### 4. Verify

```bash
python3 verify_model.py \
  --base-url http://localhost:11434/v1 \
  --model gerbil-qwen -v
```

### 5. Push to Ollama registry

```bash
ollama cp gerbil-qwen YOUR_USERNAME/gerbil-qwen
ollama push YOUR_USERNAME/gerbil-qwen
```

## Training Data

**5,985 entries** from 11 sources, post-processed to use idiomatic Gerbil conventions (`def` not `define`).

| Source | Count | Description |
|--------|-------|-------------|
| doc | 2,301 | Official Gerbil reference docs |
| cookbook | 1,783 | Verified working code recipes |
| api | 1,635 | Individual API function docs |
| test | 91 | Test files showing real usage |
| security | 70 | Vulnerability patterns and fixes |
| source | 34 | Tutorial/example .ss files |
| gambit | 31 | Gambit interop examples |
| std-source | 21 | Standard library source excerpts |
| convention | 15 | Gerbil idiom teaching examples |
| tutorial | 3 | Official tutorials |
| errorfix | 1 | Error-to-fix mappings |

### Output formats

| File | Format | Use with |
|------|--------|----------|
| `training_data_together.jsonl` | Together AI messages | Together AI fine-tuning |
| `training_data.jsonl` | ChatML/ShareGPT | LLaMA-Factory, Axolotl |
| `training_data_alpaca.jsonl` | Alpaca JSONL | Unsloth, HuggingFace |

## Scripts

| Script | Purpose |
|--------|---------|
| `convert_training_data.py` | Generate training data from source repos |
| `train_together.py` | Upload, train, and monitor on Together AI |
| `download_and_convert.sh` | Download adapter, convert to GGUF, set up Ollama |
| `deploy_runpod.sh` | Upload merged model to HuggingFace for RunPod deployment |
| `verify_model.py` | Run 10 Gerbil-specific test prompts |
| `train_unsloth.py` | Local GPU training with Unsloth |
| `merge_and_export.py` | Merge adapter + base to GGUF (needs 32GB RAM or GPU) |
| `train_runpod.sh` | One-shot training on rented GPU |
| `Modelfile` | Ollama model definition |

## Iterating

To improve the model with more training data:

1. Add recipes to the gerbil-mcp cookbook
2. `python3 convert_training_data.py`
3. `python3 train_together.py upload`
4. `python3 train_together.py train`
5. `./download_and_convert.sh` (local) or `./deploy_runpod.sh` (hosted)
