# Gerbil Scheme LoRA

A fine-tuned Qwen 2.5 7B model that knows Gerbil Scheme. Run locally with Ollama or pull from the registry.

## Quick Start

```bash
ollama pull jaimef/gerbil-qwen
ollama run jaimef/gerbil-qwen "How do I parse JSON in Gerbil Scheme?"
```

## Use with OpenCode

Add the Ollama provider and model to `~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "ollama": {
      "type": "@ai-sdk/openai-compatible",
      "baseURL": "http://localhost:11434/v1"
    }
  },
  "model": {
    "gerbil-qwen": {
      "provider": "ollama",
      "model": "jaimef/gerbil-qwen",
      "contextWindow": 32768
    }
  }
}
```

Then select `gerbil-qwen` as your model in OpenCode.

## Build from Source

If you want to retrain or customize the model yourself:

### 1. Generate training data

Requires the Gerbil source repos:

```bash
# Clone dependencies
git clone https://github.com/mighty-gerbils/gerbil.git ~/mine/gerbil
git clone https://github.com/gambit/gambit.git ~/mine/gambit
git clone https://github.com/ober/gerbil-mcp.git ~/mine/gerbil-mcp

# Generate training data (5,985 entries, ~8.5MB)
python3 convert_training_data.py
```

### 2. Train on Together AI (~$3, ~7 minutes)

```bash
pip install together
export TOGETHER_API_KEY="your-key"

python3 train_together.py upload   # upload training data
python3 train_together.py train    # start LoRA fine-tuning
python3 train_together.py status   # check progress
```

### 3. Download adapter and convert to GGUF

No GPU or 32GB RAM needed — Ollama applies the LoRA adapter at runtime.

```bash
./download_and_convert.sh
```

This script:
1. Downloads the LoRA adapter from Together AI (~150MB)
2. Clones llama.cpp's converter (sparse checkout)
3. Converts the adapter to GGUF format
4. Pulls `qwen2.5:7b-instruct` base model in Ollama
5. Creates the `gerbil-qwen` model with the adapter applied

### 4. Verify

```bash
ollama run gerbil-qwen "How do I iterate over a hash table in Gerbil?"
```

Or run the full test suite:

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
| `train_unsloth.py` | Local GPU training with Unsloth |
| `merge_and_export.py` | Merge adapter + base model to GGUF (needs 32GB RAM or GPU) |
| `download_and_convert.sh` | Download adapter, convert to GGUF, set up Ollama (no GPU needed) |
| `verify_model.py` | Run 10 Gerbil-specific test prompts |
| `train_runpod.sh` | One-shot training on rented GPU |
| `Modelfile` | Ollama model definition |

## Iterating

To improve the model with more training data:

1. Add recipes to the gerbil-mcp cookbook
2. `python3 convert_training_data.py`
3. `python3 train_together.py upload`
4. `python3 train_together.py train`
5. `./download_and_convert.sh`
