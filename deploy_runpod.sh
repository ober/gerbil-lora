#!/bin/bash
# Deploy the fine-tuned Gerbil Qwen model to RunPod Serverless via HuggingFace.
#
# This script:
#   1. Downloads the pre-merged model from Together AI (~14GB, merged server-side)
#   2. Creates a HuggingFace repo and uploads the model
#   3. Prints instructions for creating the RunPod serverless endpoint
#
# Prerequisites:
#   pip install together huggingface_hub
#   export TOGETHER_API_KEY="your-key"
#   hf auth login   # needs write token from https://huggingface.co/settings/tokens
#
# Usage:
#   ./deploy_runpod.sh [HF_REPO]
#
# Example:
#   ./deploy_runpod.sh jaimef/gerbil-qwen-7b

set -euo pipefail
cd "$(dirname "$0")"

HF_REPO="${1:-jaimef/gerbil-qwen-7b}"
JOB_ID="ft-5f979336-8831"
MERGED_DIR="./together-merged"

echo "=== Deploying gerbil-qwen to RunPod via HuggingFace ==="
echo "HuggingFace repo: $HF_REPO"
echo ""

# ── Check prerequisites ──────────────────────────────────────────────
if ! command -v together &>/dev/null; then
    echo "ERROR: 'together' CLI not found. Run: pip install together"
    exit 1
fi

if [ -z "${TOGETHER_API_KEY:-}" ]; then
    echo "ERROR: TOGETHER_API_KEY not set. Run: export TOGETHER_API_KEY=your-key"
    exit 1
fi

python3 -c "from huggingface_hub import HfApi; HfApi().whoami()" 2>/dev/null || {
    echo "ERROR: Not logged into HuggingFace. Run: hf auth login"
    exit 1
}

# ── Step 1: Download pre-merged model from Together AI ───────────────
echo "=== Step 1: Download pre-merged model from Together AI ==="
if [ -d "$MERGED_DIR" ] && [ -f "$MERGED_DIR/config.json" ]; then
    echo "Merged model already downloaded at $MERGED_DIR"
else
    echo "Downloading merged model for job $JOB_ID (~14GB) ..."
    echo "Together AI merges the LoRA adapter with the base model server-side."
    echo "No local GPU or 32GB RAM needed."
    mkdir -p "$MERGED_DIR"
    together fine-tuning download "$JOB_ID" \
        --checkpoint-type merged \
        --output_dir "$MERGED_DIR"

    # Together AI downloads a zstd-compressed tar — extract it
    if [ ! -f "$MERGED_DIR/config.json" ]; then
        echo "Extracting compressed archive ..."
        cd "$MERGED_DIR"
        for f in *; do
            if file "$f" | grep -q "Zstandard"; then
                tar --zstd -xf "$f" && rm "$f"
                break
            fi
        done
        cd ..
    fi

    echo "Downloaded to $MERGED_DIR"
fi

# Verify
if [ ! -f "$MERGED_DIR/config.json" ]; then
    echo "ERROR: config.json not found in $MERGED_DIR"
    ls -la "$MERGED_DIR"/
    exit 1
fi

echo ""
echo "=== Step 2: Upload to HuggingFace ==="
echo "Uploading to https://huggingface.co/$HF_REPO ..."

python3 -c "
from huggingface_hub import HfApi, create_repo
api = HfApi()

# Create repo if it doesn't exist
try:
    create_repo('$HF_REPO', repo_type='model', exist_ok=True)
    print('Repo ready: $HF_REPO')
except Exception as e:
    print(f'Repo creation: {e}')

# Upload the merged model
api.upload_folder(
    folder_path='$MERGED_DIR',
    repo_id='$HF_REPO',
    commit_message='Upload Gerbil Qwen 7B - fine-tuned for Gerbil Scheme',
)
print('Upload complete!')
"

echo ""
echo "=== Step 3: Create RunPod Serverless Endpoint ==="
echo ""
echo "Model uploaded to: https://huggingface.co/$HF_REPO"
echo ""
echo "Now create a RunPod serverless endpoint:"
echo ""
echo "  1. Go to https://www.runpod.io/console/serverless"
echo "  2. Click 'New Endpoint'"
echo "  3. Search for 'vLLM' in RunPod Hub and click Deploy"
echo "  4. Set these options:"
echo "     - Model: $HF_REPO"
echo "     - GPU: 16GB+ (RTX 4000 SFF Ada = cheapest for 7B)"
echo "     - Min Workers: 0  (scale to zero when idle)"
echo "     - Max Workers: 1  (cap spend)"
echo "     - Idle Timeout: 60 seconds"
echo "  5. Click Deploy"
echo ""
echo "  Your endpoint URL will be:"
echo "    https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1"
echo ""
echo "  Get your RunPod API key from:"
echo "    https://www.runpod.io/console/user/settings"
echo ""
echo "=== Step 4: Configure OpenCode ==="
echo ""
echo "Add to ~/.config/opencode/opencode.json:"
echo ""
cat <<'OPENCODE'
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
OPENCODE
echo ""
echo "Replace <ENDPOINT_ID> and <RUNPOD_API_KEY> with your values."
echo ""
echo "=== Step 5: Test it ==="
echo ""
echo "  python3 verify_model.py \\"
echo "    --base-url https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1 \\"
echo "    --model $HF_REPO \\"
echo "    --api-key <RUNPOD_API_KEY> -v"
echo ""
echo "=== Estimated costs ==="
echo ""
echo "  Idle:            \$0/month (scale to zero)"
echo "  Light (1hr/day): ~\$10/month"
echo "  Heavy (8hr/day): ~\$82/month"
echo "  Always on 24/7:  ~\$245/month"
echo ""
echo "Done!"
