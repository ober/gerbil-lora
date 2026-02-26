#!/bin/bash
# Deploy the fine-tuned Gerbil Qwen model to RunPod Serverless via HuggingFace.
#
# This script:
#   1. Downloads the pre-merged model from Together AI (~14GB, merged server-side)
#   2. Creates a HuggingFace repo and uploads the model
#   3. Creates a RunPod serverless vLLM endpoint via GraphQL API
#
# Prerequisites:
#   pip install together huggingface_hub
#   export TOGETHER_API_KEY="your-key"
#   export RUNPOD_API_KEY="your-key"    # from https://www.runpod.io/console/user/settings
#   hf auth login                        # needs write token from https://huggingface.co/settings/tokens
#
# Usage:
#   ./deploy_runpod.sh [HF_REPO]
#
# Example:
#   ./deploy_runpod.sh jaimef21/gerbil-qwen-7b

set -euo pipefail
cd "$(dirname "$0")"

HF_REPO="${1:-jaimef21/gerbil-qwen-7b}"
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

if [ -z "${RUNPOD_API_KEY:-}" ]; then
    echo "ERROR: RUNPOD_API_KEY not set."
    echo "Get your key from: https://www.runpod.io/console/user/settings"
    echo "Then run: export RUNPOD_API_KEY=your-key"
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
echo "Model uploaded to: https://huggingface.co/$HF_REPO"

echo ""
echo "=== Step 3: Create RunPod Serverless Endpoint ==="

# First, create a serverless template with vLLM worker + our model
echo "Creating RunPod serverless template ..."
TEMPLATE_RESULT=$(curl -s --request POST \
    --header 'content-type: application/json' \
    --url "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" \
    --data "{\"query\": \"mutation { saveTemplate(input: { name: \\\"gerbil-qwen-vllm\\\", imageName: \\\"runpod/worker-v1-vllm:stable-cuda12.1.0\\\", isServerless: true, containerDiskInGb: 20, dockerArgs: \\\"\\\", volumeInGb: 0, env: [ { key: \\\"MODEL_NAME\\\", value: \\\"$HF_REPO\\\" }, { key: \\\"MAX_MODEL_LEN\\\", value: \\\"8192\\\" }, { key: \\\"GPU_MEMORY_UTILIZATION\\\", value: \\\"0.90\\\" } ] }) { id name imageName } }\"}")

TEMPLATE_ID=$(echo "$TEMPLATE_RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data']['saveTemplate']['id'])" 2>/dev/null || true)

if [ -z "$TEMPLATE_ID" ]; then
    echo "ERROR: Failed to create template."
    echo "Response: $TEMPLATE_RESULT"
    echo ""
    echo "You may need to create the endpoint manually via the RunPod UI."
    echo "See instructions below."
    MANUAL=1
else
    echo "Template created: $TEMPLATE_ID"

    # Now create the serverless endpoint using that template
    echo "Creating serverless endpoint ..."
    ENDPOINT_RESULT=$(curl -s --request POST \
        --header 'content-type: application/json' \
        --url "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" \
        --data "{\"query\": \"mutation { saveEndpoint(input: { name: \\\"gerbil-qwen\\\", templateId: \\\"$TEMPLATE_ID\\\", gpuIds: \\\"AMPERE_24\\\", workersMin: 0, workersMax: 1, idleTimeout: 60, scalerType: \\\"QUEUE_DELAY\\\", scalerValue: 4 }) { id name gpuIds templateId workersMin workersMax idleTimeout } }\"}")

    ENDPOINT_ID=$(echo "$ENDPOINT_RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data']['saveEndpoint']['id'])" 2>/dev/null || true)

    if [ -z "$ENDPOINT_ID" ]; then
        echo "ERROR: Failed to create endpoint."
        echo "Response: $ENDPOINT_RESULT"
        MANUAL=1
    else
        MANUAL=0
        echo ""
        echo "=== Endpoint Created! ==="
        echo ""
        echo "  Endpoint ID:  $ENDPOINT_ID"
        echo "  API URL:      https://api.runpod.ai/v2/$ENDPOINT_ID/openai/v1"
        echo "  Console:      https://www.runpod.io/console/serverless/$ENDPOINT_ID"
        echo "  GPU:          AMPERE_24 (~\$0.39/hr)"
        echo "  Min Workers:  0 (scale to zero)"
        echo "  Max Workers:  1"
        echo "  Idle Timeout: 60s"
    fi
fi

if [ "${MANUAL:-0}" = "1" ]; then
    echo ""
    echo "=== Manual RunPod Setup ==="
    echo ""
    echo "  1. Go to https://www.runpod.io/console/serverless"
    echo "  2. Click 'New Endpoint'"
    echo "  3. Search for 'vLLM' and click Deploy"
    echo "  4. Set: Model = $HF_REPO"
    echo "     GPU: 24GB+ (AMPERE_24 recommended for 7B)"
    echo "     Min Workers: 0, Max Workers: 1"
    echo "     Idle Timeout: 60s"
    echo "  5. Click Deploy"
    echo ""
    echo "  Your endpoint URL will be:"
    echo "    https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1"
fi

echo ""
echo "=== Configure OpenCode ==="
echo ""
echo "Add to ~/.config/opencode/opencode.json:"
echo ""

if [ "${MANUAL:-0}" = "0" ] && [ -n "${ENDPOINT_ID:-}" ]; then
    cat <<OPENCODE
{
  "\$schema": "https://opencode.ai/config.json",
  "provider": {
    "runpod": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "RunPod (serverless)",
      "options": {
        "baseURL": "https://api.runpod.ai/v2/$ENDPOINT_ID/openai/v1",
        "apiKey": "$RUNPOD_API_KEY"
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
else
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
fi

echo ""
echo "=== Test it ==="
echo ""
if [ "${MANUAL:-0}" = "0" ] && [ -n "${ENDPOINT_ID:-}" ]; then
    echo "  python3 verify_model.py \\"
    echo "    --base-url https://api.runpod.ai/v2/$ENDPOINT_ID/openai/v1 \\"
    echo "    --model $HF_REPO \\"
    echo "    --api-key $RUNPOD_API_KEY -v"
else
    echo "  python3 verify_model.py \\"
    echo "    --base-url https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1 \\"
    echo "    --model $HF_REPO \\"
    echo "    --api-key <RUNPOD_API_KEY> -v"
fi

echo ""
echo "=== Estimated costs ==="
echo ""
echo "  Idle:            \$0/month (scale to zero)"
echo "  Light (1hr/day): ~\$10/month"
echo "  Moderate (3hr/day): ~\$31/month"
echo "  Heavy (8hr/day): ~\$82/month"
echo ""
echo "  First request after idle: ~10-30s cold start"
echo ""
echo "Done!"
