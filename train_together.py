#!/usr/bin/env python3
"""
Train a Gerbil Scheme LoRA on Together AI.

Prerequisites:
  pip install together
  export TOGETHER_API_KEY="your-key-here"

Usage:
  python3 train_together.py upload     # Upload training data
  python3 train_together.py train      # Start fine-tuning (after upload)
  python3 train_together.py status     # Check training status
  python3 train_together.py test       # Test the fine-tuned model
"""

import sys
import os
import json
import time

try:
    from together import Together
except ImportError:
    print("Install the Together SDK: pip install together")
    sys.exit(1)

# ── Config ──────────────────────────────────────────────────────────
BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
TRAINING_FILE = os.path.join(os.path.dirname(__file__), "training_data_together.jsonl")
STATE_FILE = os.path.join(os.path.dirname(__file__), ".together_state.json")

LORA_R = 16
LORA_ALPHA = 32
EPOCHS = 3
LEARNING_RATE = 1e-5
BATCH_SIZE = 4


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def cmd_upload():
    """Upload training data to Together AI."""
    client = Together()

    print(f"Uploading {TRAINING_FILE} ...")
    response = client.files.upload(file=TRAINING_FILE, purpose="fine-tune")
    file_id = response.id

    state = load_state()
    state["file_id"] = file_id
    save_state(state)

    print(f"Uploaded! File ID: {file_id}")
    print(f"Saved to {STATE_FILE}")
    print(f"\nNext: python3 {sys.argv[0]} train")


def cmd_train():
    """Start a fine-tuning job."""
    client = Together()
    state = load_state()

    file_id = state.get("file_id")
    if not file_id:
        print("No file_id found. Run 'upload' first.")
        sys.exit(1)

    print(f"Starting fine-tune on {BASE_MODEL} ...")
    print(f"  File: {file_id}")
    print(f"  Epochs: {EPOCHS}, LR: {LEARNING_RATE}, Batch: {BATCH_SIZE}")
    print(f"  LoRA r={LORA_R}, alpha={LORA_ALPHA}")

    response = client.fine_tuning.create(
        model=BASE_MODEL,
        training_file=file_id,
        n_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        lora=True,
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
    )

    job_id = response.id
    state["job_id"] = job_id
    save_state(state)

    print(f"\nJob started! ID: {job_id}")
    print(f"Saved to {STATE_FILE}")
    print(f"\nMonitor: python3 {sys.argv[0]} status")


def cmd_status():
    """Check fine-tuning job status."""
    client = Together()
    state = load_state()

    job_id = state.get("job_id")
    if not job_id:
        print("No job_id found. Run 'train' first.")
        sys.exit(1)

    response = client.fine_tuning.retrieve(id=job_id)

    print(f"Job: {job_id}")
    print(f"Status: {response.status}")

    if hasattr(response, "output_name") and response.output_name:
        state["model_name"] = response.output_name
        save_state(state)
        print(f"Model: {response.output_name}")
        print(f"\nReady! Test: python3 {sys.argv[0]} test")

    if hasattr(response, "events") and response.events:
        print("\nRecent events:")
        for event in response.events[-5:]:
            print(f"  {event}")


def cmd_test():
    """Test the fine-tuned model."""
    client = Together()
    state = load_state()

    model_name = state.get("model_name")
    if not model_name:
        print("No model_name found. Check 'status' — training may still be running.")
        sys.exit(1)

    test_prompts = [
        "How do I iterate over a hash table in Gerbil Scheme?",
        "What's the difference between hash-get and hash-ref in Gerbil?",
        "Show me how to parse JSON in Gerbil Scheme.",
        "How do I define a custom error class in Gerbil?",
        "What's wrong with passing u8vector to a (pointer void) FFI parameter in Gerbil?",
    ]

    for prompt in test_prompts:
        print(f"\n{'='*60}")
        print(f"Q: {prompt}")
        print(f"{'='*60}")

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert in Gerbil Scheme, a dialect of Scheme "
                        "built on Gambit. You provide accurate, idiomatic Gerbil "
                        "code with correct imports, function names, and arities."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=1024,
            temperature=0.2,
        )

        print(response.choices[0].message.content)


def cmd_opencode_config():
    """Print OpenCode configuration for the fine-tuned model."""
    state = load_state()
    model_name = state.get("model_name")
    if not model_name:
        print("No model_name found yet. Train first.")
        sys.exit(1)

    print("Add to your OpenCode config:\n")
    config = {
        "provider": "openai-compatible",
        "base_url": "https://api.together.xyz/v1",
        "api_key": "${TOGETHER_API_KEY}",
        "model": model_name,
    }
    print(json.dumps(config, indent=2))


COMMANDS = {
    "upload": cmd_upload,
    "train": cmd_train,
    "status": cmd_status,
    "test": cmd_test,
    "config": cmd_opencode_config,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(f"Usage: python3 {sys.argv[0]} <command>")
        print(f"Commands: {', '.join(COMMANDS.keys())}")
        print()
        print("Workflow:")
        print("  1. upload  — Upload training_data.jsonl to Together AI")
        print("  2. train   — Start LoRA fine-tuning job")
        print("  3. status  — Check if training is done")
        print("  4. test    — Run verification prompts")
        print("  5. config  — Print OpenCode configuration")
        sys.exit(1)

    COMMANDS[sys.argv[1]]()
