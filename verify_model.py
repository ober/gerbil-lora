#!/usr/bin/env python3
"""
Verify a fine-tuned Gerbil Scheme model by running test prompts.

Works with any OpenAI-compatible API (Together AI, Ollama, OpenRouter, etc.)

Usage:
  # Test against Ollama
  python3 verify_model.py --base-url http://localhost:11434/v1 --model gerbil-qwen

  # Test against Together AI
  python3 verify_model.py --base-url https://api.together.xyz/v1 --model your-org/gerbil-qwen --api-key $TOGETHER_API_KEY

  # Test against OpenRouter
  python3 verify_model.py --base-url https://openrouter.ai/api/v1 --model your-model --api-key $OPENROUTER_API_KEY
"""

import argparse
import json
import sys

try:
    from openai import OpenAI
except ImportError:
    print("Install the OpenAI SDK: pip install openai")
    sys.exit(1)


SYSTEM_PROMPT = (
    "You are an expert in Gerbil Scheme, a dialect of Scheme built on Gambit. "
    "You provide accurate, idiomatic Gerbil code with correct imports, function "
    "names, and arities."
)

TEST_CASES = [
    {
        "prompt": "How do I iterate over a hash table in Gerbil Scheme?",
        "must_contain": ["import", "in-hash", "for"],
        "must_not_contain": ["hash-table-for-each"],
    },
    {
        "prompt": "What's the difference between hash-get and hash-ref in Gerbil?",
        "must_contain": ["hash-get", "hash-ref", "#f"],
        "must_not_contain": [],
    },
    {
        "prompt": "Show me how to parse JSON in Gerbil Scheme.",
        "must_contain": [":std/text/json", "read-json"],
        "must_not_contain": ["require", "json-parse"],
    },
    {
        "prompt": "How do I define a custom error class in Gerbil?",
        "must_contain": ["deferror-class", ":std/error"],
        "must_not_contain": ["define-condition-type"],
    },
    {
        "prompt": "What's wrong with passing u8vector to a (pointer void) FFI parameter in Gerbil?",
        "must_contain": ["scheme-object"],
        "must_not_contain": [],
    },
    {
        "prompt": "How do I spawn an actor in Gerbil Scheme?",
        "must_contain": ["spawn"],
        "must_not_contain": ["make-actor", "create-actor"],
    },
    {
        "prompt": "Show me pattern matching with struct destructuring in Gerbil.",
        "must_contain": ["match", "defstruct"],
        "must_not_contain": [],
    },
    {
        "prompt": "How do I write unit tests in Gerbil Scheme?",
        "must_contain": [":std/test", "test-suite", "check"],
        "must_not_contain": [],
    },
    {
        "prompt": "What imports do I need to use channels in Gerbil?",
        "must_contain": [":std/misc/channel"],
        "must_not_contain": [],
    },
    {
        "prompt": "How do I use the for/collect macro in Gerbil?",
        "must_contain": [":std/iter", "for/collect"],
        "must_not_contain": [],
    },
]


def run_test(client, model, test_case):
    """Run a single test and return pass/fail with details."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": test_case["prompt"]},
        ],
        max_tokens=1024,
        temperature=0.2,
    )

    answer = response.choices[0].message.content
    answer_lower = answer.lower()

    passed = True
    issues = []

    for term in test_case["must_contain"]:
        if term.lower() not in answer_lower:
            passed = False
            issues.append(f"missing '{term}'")

    for term in test_case["must_not_contain"]:
        if term.lower() in answer_lower:
            passed = False
            issues.append(f"contains wrong term '{term}'")

    return passed, answer, issues


def main():
    parser = argparse.ArgumentParser(description="Verify Gerbil LoRA model")
    parser.add_argument("--base-url", required=True, help="API base URL")
    parser.add_argument("--model", required=True, help="Model name/ID")
    parser.add_argument("--api-key", default="not-needed", help="API key")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show full responses")
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    total = len(TEST_CASES)
    passed = 0

    for i, test in enumerate(TEST_CASES, 1):
        print(f"\n[{i}/{total}] {test['prompt'][:70]}...")

        try:
            ok, answer, issues = run_test(client, args.model, test)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        if ok:
            passed += 1
            print(f"  PASS")
        else:
            print(f"  FAIL: {', '.join(issues)}")

        if args.verbose:
            print(f"  Response: {answer[:200]}...")

    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} passed ({100*passed//total}%)")

    if passed >= total * 0.8:
        print("Model looks good for Gerbil Scheme!")
    elif passed >= total * 0.5:
        print("Partial success — consider more training epochs or data.")
    else:
        print("Model needs more training. Check data format and hyperparameters.")


if __name__ == "__main__":
    main()
