# Gerbil Scheme LoRA Training Data

Training dataset for fine-tuning open-source LLMs to understand Gerbil Scheme, a dialect of Scheme built on Gambit.

## Dataset Files

| File | Format | Size | Description |
|------|--------|------|-------------|
| `training_data.jsonl` | ChatML/ShareGPT | ~8.9MB | Primary format for most LoRA tools (LLaMA-Factory, Axolotl, etc.) |
| `training_data_alpaca.jsonl` | Alpaca JSONL | ~6.2MB | Alpaca format (instruction/input/output) per line |
| `training_data_alpaca.json` | Alpaca JSON | ~6.3MB | Same as above, as a single JSON array |

## Statistics

- **Total entries:** 5,970
- **Median entry length:** 854 characters
- **Mean entry length:** 1,338 characters
- **Entries > ~4096 tokens:** 5 (0.08%)

### By source category

| Source | Count | Description |
|--------|-------|-------------|
| doc | 2,301 | Official Gerbil reference documentation |
| cookbook | 1,783 | Verified working code recipes with imports |
| api | 1,635 | Individual API function documentation |
| tutorial | 142 | Tutorial source code and explanations |
| test | 91 | Test files showing real API usage |
| security | 70 | Security vulnerability patterns and fixes |
| source | 34 | Tutorial and example .ss source files |
| gambit | 31 | Gambit Scheme examples (FFI, threading) |
| std-source | 21 | Standard library implementation excerpts |
| errorfix | 1 | Error message to fix mappings |

## Data Sources

1. **cookbooks.json** (683 recipes) - Curated, verified Gerbil code patterns with correct imports, arities, and gotcha documentation
2. **security-rules.json** (35 rules) - Vulnerability patterns for Gerbil FFI and C code
3. **error-fixes.json** - Error message to fix mappings
4. **gerbil-mcp resource docs** - Idiom guides, stdlib map, pattern matching, actors, FFI interop
5. **Gerbil official docs** (~150 .md files) - Full API reference, guides, tutorials
6. **Gerbil source code** (~830 .ss files) - Standard library and core implementation
7. **Gambit examples** - FFI examples, threading, Tcl/Tk, web-repl
8. **Test files** (~100 test files) - Real API usage patterns

## ChatML Format

Each entry in `training_data.jsonl`:

```json
{
  "conversations": [
    {"role": "system", "content": "You are an expert in Gerbil Scheme..."},
    {"role": "user", "content": "How do I parse JSON in Gerbil?"},
    {"role": "assistant", "content": "You'll need to import :std/text/json\n\n```scheme\n(import :std/text/json)\n(def data (string->json-object \"{\\\"key\\\": \\\"value\\\"}\"))\n```\n\n**Notes:** ..."}
  ],
  "source": "cookbook:json-parse:howto"
}
```

## Alpaca Format

Each entry in `training_data_alpaca.jsonl`:

```json
{
  "instruction": "How do I parse JSON in Gerbil?",
  "input": "",
  "output": "You'll need to import :std/text/json ...",
  "source": "cookbook:json-parse:howto"
}
```

## Entry Types

- **howto** - "How do I X in Gerbil?" with complete code and imports
- **example** - "Show me an example of X" with just code + notes
- **imports** - "What imports do I need for X?"
- **gotcha** - "What's wrong with this code?" showing common mistakes
- **security** - "Is this safe?" pattern for FFI and shell code
- **errorfix** - "I'm getting error X, how do I fix it?"
- **full** - Complete document or source file as a teaching entry
- **section** - Individual documentation section as a focused Q&A

## Usage

### With LLaMA-Factory

```yaml
dataset_info:
  gerbil_scheme:
    file_name: training_data.jsonl
    formatting: sharegpt
    columns:
      messages: conversations
```

### With Axolotl

```yaml
datasets:
  - path: ./training_data.jsonl
    type: sharegpt
```

### With Unsloth / HuggingFace

Load as a standard JSONL dataset and use the Alpaca format.

## Regenerating

```bash
python3 convert_training_data.py
```

Requires:
- `~/mine/gerbil-mcp/` - MCP server repo with cookbook/security data
- `~/mine/gerbil/` - Gerbil Scheme source repo
- `~/mine/gambit/` - Gambit Scheme source repo
# gerbil-lora
