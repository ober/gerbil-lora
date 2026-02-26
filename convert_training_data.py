#!/usr/bin/env python3
"""
Convert Gerbil Scheme knowledge sources into LoRA training data.

Sources:
  1. cookbooks.json      - 683 verified code recipes
  2. security-rules.json - 35 vulnerability patterns
  3. error-fixes.json    - Error→fix mappings
  4. Resource markdown    - Idiom guides, stdlib map, pattern matching, actors, FFI
  5. Gerbil doc/reference - Official API reference docs
  6. Gerbil doc/tutorials - Official tutorials
  7. Gerbil doc/guide     - Official guide (intro, FFI)
  8. Gerbil source code   - .ss implementation files with doc comments

Output formats:
  - training_data.jsonl  (ChatML / ShareGPT format for most LoRA tools)
  - training_data_alpaca.jsonl (Alpaca format: instruction/input/output)

Each entry is a single-turn or multi-turn conversation teaching
the model about Gerbil Scheme.
"""

import json
import os
import re
import glob
import hashlib
from pathlib import Path
from typing import Optional

# ── Paths ────────────────────────────────────────────────────────────
MCP_DIR     = os.path.expanduser("~/mine/gerbil-mcp")
GERBIL_DIR  = os.path.expanduser("~/mine/gerbil")
GAMBIT_DIR  = os.path.expanduser("~/mine/gambit")
OUTPUT_DIR  = os.path.expanduser("~/mine/gerbil-lora")

SYSTEM_PROMPT = (
    "You are an expert in Gerbil Scheme, a dialect of Scheme built on Gambit. "
    "You provide accurate, idiomatic Gerbil code with correct imports, function "
    "names, and arities. You know the standard library (:std/*), the actor system, "
    "the FFI interface, the macro system (defrules, syntax-case), pattern matching, "
    "the module system, and common gotchas. When writing code, always include "
    "required import statements."
)


def normalize_code(code: str) -> str:
    """Normalize escaped newlines and clean up code strings."""
    code = code.replace("\\n", "\n")
    code = code.replace("\\\"", '"')
    code = code.replace("\\\\", "\\")
    # Remove excessive blank lines
    code = re.sub(r'\n{3,}', '\n\n', code)
    return code.strip()


def make_chatml(conversations: list[dict], source_id: str) -> dict:
    """Create a ChatML/ShareGPT format entry."""
    return {
        "conversations": conversations,
        "source": source_id,
    }


def make_alpaca(instruction: str, output: str, inp: str = "", source_id: str = "") -> dict:
    """Create an Alpaca format entry."""
    return {
        "instruction": instruction,
        "input": inp,
        "output": output,
        "source": source_id,
    }


# ═══════════════════════════════════════════════════════════════════════
# Source 1: Cookbook Recipes
# ═══════════════════════════════════════════════════════════════════════

def convert_cookbooks(recipes: list[dict]) -> tuple[list, list]:
    """Convert cookbook recipes into training pairs."""
    chatml_entries = []
    alpaca_entries = []

    for r in recipes:
        if r.get("deprecated"):
            continue

        rid   = r["id"]
        title = r["title"]
        tags  = r.get("tags", [])
        imports = r.get("imports", [])
        code  = normalize_code(r["code"])
        notes = r.get("notes", "")
        version = r.get("gerbil_version", "")

        # ── Variant 1: "How do I..." question ──────────────────────
        question = f"How do I {title.lower().rstrip('.')} in Gerbil Scheme?"
        answer_parts = []
        if imports:
            answer_parts.append(f"You'll need to import: {', '.join(imports)}")
        answer_parts.append(f"Here's how:\n\n```scheme\n{code}\n```")
        if notes:
            answer_parts.append(f"\n**Notes:** {notes}")
        answer = "\n\n".join(answer_parts)

        chatml_entries.append(make_chatml([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ], f"cookbook:{rid}:howto"))

        alpaca_entries.append(make_alpaca(question, answer, source_id=f"cookbook:{rid}:howto"))

        # ── Variant 2: "Show me an example of..." ──────────────────
        example_q = f"Show me an example of {title.lower().rstrip('.')} in Gerbil Scheme."
        example_a = f"```scheme\n{code}\n```"
        if notes:
            example_a += f"\n\n{notes}"

        chatml_entries.append(make_chatml([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example_q},
            {"role": "assistant", "content": example_a},
        ], f"cookbook:{rid}:example"))

        alpaca_entries.append(make_alpaca(example_q, example_a, source_id=f"cookbook:{rid}:example"))

        # ── Variant 3: "What imports do I need for..." ──────────────
        if imports:
            import_q = f"What imports do I need to {title.lower().rstrip('.')} in Gerbil?"
            import_lines = "\n".join(f"(import {i})" for i in imports)
            import_a = f"You need:\n\n```scheme\n{import_lines}\n```"
            if notes and "import" in notes.lower():
                import_a += f"\n\n{notes}"

            chatml_entries.append(make_chatml([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": import_q},
                {"role": "assistant", "content": import_a},
            ], f"cookbook:{rid}:imports"))

            alpaca_entries.append(make_alpaca(import_q, import_a, source_id=f"cookbook:{rid}:imports"))

        # ── Variant 4: Gotcha recipes get "What's wrong with..." ────
        if "gotcha" in tags or "GOTCHA" in title.upper():
            # Extract the wrong pattern if present
            wrong_match = re.search(r';;\s*WRONG[:\s]*(.*?)(?:\n;;|\n\n|\Z)', code, re.S)
            if wrong_match:
                wrong_code = wrong_match.group(0).strip()
                gotcha_q = f"What's a common mistake when {' '.join(tags[:3])} in Gerbil Scheme?"
                gotcha_a = f"**Common mistake:**\n\n```scheme\n{wrong_code}\n```\n\n**The fix:** {notes}"

                chatml_entries.append(make_chatml([
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": gotcha_q},
                    {"role": "assistant", "content": gotcha_a},
                ], f"cookbook:{rid}:gotcha"))

                alpaca_entries.append(make_alpaca(gotcha_q, gotcha_a, source_id=f"cookbook:{rid}:gotcha"))

    return chatml_entries, alpaca_entries


# ═══════════════════════════════════════════════════════════════════════
# Source 2: Security Rules
# ═══════════════════════════════════════════════════════════════════════

def convert_security_rules(rules: list[dict]) -> tuple[list, list]:
    """Convert security rules into training pairs about safe Gerbil coding."""
    chatml_entries = []
    alpaca_entries = []

    for rule in rules:
        rid = rule["id"]
        title = rule["title"]
        severity = rule["severity"]
        scope = rule["scope"]
        message = rule["message"]
        remediation = rule["remediation"]
        tags = rule.get("tags", [])

        # ── Variant 1: Security question ────────────────────────────
        q = f"What is the security risk of {title.lower()} in Gerbil Scheme?"
        a = (
            f"**Severity:** {severity}\n"
            f"**Scope:** {scope}\n\n"
            f"**Risk:** {message}\n\n"
            f"**Fix:** {remediation}"
        )

        chatml_entries.append(make_chatml([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ], f"security:{rid}"))

        alpaca_entries.append(make_alpaca(q, a, source_id=f"security:{rid}"))

        # ── Variant 2: "Is this safe?" pattern ─────────────────────
        safe_q = f"Is it safe to use {' '.join(tags[:3])} in Gerbil FFI code? What should I watch out for?"
        safe_a = f"{message}\n\n**Remediation:** {remediation}"

        chatml_entries.append(make_chatml([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": safe_q},
            {"role": "assistant", "content": safe_a},
        ], f"security:{rid}:safe"))

        alpaca_entries.append(make_alpaca(safe_q, safe_a, source_id=f"security:{rid}:safe"))

    return chatml_entries, alpaca_entries


# ═══════════════════════════════════════════════════════════════════════
# Source 3: Error Fixes
# ═══════════════════════════════════════════════════════════════════════

def convert_error_fixes(fixes: list[dict]) -> tuple[list, list]:
    """Convert error→fix mappings into training pairs."""
    chatml_entries = []
    alpaca_entries = []

    for fix in fixes:
        if fix["id"].startswith("test-"):
            continue

        pattern = fix.get("pattern", "")
        fix_text = fix.get("fix", "")
        code_example = fix.get("code_example", "")
        wrong_example = fix.get("wrong_example", "")
        fix_type = fix.get("type", "Error")

        q = f'I\'m getting this Gerbil error: "{pattern}". How do I fix it?'
        a_parts = [f"**Type:** {fix_type}\n\n**Explanation:** {fix_text}"]
        if wrong_example:
            a_parts.append(f"**Wrong:**\n```scheme\n{normalize_code(wrong_example)}\n```")
        if code_example:
            a_parts.append(f"**Correct:**\n```scheme\n{normalize_code(code_example)}\n```")
        a = "\n\n".join(a_parts)

        chatml_entries.append(make_chatml([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ], f"errorfix:{fix['id']}"))

        alpaca_entries.append(make_alpaca(q, a, source_id=f"errorfix:{fix['id']}"))

    return chatml_entries, alpaca_entries


# ═══════════════════════════════════════════════════════════════════════
# Source 4: Markdown Reference Documents
# ═══════════════════════════════════════════════════════════════════════

def split_markdown_sections(content: str) -> list[tuple[str, str]]:
    """Split markdown into (heading, content) sections."""
    sections = []
    lines = content.split("\n")
    current_heading = ""
    current_body = []

    for line in lines:
        if line.startswith("#"):
            if current_heading or current_body:
                sections.append((current_heading, "\n".join(current_body).strip()))
            current_heading = line.lstrip("#").strip()
            current_body = []
        else:
            current_body.append(line)

    if current_heading or current_body:
        sections.append((current_heading, "\n".join(current_body).strip()))

    return sections


def convert_markdown_doc(filepath: str, doc_type: str = "reference") -> tuple[list, list]:
    """Convert a markdown documentation file into training pairs."""
    chatml_entries = []
    alpaca_entries = []

    try:
        with open(filepath, "r") as f:
            content = f.read()
    except (FileNotFoundError, PermissionError):
        return chatml_entries, alpaca_entries

    if not content.strip():
        return chatml_entries, alpaca_entries

    relpath = os.path.relpath(filepath, GERBIL_DIR) if filepath.startswith(GERBIL_DIR) else os.path.basename(filepath)
    source_id = f"doc:{relpath}"

    # Full document as a single training entry
    # Only for reasonably sized docs (< 8000 chars)
    if len(content) < 8000:
        doc_title = ""
        first_line = content.strip().split("\n")[0]
        if first_line.startswith("#"):
            doc_title = first_line.lstrip("#").strip()

        if doc_title:
            q = f"Explain {doc_title} in Gerbil Scheme."
            chatml_entries.append(make_chatml([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
                {"role": "assistant", "content": content.strip()},
            ], f"{source_id}:full"))
            alpaca_entries.append(make_alpaca(q, content.strip(), source_id=f"{source_id}:full"))

    # Section-level entries for larger docs
    sections = split_markdown_sections(content)
    for heading, body in sections:
        if not heading or not body or len(body) < 100:
            continue

        # Skip purely structural headings
        if heading.lower() in ("table of contents", "contents", "readme"):
            continue

        q = f"Explain {heading} in Gerbil Scheme."
        chatml_entries.append(make_chatml([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
            {"role": "assistant", "content": body},
        ], f"{source_id}:{heading[:40]}"))

        alpaca_entries.append(make_alpaca(q, body, source_id=f"{source_id}:{heading[:40]}"))

    return chatml_entries, alpaca_entries


# ═══════════════════════════════════════════════════════════════════════
# Source 5: API Reference Docs (structured function docs)
# ═══════════════════════════════════════════════════════════════════════

def extract_api_entries(filepath: str) -> tuple[list, list]:
    """Extract individual API function docs from reference markdown files."""
    chatml_entries = []
    alpaca_entries = []

    try:
        with open(filepath, "r") as f:
            content = f.read()
    except (FileNotFoundError, PermissionError):
        return chatml_entries, alpaca_entries

    relpath = os.path.relpath(filepath, GERBIL_DIR)
    source_id = f"api:{relpath}"

    # Find function/macro documentation blocks
    # Pattern: ### `function-name` or ### function-name followed by description
    pattern = re.compile(
        r'^#{2,3}\s+`?([a-zA-Z_!?*<>=+\-/][a-zA-Z0-9_!?*<>=+\-/]*)`?\s*\n(.*?)(?=^#{2,3}\s|\Z)',
        re.MULTILINE | re.DOTALL
    )

    for match in pattern.finditer(content):
        func_name = match.group(1).strip()
        func_doc = match.group(2).strip()

        if len(func_doc) < 50:
            continue

        # Determine module from path
        module = ""
        if "/std/" in filepath:
            parts = filepath.split("/std/")[1].replace(".md", "").replace("/", "/")
            module = f":std/{parts}"
        elif "/gerbil/" in filepath:
            parts = filepath.split("/gerbil/")[1].replace(".md", "").replace("/", "/")
            module = f":gerbil/{parts}"

        q = f"What does `{func_name}` do in Gerbil Scheme?"
        if module:
            q += f" (from {module})"

        chatml_entries.append(make_chatml([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
            {"role": "assistant", "content": func_doc},
        ], f"{source_id}:{func_name}"))

        alpaca_entries.append(make_alpaca(q, func_doc, source_id=f"{source_id}:{func_name}"))

    return chatml_entries, alpaca_entries


# ═══════════════════════════════════════════════════════════════════════
# Source 6: Tutorial Source Code
# ═══════════════════════════════════════════════════════════════════════

def convert_source_file(filepath: str, description: str = "") -> tuple[list, list]:
    """Convert a Gerbil source file into training entries."""
    chatml_entries = []
    alpaca_entries = []

    try:
        with open(filepath, "r") as f:
            content = f.read()
    except (FileNotFoundError, PermissionError):
        return chatml_entries, alpaca_entries

    if not content.strip() or len(content) < 50:
        return chatml_entries, alpaca_entries

    relpath = os.path.relpath(filepath, GERBIL_DIR) if filepath.startswith(GERBIL_DIR) else os.path.basename(filepath)
    source_id = f"source:{relpath}"
    basename = os.path.basename(filepath)

    if not description:
        description = f"the {basename} module"

    # Entry 1: "Show me the source code for..."
    q = f"Show me an example implementation of {description} in Gerbil Scheme."
    a = f"Here's the implementation from `{relpath}`:\n\n```scheme\n{content.strip()}\n```"

    chatml_entries.append(make_chatml([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": q},
        {"role": "assistant", "content": a},
    ], f"{source_id}:full"))

    alpaca_entries.append(make_alpaca(q, a, source_id=f"{source_id}:full"))

    return chatml_entries, alpaca_entries


# ═══════════════════════════════════════════════════════════════════════
# Source 7: Tutorial Docs (narrative + code)
# ═══════════════════════════════════════════════════════════════════════

def convert_tutorial(filepath: str) -> tuple[list, list]:
    """Convert tutorial markdown (narrative + inline code) into training pairs."""
    chatml_entries = []
    alpaca_entries = []

    try:
        with open(filepath, "r") as f:
            content = f.read()
    except (FileNotFoundError, PermissionError):
        return chatml_entries, alpaca_entries

    relpath = os.path.relpath(filepath, GERBIL_DIR)
    source_id = f"tutorial:{relpath}"

    # Full tutorial as a single training entry if small enough
    if len(content) < 12000:
        title = ""
        first_line = content.strip().split("\n")[0]
        if first_line.startswith("#"):
            title = first_line.lstrip("#").strip()

        if title:
            q = f"Walk me through building {title.lower()} in Gerbil Scheme."
            chatml_entries.append(make_chatml([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
                {"role": "assistant", "content": content.strip()},
            ], f"{source_id}:full"))
            alpaca_entries.append(make_alpaca(q, content.strip(), source_id=f"{source_id}:full"))

    # Also create section-level entries
    c, a = convert_markdown_doc(filepath, "tutorial")
    chatml_entries.extend(c)
    alpaca_entries.extend(a)

    return chatml_entries, alpaca_entries


# ═══════════════════════════════════════════════════════════════════════
# Source 8: Gambit Documentation
# ═══════════════════════════════════════════════════════════════════════

def convert_gambit_examples() -> tuple[list, list]:
    """Convert Gambit example files into training pairs about FFI and low-level features."""
    chatml_entries = []
    alpaca_entries = []

    examples = {
        "tcltk": "Tcl/Tk GUI integration via Gambit FFI",
        "web-repl": "web-based REPL server",
        "web-server": "HTTP web server",
        "ring": "distributed ring topology",
        "pthread": "POSIX thread integration",
        "pi": "Pi computation",
        "misc": "miscellaneous Gambit examples",
    }

    for dirname, description in examples.items():
        dirpath = os.path.join(GAMBIT_DIR, "examples", dirname)
        if not os.path.isdir(dirpath):
            continue
        for scm_file in glob.glob(os.path.join(dirpath, "*.scm")):
            try:
                with open(scm_file, "r") as f:
                    content = f.read()
            except:
                continue

            if len(content) < 50:
                continue

            basename = os.path.basename(scm_file)
            source_id = f"gambit:examples/{dirname}/{basename}"
            q = f"Show me a Gambit Scheme example of {description}."
            a = f"Here's `{dirname}/{basename}` from the Gambit examples:\n\n```scheme\n{content.strip()}\n```"

            chatml_entries.append(make_chatml([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ], source_id))

            alpaca_entries.append(make_alpaca(q, a, source_id=source_id))

    return chatml_entries, alpaca_entries


# ═══════════════════════════════════════════════════════════════════════
# Source 9: Gerbil Test Files (as usage examples)
# ═══════════════════════════════════════════════════════════════════════

def convert_test_files() -> tuple[list, list]:
    """Convert Gerbil test files into training pairs showing API usage."""
    chatml_entries = []
    alpaca_entries = []

    test_files = glob.glob(os.path.join(GERBIL_DIR, "src/std/**/*-test.ss"), recursive=True)

    for tf in test_files:
        try:
            with open(tf, "r") as f:
                content = f.read()
        except:
            continue

        if len(content) < 100 or len(content) > 15000:
            continue

        relpath = os.path.relpath(tf, GERBIL_DIR)
        source_id = f"test:{relpath}"

        # Derive module name from test path
        module_path = relpath.replace("src/std/", ":std/").replace("-test.ss", "").replace("/", "/")
        basename = os.path.basename(tf).replace("-test.ss", "")

        q = f"Show me test examples for the {module_path} module in Gerbil Scheme."
        a = f"Here are test examples from `{relpath}`:\n\n```scheme\n{content.strip()}\n```"

        chatml_entries.append(make_chatml([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ], f"{source_id}:full"))

        alpaca_entries.append(make_alpaca(q, a, source_id=f"{source_id}:full"))

    return chatml_entries, alpaca_entries


# ═══════════════════════════════════════════════════════════════════════
# Source 10: Synthesized Q&A from Resource MDs
# ═══════════════════════════════════════════════════════════════════════

def convert_resource_mds() -> tuple[list, list]:
    """Convert gerbil-mcp resource markdown files into focused Q&A pairs."""
    chatml_entries = []
    alpaca_entries = []

    resources_dir = os.path.join(MCP_DIR, "src", "resources")
    if not os.path.isdir(resources_dir):
        return chatml_entries, alpaca_entries

    for md_file in glob.glob(os.path.join(resources_dir, "*.md")):
        c, a = convert_markdown_doc(md_file, "resource")
        chatml_entries.extend(c)
        alpaca_entries.extend(a)

    return chatml_entries, alpaca_entries


# ═══════════════════════════════════════════════════════════════════════
# Source 11: Guide documents (intro.md, ffi.md, etc.)
# ═══════════════════════════════════════════════════════════════════════

def convert_guide_docs() -> tuple[list, list]:
    """Convert Gerbil guide docs into training pairs."""
    chatml_entries = []
    alpaca_entries = []

    guide_dir = os.path.join(GERBIL_DIR, "doc", "guide")
    if not os.path.isdir(guide_dir):
        return chatml_entries, alpaca_entries

    for md_file in glob.glob(os.path.join(guide_dir, "*.md")):
        c, a = convert_markdown_doc(md_file, "guide")
        chatml_entries.extend(c)
        alpaca_entries.extend(a)

    return chatml_entries, alpaca_entries


# ═══════════════════════════════════════════════════════════════════════
# Source 12: All reference docs
# ═══════════════════════════════════════════════════════════════════════

def convert_all_reference_docs() -> tuple[list, list]:
    """Convert ALL reference documentation markdown files."""
    chatml_entries = []
    alpaca_entries = []

    ref_dir = os.path.join(GERBIL_DIR, "doc", "reference")
    if not os.path.isdir(ref_dir):
        return chatml_entries, alpaca_entries

    for md_file in glob.glob(os.path.join(ref_dir, "**", "*.md"), recursive=True):
        # Full doc entries
        c, a = convert_markdown_doc(md_file, "reference")
        chatml_entries.extend(c)
        alpaca_entries.extend(a)

        # Individual API function entries
        c2, a2 = extract_api_entries(md_file)
        chatml_entries.extend(c2)
        alpaca_entries.extend(a2)

    return chatml_entries, alpaca_entries


# ═══════════════════════════════════════════════════════════════════════
# Source 13: Tutorial source code and docs
# ═══════════════════════════════════════════════════════════════════════

def convert_all_tutorials() -> tuple[list, list]:
    """Convert tutorial source code and documentation."""
    chatml_entries = []
    alpaca_entries = []

    # Tutorial source files
    tutorial_dir = os.path.join(GERBIL_DIR, "src", "tutorial")
    if os.path.isdir(tutorial_dir):
        descriptions = {
            "httpd": "a simple HTTP server",
            "kvstore": "a key-value store with RPC",
            "proxy": "a TCP proxy",
            "lang": "a custom language extension",
            "ensemble": "a distributed actor ensemble",
        }
        for ss_file in glob.glob(os.path.join(tutorial_dir, "**", "*.ss"), recursive=True):
            # Determine description from directory
            parts = os.path.relpath(ss_file, tutorial_dir).split(os.sep)
            desc = descriptions.get(parts[0], parts[0]) if parts else "a tutorial"
            c, a = convert_source_file(ss_file, desc)
            chatml_entries.extend(c)
            alpaca_entries.extend(a)

    # Tutorial docs
    tut_doc_dir = os.path.join(GERBIL_DIR, "doc", "tutorials")
    if os.path.isdir(tut_doc_dir):
        for md_file in glob.glob(os.path.join(tut_doc_dir, "*.md")):
            c, a = convert_tutorial(md_file)
            chatml_entries.extend(c)
            alpaca_entries.extend(a)

    return chatml_entries, alpaca_entries


# ═══════════════════════════════════════════════════════════════════════
# Source 14: Std library source code (key modules)
# ═══════════════════════════════════════════════════════════════════════

def convert_std_source_files() -> tuple[list, list]:
    """Convert key standard library source files."""
    chatml_entries = []
    alpaca_entries = []

    key_modules = {
        "src/std/sugar.ss": "syntactic sugar (try/catch, hash, chain, when-let)",
        "src/std/iter.ss": "the iteration framework (for, for/collect, in-range)",
        "src/std/error.ss": "error handling and custom error classes",
        "src/std/test.ss": "the unit testing framework",
        "src/std/sort.ss": "sorting algorithms",
        "src/std/event.ss": "the event system",
        "src/std/coroutine.ss": "coroutines",
        "src/std/amb.ss": "nondeterministic computation (amb operator)",
        "src/std/generic.ss": "generic function dispatch",
        "src/std/interface.ss": "interface definitions",
        "src/std/actor.ss": "the actor system",
        "src/std/misc/hash.ss": "extended hash table operations",
        "src/std/misc/list.ss": "extended list operations",
        "src/std/misc/string.ss": "extended string operations",
        "src/std/misc/path.ss": "filesystem path operations",
        "src/std/misc/channel.ss": "Go-style channels",
        "src/std/misc/threads.ss": "thread utilities",
        "src/std/misc/alist.ss": "association list operations",
        "src/std/misc/bytes.ss": "byte vector operations",
        "src/std/misc/process.ss": "process execution (run-process)",
        "src/std/text/json.ss": "JSON parsing and generation",
    }

    for mod_path, description in key_modules.items():
        filepath = os.path.join(GERBIL_DIR, mod_path)
        # Some modules are just re-exports; try the api.ss variant too
        if not os.path.exists(filepath):
            alt = filepath.replace(".ss", "/api.ss")
            if os.path.exists(alt):
                filepath = alt
            else:
                continue

        try:
            with open(filepath, "r") as f:
                content = f.read()
        except:
            continue

        if len(content) < 50:
            continue

        # Only include if not too large (skip huge files)
        if len(content) > 20000:
            # For large files, just include the export section and first few defs
            lines = content.split("\n")
            truncated = []
            for line in lines[:500]:
                truncated.append(line)
            content = "\n".join(truncated) + "\n;; ... (truncated)"

        source_id = f"std-source:{mod_path}"
        q = f"Show me the implementation of {description} in Gerbil's standard library."
        a = f"Here's the source from `{mod_path}`:\n\n```scheme\n{content.strip()}\n```"

        chatml_entries.append(make_chatml([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ], f"{source_id}:full"))

        alpaca_entries.append(make_alpaca(q, a, source_id=f"{source_id}:full"))

    return chatml_entries, alpaca_entries


# ═══════════════════════════════════════════════════════════════════════
# Deduplication
# ═══════════════════════════════════════════════════════════════════════

def deduplicate(entries: list[dict], key_field: str = "source") -> list[dict]:
    """Remove duplicate entries by source ID."""
    seen = set()
    deduped = []
    for entry in entries:
        sid = entry.get(key_field, "")
        if sid and sid not in seen:
            seen.add(sid)
            deduped.append(entry)
        elif not sid:
            # No source ID, use content hash
            content_hash = hashlib.md5(json.dumps(entry, sort_keys=True).encode()).hexdigest()
            if content_hash not in seen:
                seen.add(content_hash)
                deduped.append(entry)
    return deduped


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_chatml = []
    all_alpaca = []

    # ── 1. Cookbooks ──────────────────────────────────────────────────
    print("Converting cookbooks.json ...")
    with open(os.path.join(MCP_DIR, "cookbooks.json")) as f:
        recipes = json.load(f)
    c, a = convert_cookbooks(recipes)
    all_chatml.extend(c)
    all_alpaca.extend(a)
    print(f"  → {len(c)} ChatML, {len(a)} Alpaca entries")

    # ── 2. Security Rules ─────────────────────────────────────────────
    print("Converting security-rules.json ...")
    with open(os.path.join(MCP_DIR, "security-rules.json")) as f:
        rules = json.load(f)
    c, a = convert_security_rules(rules)
    all_chatml.extend(c)
    all_alpaca.extend(a)
    print(f"  → {len(c)} ChatML, {len(a)} Alpaca entries")

    # ── 3. Error Fixes ────────────────────────────────────────────────
    print("Converting error-fixes.json ...")
    with open(os.path.join(MCP_DIR, "error-fixes.json")) as f:
        fixes = json.load(f)
    c, a = convert_error_fixes(fixes)
    all_chatml.extend(c)
    all_alpaca.extend(a)
    print(f"  → {len(c)} ChatML, {len(a)} Alpaca entries")

    # ── 4. Resource MDs ───────────────────────────────────────────────
    print("Converting gerbil-mcp resource docs ...")
    c, a = convert_resource_mds()
    all_chatml.extend(c)
    all_alpaca.extend(a)
    print(f"  → {len(c)} ChatML, {len(a)} Alpaca entries")

    # ── 5. Guide Docs ─────────────────────────────────────────────────
    print("Converting Gerbil guide docs ...")
    c, a = convert_guide_docs()
    all_chatml.extend(c)
    all_alpaca.extend(a)
    print(f"  → {len(c)} ChatML, {len(a)} Alpaca entries")

    # ── 6. All Reference Docs ─────────────────────────────────────────
    print("Converting Gerbil reference docs ...")
    c, a = convert_all_reference_docs()
    all_chatml.extend(c)
    all_alpaca.extend(a)
    print(f"  → {len(c)} ChatML, {len(a)} Alpaca entries")

    # ── 7. Tutorials ──────────────────────────────────────────────────
    print("Converting tutorials ...")
    c, a = convert_all_tutorials()
    all_chatml.extend(c)
    all_alpaca.extend(a)
    print(f"  → {len(c)} ChatML, {len(a)} Alpaca entries")

    # ── 8. Gambit Examples ────────────────────────────────────────────
    print("Converting Gambit examples ...")
    c, a = convert_gambit_examples()
    all_chatml.extend(c)
    all_alpaca.extend(a)
    print(f"  → {len(c)} ChatML, {len(a)} Alpaca entries")

    # ── 9. Test Files ─────────────────────────────────────────────────
    print("Converting test files ...")
    c, a = convert_test_files()
    all_chatml.extend(c)
    all_alpaca.extend(a)
    print(f"  → {len(c)} ChatML, {len(a)} Alpaca entries")

    # ── 10. Std Source ────────────────────────────────────────────────
    print("Converting std library source ...")
    c, a = convert_std_source_files()
    all_chatml.extend(c)
    all_alpaca.extend(a)
    print(f"  → {len(c)} ChatML, {len(a)} Alpaca entries")

    # ── Dedup and write ──────────────────────────────────────────────
    print(f"\nTotal before dedup: {len(all_chatml)} ChatML, {len(all_alpaca)} Alpaca")

    all_chatml = deduplicate(all_chatml, "source")
    all_alpaca = deduplicate(all_alpaca, "source")

    print(f"Total after dedup:  {len(all_chatml)} ChatML, {len(all_alpaca)} Alpaca")

    # Write ChatML/ShareGPT format
    chatml_path = os.path.join(OUTPUT_DIR, "training_data.jsonl")
    with open(chatml_path, "w") as f:
        for entry in all_chatml:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"\nWrote {chatml_path}")

    # Write Alpaca format
    alpaca_path = os.path.join(OUTPUT_DIR, "training_data_alpaca.jsonl")
    with open(alpaca_path, "w") as f:
        for entry in all_alpaca:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Wrote {alpaca_path}")

    # Write a combined JSON array (some tools prefer this)
    alpaca_json_path = os.path.join(OUTPUT_DIR, "training_data_alpaca.json")
    with open(alpaca_json_path, "w") as f:
        json.dump(all_alpaca, f, ensure_ascii=False, indent=2)
    print(f"Wrote {alpaca_json_path}")

    # Print stats by source
    print("\n── Stats by source category ──")
    source_counts = {}
    for entry in all_chatml:
        src = entry.get("source", "unknown")
        category = src.split(":")[0]
        source_counts[category] = source_counts.get(category, 0) + 1
    for cat, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    # Print total size
    chatml_size = os.path.getsize(chatml_path)
    alpaca_size = os.path.getsize(alpaca_path)
    print(f"\nFile sizes: ChatML={chatml_size/1024/1024:.1f}MB, Alpaca={alpaca_size/1024/1024:.1f}MB")


if __name__ == "__main__":
    main()
