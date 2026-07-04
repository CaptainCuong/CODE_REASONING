#!/usr/bin/env python3
"""Report conversation token-length stats for ShareGPT-format JSON files.

Accepts a mix of files and directories. Directories are scanned (non-recursively)
for *.json files; files that aren't ShareGPT-format ({"conversations": [...]})
are skipped with a warning.

Example:
    python3 utils/count_tokens.py data/nemotron_complete.json data/nemotron_mixed_data_processing.json
    python3 utils/count_tokens.py data/ --tokenizer Qwen/Qwen3-8B
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

from transformers import AutoTokenizer

BATCH_SIZE = 256


def to_messages(conversations: list[dict]) -> list[dict]:
    role_map = {"human": "user", "gpt": "assistant", "system": "system"}
    return [{"role": role_map.get(c["from"], c["from"]), "content": c["value"]} for c in conversations]


def resolve_files(paths: list[str]) -> list[Path]:
    files = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            files.extend(sorted(path.glob("*.json")))
        elif path.is_file():
            files.append(path)
        else:
            print(f"warning: path not found, skipping: {path}", file=sys.stderr)
    return files


def token_lengths(tokenizer, examples: list[dict]) -> list[int]:
    lengths = []
    texts = []
    for example in examples:
        convs = example.get("conversations")
        if not convs:
            continue
        texts.append(tokenizer.apply_chat_template(to_messages(convs), tokenize=False))

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        encoded = tokenizer(batch, add_special_tokens=False)["input_ids"]
        lengths.extend(len(ids) for ids in encoded)

    return lengths


def print_stats(label: str, lengths: list[int]) -> None:
    if not lengths:
        print(f"{label}: no examples")
        return
    sorted_lengths = sorted(lengths)
    n = len(sorted_lengths)
    p50 = sorted_lengths[int(n * 0.50)]
    p90 = sorted_lengths[min(int(n * 0.90), n - 1)]
    p99 = sorted_lengths[min(int(n * 0.99), n - 1)]
    print(
        f"{label:<50} n={n:>7,}  total={sum(lengths):>12,}  "
        f"mean={statistics.mean(lengths):>8.1f}  min={min(lengths):>7,}  "
        f"p50={p50:>7,}  p90={p90:>7,}  p99={p99:>7,}  max={max(lengths):>7,}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="ShareGPT JSON files and/or directories containing them")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-8B", help="HF tokenizer name or path")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    files = resolve_files(args.paths)
    if not files:
        print("no files found", file=sys.stderr)
        sys.exit(1)

    all_lengths = []
    for path in files:
        with path.open() as f:
            examples = json.load(f)

        if not isinstance(examples, list) or not examples or "conversations" not in examples[0]:
            print(f"warning: not ShareGPT format, skipping: {path}", file=sys.stderr)
            continue

        lengths = token_lengths(tokenizer, examples)
        print_stats(path.name, lengths)
        all_lengths.extend(lengths)

    print("-" * 100)
    print_stats("TOTAL", all_lengths)


if __name__ == "__main__":
    main()
