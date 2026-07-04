#!/usr/bin/env python3
"""Split overlong ShareGPT conversations into multiple training examples.

Terminal-bench trajectories can run to tens of thousands of tokens. This
walks each conversation turn-by-turn (human/gpt pairs) and, whenever adding
the next pair would push the running token count past --max-tokens, cuts the
conversation there and starts a new one seeded with the task description
(taken from the first message, up to the "Current terminal state:" marker)
concatenated with the human turn that would have overflowed.

See data/example_trimming.json for a worked example: its second conversation
is exactly what trimming the first one produces.

Usage:
    python3 data_conversion/trim_conversations.py data/nemotron_complete.json
    python3 data_conversion/trim_conversations.py data/ --max-tokens 8192
"""

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

TASK_MARKER = "Current terminal state:"
BATCH_SIZE = 256


def extract_task_prefix(first_human_value: str) -> str:
    idx = first_human_value.find(TASK_MARKER)
    return first_human_value[:idx] if idx != -1 else first_human_value


def batch_token_lengths(tokenizer, texts: list[str]) -> list[int]:
    lengths = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        encoded = tokenizer(batch, add_special_tokens=False)["input_ids"]
        lengths.extend(len(ids) for ids in encoded)
    return lengths


def trim_conversation(
    conversations: list[dict], task_prefix: str, task_prefix_len: int, msg_lengths: list[int], max_tokens: int
) -> list[list[dict]]:
    if not conversations:
        return []

    pairs = [conversations[i : i + 2] for i in range(0, len(conversations), 2)]
    length_pairs = [msg_lengths[i : i + 2] for i in range(0, len(msg_lengths), 2)]

    chunks = []
    current: list[dict] = []
    current_tokens = 0

    for i, (pair, lengths) in enumerate(zip(pairs, length_pairs)):
        plain_pair_tokens = sum(lengths)

        if i == 0:
            current = list(pair)
            current_tokens = plain_pair_tokens
            continue

        if current_tokens + plain_pair_tokens > max_tokens:
            chunks.append(current)
            human, *rest = pair
            seeded_human = {"from": human["from"], "value": task_prefix + human["value"]}
            current = [seeded_human, *rest]
            current_tokens = task_prefix_len + plain_pair_tokens
        else:
            current.extend(pair)
            current_tokens += plain_pair_tokens

    if current:
        chunks.append(current)

    return chunks


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


def process_file(tokenizer, path: Path, max_tokens: int) -> list[dict]:
    with path.open() as f:
        examples = json.load(f)

    if not isinstance(examples, list) or not examples or "conversations" not in examples[0]:
        print(f"warning: not ShareGPT format, skipping: {path}", file=sys.stderr)
        return []

    task_prefixes = [extract_task_prefix(ex["conversations"][0]["value"]) for ex in examples]

    texts = list(task_prefixes)
    offsets = []
    for ex in examples:
        offsets.append(len(texts))
        texts.extend(m["value"] for m in ex["conversations"])

    lengths = batch_token_lengths(tokenizer, texts)
    task_prefix_lens = lengths[: len(examples)]

    output = []
    for idx, example in enumerate(examples):
        n_msgs = len(example["conversations"])
        msg_lengths = lengths[offsets[idx] : offsets[idx] + n_msgs]
        for chunk in trim_conversation(
            example["conversations"], task_prefixes[idx], task_prefix_lens[idx], msg_lengths, max_tokens
        ):
            output.append({"conversations": chunk})

    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="ShareGPT JSON files and/or directories containing them")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    parser.add_argument("--suffix", default="_trimmed", help="Suffix inserted before .json in output filenames")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    files = resolve_files(args.paths)
    if not files:
        print("no files found", file=sys.stderr)
        sys.exit(1)

    for path in files:
        with path.open() as f:
            n_in = len(json.load(f))

        output = process_file(tokenizer, path, args.max_tokens)
        if not output:
            continue

        out_path = path.with_name(f"{path.stem}{args.suffix}.json")
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"{path.name:<45} {n_in:>7,} -> {len(output):>7,} conversations  ({out_path.name})")


if __name__ == "__main__":
    main()
