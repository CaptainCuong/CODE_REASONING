#!/usr/bin/env python3
"""Split all data/nemotron_{easy,medium,mixed}_*.json files into two files based on
whether their trajectory's final task_complete flag is true.

Uses the same completeness rule as nemotron_stats.py: complete = last gpt turn
contains "task_complete": true; task_complete=false or missing = incomplete.

Writes data/nemotron_complete.json and data/nemotron_incomplete.json.
"""

import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_COMPLETE = DATA_DIR / "nemotron_complete.json"
OUT_INCOMPLETE = DATA_DIR / "nemotron_incomplete.json"


def extract_task_complete(last_value: str) -> bool | None:
    match = re.search(r'"task_complete"\s*:\s*(true|false)', last_value)
    if match:
        return match.group(1) == "true"
    return None


def main() -> None:
    source_files = sorted(DATA_DIR.glob("nemotron_*.json"))

    complete, incomplete = [], []

    for path in source_files:
        with path.open() as f:
            examples = json.load(f)

        for example in examples:
            convs = example["conversations"]
            last_value = convs[-1]["value"] if convs else ""
            status = extract_task_complete(last_value)
            (complete if status is True else incomplete).append(example)

        print(f"{path.name:<45} total={len(examples):>6,}")

    print(f"\nComplete:   {len(complete):,}")
    print(f"Incomplete: {len(incomplete):,}")

    with OUT_COMPLETE.open("w") as f:
        json.dump(complete, f)
    with OUT_INCOMPLETE.open("w") as f:
        json.dump(incomplete, f)

    print(f"\nWrote {OUT_COMPLETE}")
    print(f"Wrote {OUT_INCOMPLETE}")


if __name__ == "__main__":
    main()
