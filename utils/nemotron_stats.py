"""
Statistics for nvidia/Nemotron-Terminal-Corpus.

Counts complete (task_complete=true) and still-running (task_complete=false)
trajectories per difficulty/skill category and in total.
"""

import json
import re
from collections import defaultdict

import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID = "nvidia/Nemotron-Terminal-Corpus"
SKILL_BASE = "synthetic_tasks/skill_based"


def extract_task_complete(last_content: str) -> bool | None:
    """Return True/False from task_complete field in last assistant message, or None if missing."""
    match = re.search(r'"task_complete"\s*:\s*(true|false)', last_content)
    if match:
        return match.group(1) == "true"
    return None


def count_parquet(parquet_path: str) -> dict:
    df = pd.read_parquet(parquet_path)
    complete = 0
    running = 0
    unknown = 0
    for _, row in df.iterrows():
        convs = row["conversations"]
        last = convs[-1]["content"] if len(convs) > 0 else ""
        status = extract_task_complete(last)
        if status is True:
            complete += 1
        else:
            # task_complete=false OR missing (truncated/malformed output = still running)
            running += 1
    return {"complete": complete, "running": running, "total": len(df)}


def main():
    repo_files = sorted(
        f for f in list_repo_files(REPO_ID, repo_type="dataset")
        if f.startswith(SKILL_BASE) and f.endswith("data_filtered.parquet")
    )

    totals = defaultdict(int)
    by_difficulty = defaultdict(lambda: defaultdict(int))

    header = f"{'Category':<45} {'Complete':>10} {'Running':>10} {'Total':>8}"
    print(header)
    print("-" * len(header))

    for repo_file in repo_files:
        parts = repo_file[len(SKILL_BASE) + 1:].split("/")
        difficulty, skill = parts[0], parts[1]
        label = f"{difficulty}/{skill}"

        parquet_path = hf_hub_download(repo_id=REPO_ID, filename=repo_file, repo_type="dataset")
        counts = count_parquet(parquet_path)

        print(
            f"{label:<45} {counts['complete']:>10,} {counts['running']:>10,}"
            f" {counts['total']:>8,}"
        )

        for k, v in counts.items():
            totals[k] += v
            by_difficulty[difficulty][k] += v

    print("-" * len(header))
    print("\nBy difficulty:")
    for diff, counts in sorted(by_difficulty.items()):
        print(
            f"  {diff:<10}  complete={counts['complete']:,}  running={counts['running']:,}"
            f"  total={counts['total']:,}"
        )

    print(
        f"\nGrand total:  complete={totals['complete']:,}  running={totals['running']:,}"
        f"  total={totals['total']:,}"
    )


if __name__ == "__main__":
    main()
