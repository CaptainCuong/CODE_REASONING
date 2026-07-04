#!/usr/bin/env python3
"""List tasks that passed at least one attempt in a run folder.
Example usage:
    python3 utils/list_correct_tasks.py runs/2026-06-26__12-14-21
"""

import json
import sys
from collections import defaultdict
from pathlib import Path


def list_correct_tasks(run_folder: str) -> None:
    run_path = Path(run_folder)
    results_file = run_path / "results.json"

    if not results_file.exists():
        print(f"Error: {results_file} not found", file=sys.stderr)
        sys.exit(1)

    with results_file.open() as f:
        data = json.load(f)

    results = data if isinstance(data, list) else data.get("results", [])

    task_attempts: dict[str, list[bool]] = defaultdict(list)
    for trial in results:
        task_id = trial.get("task_id", "unknown")
        is_resolved = trial.get("is_resolved")
        task_attempts[task_id].append(bool(is_resolved))

    correct = sorted(t for t, attempts in task_attempts.items() if any(attempts))
    incorrect = sorted(t for t, attempts in task_attempts.items() if not any(attempts))

    print(f"Run: {run_path.name}")
    print(f"Tasks: {len(task_attempts)} total, {len(correct)} correct, {len(incorrect)} incorrect\n")

    print(f"Correct ({len(correct)}):")
    for task in correct:
        attempts = task_attempts[task]
        n_correct = sum(attempts)
        print(f"  ✓  {task}  ({n_correct}/{len(attempts)} attempts)")

    if incorrect:
        print(f"\nIncorrect ({len(incorrect)}):")
        for task in incorrect:
            attempts = task_attempts[task]
            print(f"  ✗  {task}  (0/{len(attempts)} attempts)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <run_folder>", file=sys.stderr)
        sys.exit(1)
    list_correct_tasks(sys.argv[1])