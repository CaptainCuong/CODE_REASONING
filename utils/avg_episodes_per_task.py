#!/usr/bin/env python3
"""Report the average number of episodes per run, grouped by task.

Prints two sets of groups:
  1-3. Task-level success buckets: no success, exactly 1 success, more than 1 success
       (based on `is_resolved` in each run's results.json).
  4-5. Run-level completion buckets: complete runs, incomplete runs
       (based on `is_task_complete` in the last episode's response.json --
       i.e. whether the agent itself declared the task done, right or wrong).

An "episode" is one `episode-N` directory under a run's `agent-logs/` folder.
A "run" is a trial directory like `<task>.<i>-of-<n>.<timestamp>/`.
A "task" is the folder containing all runs/trials for that task.

Example usage:
    python3 utils/avg_episodes_per_task.py runs/Terminal-complete_k8
"""

import json
import sys
from pathlib import Path
from statistics import mean


def episode_dirs(run_dir: Path) -> list[Path]:
    agent_logs = run_dir / "agent-logs"
    if not agent_logs.is_dir():
        return []
    dirs = [p for p in agent_logs.iterdir() if p.is_dir() and p.name.startswith("episode-")]
    return sorted(dirs, key=lambda p: int(p.name.split("-", 1)[1]))


def is_resolved(run_dir: Path) -> bool:
    results_file = run_dir / "results.json"
    if not results_file.is_file():
        return False
    with results_file.open() as f:
        data = json.load(f)
    return bool(data.get("is_resolved"))


def is_task_complete(episodes: list[Path]) -> bool:
    if not episodes:
        return False
    response_file = episodes[-1] / "response.json"
    if not response_file.is_file():
        return False
    with response_file.open() as f:
        data = json.load(f)
    return bool(data.get("is_task_complete"))


def print_group_table(
    title: str, name_width: int, count_label: str, rows: dict[str, tuple[int, int, float, list[int]]]
) -> list[int]:
    print(f"=== {title}: {len(rows)} task(s) ===")
    print(f"{'Task':<{name_width}}  {'Runs':>5}  {count_label:>9}  {'AvgEpisodes':>12}")
    group_counts: list[int] = []
    for task_name in sorted(rows):
        n_runs, n_marked, avg_ep, counts = rows[task_name]
        group_counts.extend(counts)
        print(f"{task_name:<{name_width}}  {n_runs:>5}  {n_marked:>9}  {avg_ep:>12.2f}")
    if group_counts:
        print(f"--> Group average episodes/run: {mean(group_counts):.2f} across {len(group_counts)} runs")
    print()
    return group_counts


def avg_episodes_per_task(root: str) -> None:
    root_path = Path(root)
    if not root_path.is_dir():
        print(f"Error: {root_path} not found", file=sys.stderr)
        sys.exit(1)

    # task_name -> list of (episode_count, resolved, task_complete)
    task_runs: dict[str, list[tuple[int, bool, bool]]] = {}
    for task_dir in sorted(p for p in root_path.iterdir() if p.is_dir()):
        run_dirs = sorted(p for p in task_dir.iterdir() if p.is_dir())
        if not run_dirs:
            continue
        records = []
        for run_dir in run_dirs:
            episodes = episode_dirs(run_dir)
            records.append((len(episodes), is_resolved(run_dir), is_task_complete(episodes)))
        task_runs[task_dir.name] = records

    if not task_runs:
        print(f"No tasks found under {root_path}", file=sys.stderr)
        sys.exit(1)

    name_width = max(len(name) for name in task_runs)
    all_counts: list[int] = []

    # --- Groups 1-3: task-level success buckets ---
    task_successes = {t: sum(r[1] for r in recs) for t, recs in task_runs.items()}
    success_groups: list[tuple[str, list[str]]] = [
        ("No success (0)", [t for t, s in task_successes.items() if s == 0]),
        ("Exactly 1 success", [t for t, s in task_successes.items() if s == 1]),
        ("More than 1 success (>1)", [t for t, s in task_successes.items() if s > 1]),
    ]
    for group_title, task_names in success_groups:
        rows = {}
        for task_name in task_names:
            counts = [r[0] for r in task_runs[task_name]]
            rows[task_name] = (len(counts), task_successes[task_name], mean(counts), counts)
        group_counts = print_group_table(group_title, name_width, "Successes", rows)
        all_counts.extend(group_counts)

    print(f"Overall average episodes/run: {mean(all_counts):.2f} across {len(all_counts)} runs, {len(task_runs)} tasks")
    print()

    # --- Groups 4-5: run-level completion buckets (agent self-declared done) ---
    completion_groups: list[tuple[str, bool]] = [
        ("Complete runs (agent declared done)", True),
        ("Incomplete runs (agent never declared done)", False),
    ]
    for group_title, want_complete in completion_groups:
        rows = {}
        for task_name, recs in task_runs.items():
            counts = [r[0] for r in recs if r[2] == want_complete]
            if not counts:
                continue
            n_marked = sum(1 for r in recs if r[1] and r[2] == want_complete)
            rows[task_name] = (len(counts), n_marked, mean(counts), counts)
        print_group_table(group_title, name_width, "Resolved", rows)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <runs_root_folder>", file=sys.stderr)
        sys.exit(1)
    avg_episodes_per_task(sys.argv[1])
