#!/usr/bin/env python3
"""Compare average episodes-per-task between two runs/ roots and export a table.

For each task present in both roots, computes the mean episode count across
all its runs (see avg_episodes_per_task.py for what a "run" and "episode"
are) and reports which root needed more episodes on average.

Example usage:
    python3 utils/compare_avg_episodes.py runs/Terminal-complete_trimmed_8k_k8 runs/Terminal-complete_trimmed_4k_k8 \\
        --output avg_episodes_comparison.md
"""

import argparse
import sys
from pathlib import Path
from statistics import mean

from avg_episodes_per_task import episode_dirs


def task_avg_episodes(root: str) -> dict[str, float]:
    root_path = Path(root)
    if not root_path.is_dir():
        print(f"Error: {root_path} not found", file=sys.stderr)
        sys.exit(1)

    task_avgs = {}
    for task_dir in sorted(p for p in root_path.iterdir() if p.is_dir()):
        run_dirs = sorted(p for p in task_dir.iterdir() if p.is_dir())
        if not run_dirs:
            continue
        counts = [len(episode_dirs(run_dir)) for run_dir in run_dirs]
        task_avgs[task_dir.name] = mean(counts)
    return task_avgs


def build_table(label_a: str, avgs_a: dict[str, float], label_b: str, avgs_b: dict[str, float]) -> str:
    common = sorted(set(avgs_a) & set(avgs_b))
    only_a = sorted(set(avgs_a) - set(avgs_b))
    only_b = sorted(set(avgs_b) - set(avgs_a))

    diffs = [(name, avgs_a[name], avgs_b[name], avgs_a[name] - avgs_b[name]) for name in common]
    diffs.sort(key=lambda x: x[3], reverse=True)

    more_a = [d for d in diffs if d[3] > 0]
    more_b = [d for d in diffs if d[3] < 0]
    equal = [d for d in diffs if d[3] == 0]

    lines = []
    lines.append(f"# Avg episodes per task: {label_a} vs {label_b}\n")
    lines.append(f"Parsed {len(avgs_a)} tasks ({label_a}), {len(avgs_b)} tasks ({label_b}), {len(common)} common.\n")

    def section(title: str, rows: list[tuple[str, float, float, float]]) -> None:
        lines.append(f"## {title} ({len(rows)} tasks)\n")
        lines.append(f"| Task | {label_a} | {label_b} | Diff |")
        lines.append("|---|---|---|---|")
        for name, a, b, d in rows:
            lines.append(f"| {name} | {a:.2f} | {b:.2f} | {d:+.2f} |")
        lines.append("")

    section(f"{label_a} has MORE avg episodes than {label_b}", more_a)
    section(f"{label_b} has MORE avg episodes than {label_a}", more_b)
    if equal:
        section("Equal avg episodes", equal)

    if only_a:
        lines.append(f"## Tasks only in {label_a}\n")
        lines.extend(f"- {n}" for n in only_a)
        lines.append("")
    if only_b:
        lines.append(f"## Tasks only in {label_b}\n")
        lines.extend(f"- {n}" for n in only_b)
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root_a", help="first runs/ root folder")
    parser.add_argument("root_b", help="second runs/ root folder")
    parser.add_argument("--output", default="avg_episodes_comparison.md", help="output markdown file path")
    args = parser.parse_args()

    label_a = Path(args.root_a).name
    label_b = Path(args.root_b).name

    avgs_a = task_avg_episodes(args.root_a)
    avgs_b = task_avg_episodes(args.root_b)

    table = build_table(label_a, avgs_a, label_b, avgs_b)

    out_path = Path(args.output)
    out_path.write_text(table + "\n")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
