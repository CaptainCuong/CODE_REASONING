import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

DATA_DIR = "/home/cuong/CODE_REASONING/data"
DATASET_INFO_PATH = os.path.join(DATA_DIR, "dataset_info.json")

with open(DATASET_INFO_PATH) as f:
    dataset_info = json.load(f)

# Collect counts per difficulty and skill
counts = {}  # (difficulty, skill) -> count
for key, info in dataset_info.items():
    if not key.startswith("nemotron_"):
        continue
    _, difficulty, *skill_parts = key.split("_")
    skill = "_".join(skill_parts)
    file_path = os.path.join(DATA_DIR, info["file_name"])
    with open(file_path) as f:
        n = len(json.load(f))
    counts[(difficulty, skill)] = n

difficulties = ["easy", "medium", "mixed"]
diff_colors = {"easy": "#4CAF50", "medium": "#2196F3", "mixed": "#FF9800"}

all_skills = sorted({s for _, s in counts})
skill_cmap = plt.colormaps["tab20"]
skill_colors = {sk: skill_cmap(i / len(all_skills)) for i, sk in enumerate(all_skills)}

# ── Figure layout: outer ring = difficulty, inner ring = skill ──────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle("Nemotron Terminal Corpus – Taxonomy", fontsize=16, fontweight="bold", y=1.01)

# ── Left: nested donut ─────────────────────────────────────────────────────
ax = axes[0]
ax.set_aspect("equal")
ax.set_title("Distribution by Difficulty & Skill", fontsize=13, pad=12)

# Outer ring – difficulty
outer_sizes, outer_colors, outer_labels = [], [], []
for diff in difficulties:
    total = sum(v for (d, s), v in counts.items() if d == diff)
    outer_sizes.append(total)
    outer_colors.append(diff_colors[diff])
    outer_labels.append(f"{diff}\n({total:,})")

wedges_out, _ = ax.pie(
    outer_sizes,
    labels=outer_labels,
    colors=outer_colors,
    radius=1.0,
    startangle=90,
    wedgeprops=dict(width=0.3, edgecolor="white", linewidth=1.5),
    labeldistance=1.12,
    textprops={"fontsize": 10},
)

# Inner ring – skill within each difficulty
inner_sizes, inner_colors = [], []
for diff in difficulties:
    for skill in all_skills:
        v = counts.get((diff, skill), 0)
        inner_sizes.append(v)
        inner_colors.append(skill_colors[skill])

wedges_in, _ = ax.pie(
    inner_sizes,
    colors=inner_colors,
    radius=0.7,
    startangle=90,
    wedgeprops=dict(width=0.35, edgecolor="white", linewidth=0.8),
)
ax.add_patch(plt.Circle((0, 0), 0.35, color="white"))

# ── Right: stacked bar per skill ───────────────────────────────────────────
ax2 = axes[1]
ax2.set_title("Conversation Count per Skill & Difficulty", fontsize=13, pad=12)

x = np.arange(len(all_skills))
bar_width = 0.55
bottoms = np.zeros(len(all_skills))

for diff in difficulties:
    vals = [counts.get((diff, sk), 0) for sk in all_skills]
    bars = ax2.bar(x, vals, bar_width, bottom=bottoms,
                   color=diff_colors[diff], label=diff, edgecolor="white", linewidth=0.6)
    # Label non-zero segments
    for rect, v, b in zip(bars, vals, bottoms):
        if v > 500:
            ax2.text(rect.get_x() + rect.get_width() / 2,
                     b + v / 2, f"{v:,}",
                     ha="center", va="center", fontsize=6.5, color="white", fontweight="bold")
    bottoms += np.array(vals)

ax2.set_xticks(x)
ax2.set_xticklabels([s.replace("_", "\n") for s in all_skills], fontsize=8.5)
ax2.set_ylabel("Number of conversations", fontsize=10)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax2.legend(title="Difficulty", fontsize=9, title_fontsize=9)
ax2.grid(axis="y", alpha=0.3, linestyle="--")
ax2.spines[["top", "right"]].set_visible(False)

# Skill legend for donut
legend_patches = [
    mpatches.Patch(color=skill_colors[sk], label=sk.replace("_", " "))
    for sk in all_skills
]
fig.legend(handles=legend_patches, title="Skill", fontsize=8, title_fontsize=9,
           loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.08))

plt.tight_layout()
out = "/home/cuong/CODE_REASONING/temp/taxonomy_chart.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.show()
