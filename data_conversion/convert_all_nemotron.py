import json
import os
import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID = "nvidia/Nemotron-Terminal-Corpus"
SKILL_BASE = "synthetic_tasks/skill_based"
DATA_DIR = "/home/cuongdc/CODE_REASONING/data"
DATASET_INFO_PATH = os.path.join(DATA_DIR, "dataset_info.json")

role_map = {"user": "human", "assistant": "gpt"}


def convert(parquet_path: str, out_path: str) -> int:
    df = pd.read_parquet(parquet_path)
    output = []
    for _, row in df.iterrows():
        conversations = [
            {"from": role_map.get(t["role"], t["role"]), "value": t["content"]}
            for t in row["conversations"]
        ]
        output.append({"conversations": conversations})
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    return len(output)


# Load existing dataset_info.json
with open(DATASET_INFO_PATH, "r", encoding="utf-8") as f:
    dataset_info = json.load(f)

# Find all parquet files in the repo and convert
repo_files = sorted(
    f for f in list_repo_files(REPO_ID, repo_type="dataset")
    if f.startswith(SKILL_BASE) and f.endswith("data_filtered.parquet")
)
new_entries = {}

for repo_file in repo_files:
    # e.g. synthetic_tasks/skill_based/easy/data_processing/data_filtered.parquet
    parts = repo_file[len(SKILL_BASE) + 1:].split("/")
    difficulty, skill = parts[0], parts[1]
    parquet_path = hf_hub_download(repo_id=REPO_ID, filename=repo_file, repo_type="dataset")
    key = f"nemotron_{difficulty}_{skill}"
    file_name = f"{key}.json"
    out_path = os.path.join(DATA_DIR, file_name)

    count = convert(parquet_path, out_path)
    print(f"  {key}: {count} conversations -> {file_name}")

    new_entries[key] = {
        "file_name": file_name,
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations"
        },
    }

# Merge into dataset_info and write back
dataset_info.update(new_entries)
with open(DATASET_INFO_PATH, "w", encoding="utf-8") as f:
    json.dump(dataset_info, f, ensure_ascii=False, indent=2)

print(f"\nAdded {len(new_entries)} entries to dataset_info.json")