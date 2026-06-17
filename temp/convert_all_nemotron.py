import json
import glob
import os
import pandas as pd

SNAPSHOT = (
    "/helios-storage/helios4-data/cuong/hub/"
    "datasets--nvidia--Nemotron-Terminal-Corpus/snapshots/"
    "a1667c4ffdadea02a89bffe4f1bb7ca2ff19f8d9/synthetic_tasks/skill_based"
)
DATA_DIR = "/home/cuong/CODE_REASONING/data"
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

# Find all parquet files and convert
parquet_files = sorted(glob.glob(f"{SNAPSHOT}/*/*/data_filtered.parquet"))
new_entries = {}

for parquet_path in parquet_files:
    # e.g. .../skill_based/easy/data_processing/data_filtered.parquet
    parts = parquet_path.replace(SNAPSHOT + "/", "").split("/")
    difficulty, skill = parts[0], parts[1]
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