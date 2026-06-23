import json
import os
from datasets import load_dataset

DATA_DIR = "/home/cuongdc/CODE_REASONING/data"
OUT_PATH = os.path.join(DATA_DIR, "terminal_traj.json")
DATASET_INFO_PATH = os.path.join(DATA_DIR, "dataset_info.json")

role_map = {"user": "human", "assistant": "gpt"}

ds = load_dataset("m-a-p/TerminalTraj", split="train")

output = []
for row in ds:
    conversations = [
        {"from": role_map.get(turn["role"], turn["role"]), "value": turn["content"]}
        for turn in row["messages"]
    ]
    output.append({"conversations": conversations})

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"Written {len(output)} conversations to {OUT_PATH}")

# Update dataset_info.json
with open(DATASET_INFO_PATH, "r", encoding="utf-8") as f:
    dataset_info = json.load(f)

dataset_info["terminal_traj"] = {
    "file_name": "terminal_traj.json",
    "formatting": "sharegpt",
    "columns": {
        "messages": "conversations"
    },
}

with open(DATASET_INFO_PATH, "w", encoding="utf-8") as f:
    json.dump(dataset_info, f, ensure_ascii=False, indent=2)

print("Added 'terminal_traj' entry to dataset_info.json")
