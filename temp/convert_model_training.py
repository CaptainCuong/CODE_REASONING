import pandas as pd
import json

PARQUET_PATH = (
    "/helios-storage/helios4-data/cuong/hub/"
    "datasets--nvidia--Nemotron-Terminal-Corpus/snapshots/"
    "a1667c4ffdadea02a89bffe4f1bb7ca2ff19f8d9/"
    "synthetic_tasks/skill_based/medium/model_training/data_filtered.parquet"
)
OUT_PATH = "/home/cuong/CODE_REASONING/data/model_training.json"

role_map = {"user": "human", "assistant": "gpt"}

df = pd.read_parquet(PARQUET_PATH)

output = []
for _, row in df.iterrows():
    conversations = [
        {"from": role_map.get(turn["role"], turn["role"]), "value": turn["content"]}
        for turn in row["conversations"]
    ]
    output.append({"conversations": conversations})

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"Written {len(output)} conversations to {OUT_PATH}")
