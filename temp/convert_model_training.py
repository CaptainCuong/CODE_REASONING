import pandas as pd
import json
from huggingface_hub import hf_hub_download

PARQUET_PATH = hf_hub_download(
    repo_id="nvidia/Nemotron-Terminal-Corpus",
    filename="synthetic_tasks/skill_based/medium/model_training/data_filtered.parquet",
    repo_type="dataset",
)
OUT_PATH = "/home/cuongdc/CODE_REASONING/data/model_training.json"

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
