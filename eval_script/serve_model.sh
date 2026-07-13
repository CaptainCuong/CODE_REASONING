#!/usr/bin/env bash
set -euo pipefail

trap 'tput cnorm; tput sgr0' EXIT
set +u
source "/home/nvidia/miniconda3/etc/profile.d/conda.sh"
conda activate llama310
set -u

# export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

MODEL="nvidia/Nemotron-Terminal-8B"
MODEL="meta-llama/Llama-3.2-3B-Instruct"
MODEL="/helios-storage/helios4-data/cuong/model/ppo-openthinker/ckpts/global_step_50/policy"
MODEL="/data/cuong/Terminal-complete_trimmed/checkpoint-25000"
MODEL="/data/cuong/Terminal-complete_trimmed_4k/checkpoint-25000"
MODEL="/data/cuong/Terminal-complete_4k/checkpoint-14852"
HOST="0.0.0.0"
PORT=8000
TENSOR_PARALLEL=4
GPU_MEMORY_UTILIZATION=0.90
MAX_MODEL_LEN=32768
DTYPE="bfloat16"

export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "Killing any processes on port ${PORT}..."
fuser -k "${PORT}/tcp" 2>/dev/null || true

echo "Starting vLLM server for ${MODEL} on GPUs 0,1,2,3..."

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --tensor-parallel-size "${TENSOR_PARALLEL}" \
    --dtype "${DTYPE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --trust-remote-code \
    --served-model-name "Terminal-complete_4k"