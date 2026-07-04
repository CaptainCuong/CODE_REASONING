#!/usr/bin/env bash
set -euo pipefail

trap 'tput cnorm; tput sgr0' EXIT

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llama311

export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

MODEL="nvidia/Nemotron-Terminal-8B"
MODEL="meta-llama/Llama-3.2-3B-Instruct"
MODEL="/helios-storage/helios4-data/cuong/model/ppo-openthinker/ckpts/global_step_50/policy"
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
    --served-model-name "llama-3.2-3b-instruct"