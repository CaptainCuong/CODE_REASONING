set +u
source "/home/nvidia/miniconda3/etc/profile.d/conda.sh"
conda activate coding-agent
set -u
export OPENAI_API_KEY=dummy-key-for-local-vllm

docker network prune -f

tb run \
    --dataset-path ./benchmark_tasks_v1 \
    --agent terminus-2 \
    --model openai/Terminal-complete_4k \
    --agent-kwarg api_base=http://localhost:8000/v1 \
    --agent-kwarg 'model_info={"max_input_tokens":30000,"max_tokens":30000,"max_output_tokens":30000,"input_cost_per_token":0,"output_cost_per_token":0,"litellm_provider":"openai","mode":"chat"}' \
    --n-concurrent 1 \
    --n-attempts 8