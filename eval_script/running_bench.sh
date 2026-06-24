docker network prune -f

tb run \
    --dataset-path ./data_science_tasks \
    --agent terminus \
    --model openai/terminal-qwen-data-science-trajs \
    --agent-kwarg api_base=http://localhost:8000/v1 \
    --agent-kwarg 'model_info={"max_input_tokens":30000,"max_tokens":30000,"max_output_tokens":30000,"input_cost_per_token":0,"output_cost_per_token":0,"litellm_provider":"openai","mode":"chat"}' \
    --n-concurrent 1