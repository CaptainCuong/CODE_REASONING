#!/bin/bash


#-- SLURM Job Directives --#

#SBATCH --nodes=1                   # Request a single node

#SBATCH --ntasks-per-node=2         # Request 2 CPU cores

#SBATCH --time=12:00:00              # Set a 12-hour time limit

#SBATCH --partition=h200_normal_q   # Specify the GPU partition: h200_normal_q, a100_normal_q on Tinkercliffs | a30_normal_q o>

#SBATCH --account=cuong            # Your class-specific account: cuong, ece6514

#SBATCH --gres=gpu:4                # Request 2 GPUs

# Instructions for running ARC:
# https://docs.arc.vt.edu/usage/vscode_remote_ssh.html

module load CUDA/12.6.0

source ~/.bashrc
conda activate llama310
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "Using Python: $(which python)"
echo "Python version: $(python --version)"

echo "Training on Security..."

DATASETS="nemotron_easy_security,\
nemotron_medium_security,\
nemotron_mixed_security"

~/miniconda3/envs/llama310/bin/llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path Qwen/Qwen3-8B \
    --preprocessing_num_workers 64 \
    --finetuning_type full \
    --template qwen3 \
    --flash_attn auto \
    --dataset_dir data \
    --dataset "$DATASETS" \
    --cutoff_len 16384 \
    --learning_rate 1e-05 \
    --num_train_epochs 2.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_strategy epoch \
    --warmup_steps 0 \
    --packing False \
    --enable_thinking True \
    --report_to none \
    --output_dir /projects/ai_safe/cuongdc/Terminal-security \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --deepspeed examples/deepspeed/ds_z3_config.json
