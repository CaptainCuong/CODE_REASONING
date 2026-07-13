#!/bin/bash

echo "Conda env: $CONDA_DEFAULT_ENV"
echo "Using Python: $(which python)"
echo "Python version: $(python --version)"

echo "Training on Debugging..."

DATASETS="nemotron_easy_debugging,\
nemotron_medium_debugging,\
nemotron_mixed_debugging"

llamafactory-cli train \
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
    --save_steps 0 \
    --warmup_steps 0 \
    --packing False \
    --enable_thinking True \
    --report_to none \
    --output_dir /data/cuong/Terminal-debugging \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --deepspeed examples/deepspeed/ds_z3_config.json
