#!/bin/bash

DATASETS="nemotron_easy_data_processing,\
nemotron_easy_data_querying,\
nemotron_easy_data_science,\
nemotron_easy_debugging,\
nemotron_easy_dependency_management,\
nemotron_easy_file_operations,\
nemotron_easy_scientific_computing,\
nemotron_easy_security,\
nemotron_easy_software_engineering,\
nemotron_medium_data_processing,\
nemotron_medium_data_querying,\
nemotron_medium_data_science,\
nemotron_medium_debugging,\
nemotron_medium_dependency_management,\
nemotron_medium_file_operations,\
nemotron_medium_model_training,\
nemotron_medium_scientific_computing,\
nemotron_medium_security,\
nemotron_medium_software_engineering,\
nemotron_medium_system_administration,\
nemotron_mixed_data_processing,\
nemotron_mixed_data_science,\
nemotron_mixed_debugging,\
nemotron_mixed_file_operations,\
nemotron_mixed_scientific_computing,\
nemotron_mixed_security"

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
    --cutoff_len 8192 \
    --learning_rate 1e-05 \
    --num_train_epochs 5.0 \
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
    --output_dir /helios-storage/helios4-data/cuong/model/all \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --deepspeed examples/deepspeed/ds_z3_config.json
