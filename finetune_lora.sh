#!/bin/bash

GPUS_PER_NODE=1
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001

MODEL="openbmb/MiniCPM-Llama3-V-2_5"
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
DATA="/data/data/guiact/processed/training_data.json"
EVAL_DATA="/data/data/guiact/processed/test_data.json"
LLM_TYPE="llama3" 
DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS finetune.py  \
--model_name_or_path $MODEL \
--llm_type $LLM_TYPE \
--data_path $DATA \
--eval_data_path $EVAL_DATA \
--remove_unused_columns false \
--label_names "labels" \
--prediction_loss_only false \
--bf16 true \
--bf16_full_eval true \
--fp16 false \
--fp16_full_eval false \
--do_train \
--do_eval \
--dataloader_pin_memory false \
--dataloader_num_workers 1 \
--tune_vision true \
--tune_llm false \
--use_lora true \
--q_lora false \
--lora_target_modules 'llm\.model\.layers\.\d+\.self_attn\.(q_proj|v_proj|k_proj|o_proj)' \
--model_max_length 2048 \
--max_slice_nums 9 \
--max_steps 80000 \
--eval_steps 1000 \
--output_dir output/output_minicpmv3_lora_gui \
--logging_dir output/output_minicpmv3_lora_gui \
--logging_strategy "steps" \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 2 \
--evaluation_strategy "steps" \
--save_strategy "steps" \
--save_steps 1000 \
--save_total_limit 10 \
--learning_rate 1e-6 \
--weight_decay 0.1 \
--adam_beta2 0.95 \
--warmup_ratio 0.01 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--gradient_checkpointing true \
--deepspeed ds_config_zero2.json \
--report_to "tensorboard" \

