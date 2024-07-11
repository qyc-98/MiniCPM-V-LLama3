#!/bin/bash

GPUS_PER_NODE=1
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6002

MODEL="openbmb/MiniCPM-Llama3-V-2_5" # or openbmb/MiniCPM-V-2
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
DATA="path/to/train/data"
EVAL_DATA="path/to/test/data"
LLM_TYPE="llama3" # if use openbmb/MiniCPM-V-2, please set LLM_TYPE=minicpm

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
--dataloader_num_workers 0 \
--tune_vision true \
--tune_llm false \
--model_max_length 2048 \
--max_slice_nums 9 \
--max_steps 80000 \
--eval_steps 500 \
--output_dir output/output_minicpmv3 \
--logging_dir output/output_minicpmv3 \
--logging_strategy "steps" \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "steps" \
--save_strategy "steps" \
--save_steps 500 \
--save_total_limit 10 \
--learning_rate 1e-6 \
--weight_decay 0.1 \
--adam_beta2 0.95 \
--warmup_ratio 0.01 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--gradient_checkpointing true \
--report_to "tensorboard" \
--deepspeed ds_config_zero2.json \

