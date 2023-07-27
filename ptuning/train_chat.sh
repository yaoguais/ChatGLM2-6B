#!/bin/bash

python main.py \
    --do_train \
    --train_file minSamples/train.json \
    --validation_file minSamples/dev.json \
    --preprocessing_num_workers 10 \
    --prompt_column prompt \
    --response_column response \
    --history_column history \
    --source_prefix "我希望你扮演一个男性虚拟角色西奥多。我会和你进行聊天，请直接回复我，下面是我说的话。[[对话开始]]" \
    --overwrite_cache \
    --model_name_or_path THUDM/chatglm2-6b \
    --output_dir output/minsamples-chatglm2-6b-pt-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 256 \
    --max_target_length 256 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 100 \
    --logging_steps 10 \
    --save_steps 100 \
    --learning_rate 2e-2 \
    --pre_seq_len 128 \
    --quantization_bit 4

