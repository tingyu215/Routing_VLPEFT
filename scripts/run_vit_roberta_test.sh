#!/bin/sh
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port 29502 ../test_vit_roberta.py \
--base configs/vqa_task.yaml configs/lora_vl_vqa.yaml configs/vit_roberta_task.yaml \
--out_dir OUTPUT_DIR \
--train_batch_size 512 \
--val_batch_size 512 \
--roberta_type roberta-base \
--vit_type google/vit-base-patch16-224-in21k \
--model_name 3407_vitB_robertaB_elemulexpand_karpathy_r128_alpha128_lr0.0003_vllora_pool_classifier2L-019 \
--project_name efficient_vl_vqa \
--r_lora 128 --lora_alpha 128 \
--use_classifier \
--vit_use_pooler \
--use_vllora \
--val_split karpathy_test \
--use_vis_prefix

# --lora_split \


# --num_heads_lora 4 \

# --multiply_ones

# --vit_use_pooler
# split_type: split, merge, merge_res, split_attn