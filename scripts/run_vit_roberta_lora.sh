#!/bin/sh
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port 28003 ../train_vit_roberta.py \
--base configs/vqa_task.yaml configs/lora_vl_vqa.yaml configs/vit_roberta_task.yaml \
--out_dir OUTPUT_DIR \
--train_batch_size 512 \
--val_batch_size 512 \
--roberta_type roberta-base \
--vit_type google/vit-base-patch16-224-in21k \
--project_name routingVL \
--experiment_name vitB_robertaB_eleadd_karpathy \
--output_prefix vitB_robertaB_eleadd_karpathy \
--split_type split \
--r_lora 128 --lora_alpha 128 \
--lr 3e-4 \
--use_classifier \
--train_split karpathy_train --val_split karpathy_val \
--vit_use_pooler \
--use_vis_prefix \
--use_vllora \
--element_add \
--epoch 20 \
--classifier_2layer \
--freeze_ln \
--seed 18

# --lora_split \

# --offline_wandb

# --num_heads_lora 4 \

# --multiply_ones

# --vit_use_pooler
# split_type: split, merge, merge_res, split_attn