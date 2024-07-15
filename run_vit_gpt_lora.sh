#!/bin/sh
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port 26113 train_vit_gpt.py \
--base configs/coco_task.yaml configs/lora_vl.yaml configs/vit_gpt_task.yaml \
--out_dir OUTPUT_DIR \
--train_batch_size 40 \
--val_batch_size 40 \
--gpt_type gpt2 \
--vit_type google/vit-base-patch16-224-in21k \
--project_name routingVL \
--experiment_name routing_pool_gpt_vitB_beforeB_r128_drop0_eleadd \
--output_prefix routing_pool_gpt_vitB_beforeB_r128_drop0_eleadd \
--lora_only_vl \
--merge_type beforeB \
--fusion_layer 0 1 2 3 4 5 6 7 8 9 10 11 \
--r_lora 128 --lora_alpha 128 \
--num_heads_lora 4 \
--lora_dropout 0 \
--epoch 20 \
--element_add \
--vit_use_pooler \
--seed 9898

# --all_lora \
# --no_vis_prefix \

# --offline_wandb


# --multiply_ones
# --add_relu


