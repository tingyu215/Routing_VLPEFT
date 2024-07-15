#!/bin/sh
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port 29507 ../test_vit_gpt.py \
--base configs/coco_task.yaml configs/lora_vl_splitA.yaml configs/vit_gpt_task.yaml \
--out_dir OUTPUT_DIR/gpt2/vit-base-patch16-224-in21k \
--model_name 9898_routing_pool_gpt_vitB_beforeB_r128_drop0_eleadd-019 \
--gpt_type gpt2 \
--vit_type google/vit-base-patch16-224-in21k \
--val_batch_size 40 \
--vit_use_pooler
# --no_vis_prefix \
#  --add_relu
# --multiply_ones \

# --base configs/coco_task.yaml configs/ituning.yaml configs/vit_gpt_task_large.yaml \


