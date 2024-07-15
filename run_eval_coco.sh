#!/bin/sh
CUDA_VISIBLE_DEVICES=0 python eval_coco.py --out_dir OUTPUT_DIR/gpt2/vit-base-patch16-224-in21k --out_dict_name 9898_routing_pool_gpt_vitB_beforeB_r128_drop0_eleadd-019
