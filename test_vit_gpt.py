import os
from tqdm import tqdm
from models.vit_gpt import GPT2Model, GPT2LMHeadModel
import torch
from data.coco_dataset import COCOCLIPDataset, COCOCLIPDatasetITuning
from utils.task_utils import update_config
from omegaconf import OmegaConf 
import argparse
import json

from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import ViTModel, AutoImageProcessor, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup, GPT2Config
from collections import OrderedDict
import random
import numpy as np

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v 
        if v.lower() in  ("yes", "true", "t", "y", "1"):
            return True 
        if v.lower() in ("no", "false", "f", "n", "0"):
            return False 
        else:
            raise argparse.ArgumentTypeError("Boolean value expected")
    
    parser = argparse.ArgumentParser(**parser_kwargs)

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=684331,
        help="seed for seed_everything",
    )

    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list()
    )
    
    parser.add_argument(
        "-w",
        "--num_workers",
        type=int,
        default=16,
        help="num of workers for data loader"
    )

    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=None,
        help="batch size for training"
    )

    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=None, 
        help="batch size for validataion and test" 
    )

    ## args for inference -- sentence generation
    parser.add_argument(
        "--use_beam_search",
        action="store_true"
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=5
    )

    parser.add_argument(
        "--gpt_type",
        type=str,
        default="gpt2"
    )

    parser.add_argument(
        "--vit_type",
        type=str,
        default="google/vit-base-patch16-224"
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="."
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=100,
    )

    parser.add_argument("--model_name", type=str, default="vit_gpt_routing_best_val")

    parser.add_argument("--no_vis_prefix", action="store_true")
    parser.add_argument("--text_prefix", type=str, default="<s>")
    parser.add_argument("--multiply_ones", action="store_true")

    parser.add_argument("--vit_use_pooler", action="store_true")

    parser.add_argument("--add_relu", action="store_true")


    return parser 


def test_epoch(test_dataloader, tokenizer, lm_model, vis_model, device, multiply_ones):
    lm_model = lm_model.module.module
    lm_model.eval()
    output_dict = {"pred_sent":[]}
    for idx, batch in enumerate(tqdm(test_dataloader, desc="Iteration")):
        tokens, mask, img = batch["encoder_input_ids"],  batch["encoder_padding_mask"], batch["context_inputs"]
        tokens = tokens.to(device)
        img = img.to(device)
        mask = mask.to(device)

        if opt.vit_use_pooler:
            img_feat = vis_model(img)["pooler_output"]
            img_feat = img_feat.unsqueeze(1)
        else:
            img_feat = vis_model(img)["last_hidden_state"]


        if opt.no_vis_prefix:
            text_prefix = [opt.text_prefix] * len(img)
            input_ids = tokenizer.batch_encode_plus(text_prefix, return_tensors="pt")["input_ids"]
            input_ids = input_ids.to(device)

            if "ituning" in opt.model_name:
                outputs = lm_model.generate(input_ids=input_ids, vision_hidden_states=img_feat, num_beams=5)
            else:
                outputs = lm_model.generate(input_ids=input_ids, vision_hidden_states=img_feat, num_beams=5, multiply_ones=multiply_ones)
            
        else:
            if "ituning" in opt.model_name:
                outputs = lm_model.generate(inputs_embeds=img_feat, vision_hidden_states=img_feat, num_beams=5)
            else:
                outputs = lm_model.gpt.generate(inputs_embeds=img_feat, vision_hidden_states=img_feat, num_beams=5, multiply_ones=multiply_ones)
        gen_text = tokenizer.batch_decode(outputs)
        gen_text = [cap.replace("<|endoftext|>", "") for cap in gen_text]
        gen_text = [cap.replace("\n", "") for cap in gen_text]
        if opt.no_vis_prefix:
            gen_text = [cap.replace("<s>", "") for cap in gen_text]
        for i in range(len(gen_text)):
            output_dict["pred_sent"].extend([{
                    "image_id": batch["image_id"][i].item(),
                    "id": batch["sample_id"][i].item(),
                    "caption": gen_text[i]
                }])
    
    return output_dict
        


def set_random_seed(seed: int):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999
    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """
    print("Seed: {}".format(seed))

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



if __name__ == "__main__":
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    configs = [OmegaConf.load(cfg) for cfg in opt.base]

    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    _config = update_config(opt=opt, config=config, ignore_args=[])

    set_random_seed(int(opt.seed))

    local_rank = int(os.environ['LOCAL_RANK'])
    torch.distributed.init_process_group(backend="nccl")
    torch.distributed.barrier()
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
    DEVICE = torch.device("cuda", local_rank)

    # lm_model = GPT2LMHeadModel.from_pretrained(_config.model.params.gpt_type, lora_config=_config.lora.params)

    tokenizer = GPT2Tokenizer.from_pretrained(_config.model.params.gpt_type)
    vis_model = ViTModel.from_pretrained(_config.model.params.vit_type)
    img_processor = AutoImageProcessor.from_pretrained(_config.model.params.vit_type)


    test_config = _config.task.test_data_config.params
    if opt.no_vis_prefix:
        test_data = COCOCLIPDatasetITuning(test_config.split, tokenizer, test_config.data_root, test_config.version, test_config.max_len, prefix_length=test_config.prefix_length, img_version = None, img_split = test_config.img_split, transform=img_processor)
    else:
        test_data = COCOCLIPDataset(test_config.split, tokenizer, test_config.data_root, test_config.version, test_config.max_len, prefix_length=test_config.prefix_length, img_version = None, img_split = test_config.img_split, transform=img_processor)
    test_loader = DataLoader(test_data, opt.val_batch_size, num_workers=opt.num_workers, collate_fn=test_data.collate)

    # lm_config = GPT2Config.from_pretrained(_config.model.params.gpt_type)
    # lm_model = GPT2LMHeadModel(lm_config, lora_config=_config.lora.params)
    # lm_state_dict = torch.load(os.path.join(opt.out_dir, opt.model_name+".pt"), map_location="cpu")
    # lm_state_dict = OrderedDict((k.replace("module.", "") if k.startswith("module") else k, v) for k, v in lm_state_dict.items())

    # lm_model.load_state_dict(lm_state_dict)
    lm_model = torch.load(os.path.join(opt.out_dir, opt.model_name+".pt"), map_location="cpu")
    lm_model = lm_model.to(DEVICE)
    lm_model = torch.nn.parallel.DistributedDataParallel(lm_model, device_ids=[local_rank], output_device=[local_rank])

    vis_model = vis_model.to(DEVICE)

    output_dict = test_epoch(test_loader, tokenizer, lm_model, vis_model, DEVICE, opt.multiply_ones)

    with open(os.path.join(opt.out_dir, opt.model_name+".json"), "w") as f:
        json.dump(output_dict, f)


