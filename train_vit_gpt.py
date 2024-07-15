import os
import random
import numpy as np
from tqdm import tqdm
from models.vit_gpt import GPT2Model, GPT2LMHeadModel, ViTGPTLoRa, count_parameters, count_total_parameters
import torch
from data.coco_dataset import COCOCLIPDataset, COCOCLIPDatasetITuning
from utils.task_utils import update_config
from omegaconf import OmegaConf 
import argparse

from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import ViTModel, AutoImageProcessor, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup

import wandb

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
        "-p",
        "--project_name",
        type=str,
        default="efficient_vl_freeze_head",
        help="project of expriment (used in wandb)"
    )

    parser.add_argument(
        "-n",
        "--experiment_name",
        type=str,
        default="routing_vit_gpt",
        help="the name of the experiment"
    )

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
    
    parser.add_argument("-w", "--num_workers", type=int, default=16, help="num of workers for data loader")

    parser.add_argument("--train_batch_size",type=int, default=None, help="batch size for training")

    parser.add_argument("--val_batch_size",type=int,default=None, help="batch size for validataion and test" )

    parser.add_argument("--offline_wandb", action="store_true")

    parser.add_argument("--use_beam_search",action="store_true")
    parser.add_argument("--beam_size", type=int, default=5)

    parser.add_argument("--gpt_type", type=str, default="gpt2")

    parser.add_argument("--vit_type", type=str, default="google/vit-base-patch16-224")

    parser.add_argument("--out_dir", type=str, default=".")

    parser.add_argument("--epoch", type=int, default=100)

    parser.add_argument("--no_vis_prefix", action="store_true")

    parser.add_argument("--output_prefix", type=str, default="routing_vit_gpt")

    parser.add_argument("--multiply_ones", action="store_true")

    parser.add_argument("--lora_only_vl", action="store_true")

    parser.add_argument("--merge_type", type=str, default="afterB")

    parser.add_argument("--vit_use_pooler", action="store_true")

    parser.add_argument("--fusion_layer", nargs="+", type=int)
    
    parser.add_argument("--all_lora", action="store_true")

    parser.add_argument("--add_relu", action="store_true")

    parser.add_argument("--r_lora", type=int, default=128,)
    parser.add_argument("--lora_alpha", type=int, default=128,)

    parser.add_argument("--lora_dropout", type=float, default=0.1,)

    parser.add_argument("--element_mul", action="store_true")
    parser.add_argument("--element_add", action="store_true")
    parser.add_argument("--element_mul_expand", action="store_true")

    parser.add_argument("--use_adapter", action="store_true")
    parser.add_argument("--freeze_ln", action="store_true")

    parser.add_argument("--use_routing", action="store_true", help="used for adapter, to control whether use vis info or not")

    parser.add_argument("--adapt_pos", type=str, default="post_module")

    return parser 



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


def prep_for_training(lm_model, vis_model, _config, train_size, DEVICE):
    lm_model = torch.nn.parallel.DistributedDataParallel(lm_model.cuda(), device_ids=[local_rank], output_device=local_rank)
    lm_model.to(DEVICE)

    vis_model.to(DEVICE)

    optimizer = AdamW(lm_model.parameters(), lr=_config.run.params.lr)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=_config.run.params.warmup_steps, num_training_steps=_config.run.params.epoch * train_size
    )
    
    return lm_model, vis_model, optimizer, scheduler


def train_epoch(train_dataloader, epoch, lm_model, vis_model, optimizer, scheduler, output_dir, device, text_table, output_prefix, multiply_ones):
    lm_model.train()
    train_loss = 0
    step = 0
    for idx, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        lm_model.zero_grad()
        tokens, mask, img = batch["encoder_input_ids"], batch["encoder_padding_mask"], batch["context_inputs"]
        tokens = tokens.to(device)
        img = img.to(device)
        mask = mask.to(device)

        if opt.vit_use_pooler:
            img_feat = vis_model(img)["pooler_output"]
            img_feat = img_feat.unsqueeze(1)
        else:
            img_feat = vis_model(img)["last_hidden_state"]

        if opt.no_vis_prefix:
            outputs = lm_model(input_ids=tokens, attention_mask=mask, vision_hidden_states=img_feat, multiply_ones=multiply_ones)
            logits = outputs.logits[..., :-1, :].contiguous()
        else:
            outputs = lm_model(input_ids=tokens, mask=mask, vision_hidden_states=img_feat, multiply_ones=multiply_ones,)
            logits = outputs.logits[..., 1 :-1, :].contiguous()
        tokens = tokens[..., 1:].contiguous()
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        step += 1
        
        wandb.log({"loss": loss})

        if (idx + epoch*len(train_dataloader)) % 10000 == 0:
            torch.save(lm_model, os.path.join(output_dir, f"{output_prefix}_latest.pt"),)
            text_from_logits = [tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(torch.argmax(logits[i], dim=-1))) for i in range(logits.shape[0])]
            
            text_table.add_data(epoch, idx, train_loss/step, text_from_logits)
    return train_loss/step



def val_epoch(val_dataloader, epoch, lm_model, vis_model, device, val_text_table, multiply_ones):
    lm_model.eval()
    step = 0
    val_loss = 0
    for idx, batch in enumerate(tqdm(val_dataloader, desc="Iteration")):
        tokens, mask, img = batch["encoder_input_ids"], batch["encoder_padding_mask"], batch["context_inputs"]
        tokens = tokens.to(device)
        img = img.to(device)
        mask = mask.to(device)

        if opt.vit_use_pooler:
            img_feat = vis_model(img)["pooler_output"]
            img_feat = img_feat.unsqueeze(1)
        else:
            img_feat = vis_model(img)["last_hidden_state"]

        if opt.no_vis_prefix:
            outputs = lm_model(input_ids=tokens, attention_mask=mask, vision_hidden_states=img_feat, multiply_ones=multiply_ones,)
            logits = outputs.logits[..., :-1, :].contiguous()
        else:
            outputs = lm_model(input_ids=tokens, mask=mask, vision_hidden_states=img_feat, multiply_ones=multiply_ones,)
            logits = outputs.logits[..., 1 :-1, :].contiguous()
        tokens = tokens[..., 1:].contiguous()
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)

        val_loss += loss.item()
        step += 1
        
        wandb.log({"val loss": loss})
    text_from_logits = [tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(torch.argmax(logits[i], dim=-1))) for i in range(logits.shape[0])]
            
    val_text_table.add_data(epoch, val_loss / step, text_from_logits)
    return val_loss / step


def train(_config, train_dataloader, val_dataloader, lm_model, vis_model, optimizer, scheduler, output_dir, device, output_prefix, multiply_ones):
    epochs = _config.run.params.epoch
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    lm_model = lm_model.to(device)

    wandb.watch(lm_model)
    text_table = wandb.Table(columns=["epoch", "step", "loss", "text"])
    val_text_table = wandb.Table(columns=["val_epoch", "loss", "text"])

    min_val_loss = 999
    for epoch in range(epochs):
        wandb.log({"Epoch": epoch})
        loss = train_epoch(train_dataloader, epoch, lm_model, vis_model, optimizer, scheduler, output_dir, device, text_table, output_prefix, multiply_ones)
        with torch.no_grad():
            val_loss  = val_epoch(val_dataloader, epoch, lm_model, vis_model, device, val_text_table, multiply_ones)
        if epoch % _config.run.params.save_every == 0 or epoch == epochs - 1:
            torch.save(
                lm_model,
                os.path.join(output_dir, f"{output_prefix}-{epoch:03d}.pt"),
            )
        if val_loss < min_val_loss:
            torch.save(lm_model, os.path.join(output_dir, f"{output_prefix}_best_val.pt"),)
            min_val_loss = val_loss

            wandb.log({"min val loss": min_val_loss})
            
    run.log({"training_samples" : text_table})
    run.log({"validation_samples" : val_text_table})
    return lm_model


if __name__ == "__main__":
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    configs = [OmegaConf.load(cfg) for cfg in opt.base]

    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    _config = update_config(opt=opt, config=config, ignore_args=[])

    if opt.offline_wandb:
        os.environ["WANDB_MODE"] = "offline"
    
    set_random_seed(int(opt.seed))

    run = wandb.init(
        project=_config.run.params.project_name,
        name=f"{opt.seed}_" + _config.run.params.experiment_name + opt.adapt_pos,
    )

    wandb.config.update(_config)


    local_rank = int(os.environ['LOCAL_RANK'])
    print(local_rank)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    world_size = torch.cuda.device_count()
    print(f"world size: {world_size}")
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
    DEVICE = torch.device("cuda", local_rank)
    
    if opt.no_vis_prefix:
        lm_model = GPT2LMHeadModel.from_pretrained(_config.model.params.gpt_type, lora_config=_config.lora.params, lora_only_vl=opt.lora_only_vl, merge_type=opt.merge_type, fusion_layer=opt.fusion_layer, all_lora=opt.all_lora, use_adapter=opt.use_adapter, freeze_ln=opt.freeze_ln)
        print("no vis feat as prefix")
    else:
        lm_model = ViTGPTLoRa(_config.model.params.gpt_type, lora_config=_config.lora.params, lora_only_vl=opt.lora_only_vl, merge_type=opt.merge_type, fusion_layer=opt.fusion_layer, all_lora=opt.all_lora, use_adapter=opt.use_adapter, freeze_ln=opt.freeze_ln)

    tokenizer = GPT2Tokenizer.from_pretrained(_config.model.params.gpt_type)

    vis_model = ViTModel.from_pretrained(_config.model.params.vit_type)
    for params in vis_model.parameters():
        params.requires_grad = False

    run.log({"lm trainable param" : count_parameters(lm_model)})
    run.log({"vis trainable param" : count_parameters(vis_model)})

    run.log({"lm total param" : count_total_parameters(lm_model)})
    run.log({"vis total param" : count_total_parameters(vis_model)})
    
    img_processor = AutoImageProcessor.from_pretrained(_config.model.params.vit_type)

    train_config = _config.task.train_data_config.params
    if opt.no_vis_prefix:
        train_data = COCOCLIPDatasetITuning(train_config.split, tokenizer, train_config.data_root, train_config.version, train_config.max_len, prefix_length=train_config.prefix_length, img_version = None, img_split = None, transform=img_processor)
    else:
        train_data = COCOCLIPDataset(train_config.split, tokenizer, train_config.data_root, train_config.version, train_config.max_len, prefix_length=train_config.prefix_length, img_version = None, img_split = None, transform=img_processor)
    train_sampler = DistributedSampler(dataset=train_data, rank=local_rank, shuffle=True)
    train_loader = DataLoader(train_data, opt.train_batch_size, num_workers=opt.num_workers, collate_fn=train_data.collate, sampler=train_sampler)


    val_config = _config.task.val_data_config.params
    if opt.no_vis_prefix:
        val_data = COCOCLIPDatasetITuning(val_config.split, tokenizer, val_config.data_root, val_config.version, val_config.max_len, prefix_length=val_config.prefix_length, img_version = None, img_split = None, transform=img_processor)
    else:
        val_data = COCOCLIPDataset(val_config.split, tokenizer, val_config.data_root, val_config.version, val_config.max_len, prefix_length=val_config.prefix_length, img_version = None, img_split = None, transform=img_processor)
    val_sampler = DistributedSampler(dataset=val_data, rank=local_rank, shuffle=False)
    val_loader = DataLoader(val_data, opt.val_batch_size, num_workers=opt.num_workers, collate_fn=val_data.collate, sampler=val_sampler)


    lm_model, vis_model, optimizer, scheduler = prep_for_training(lm_model, vis_model, _config, len(train_loader), DEVICE)

    output_prefix = f"{opt.seed}_" + opt.output_prefix
    if opt.use_adapter:
        output_prefix = output_prefix + opt.adapt_pos
    
    print(f"freeze layer norm:{opt.freeze_ln}")
    if opt.use_adapter:
        print(f"adapter position:{opt.adapt_pos}")
    output_dir = os.path.join(_config.run.params.out_dir, _config.model.params.gpt_type, _config.model.params.vit_type.replace("google/", ""))

    train(_config, train_loader, val_loader, lm_model, vis_model, optimizer, scheduler, output_dir, DEVICE, output_prefix, opt.multiply_ones)
