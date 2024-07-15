import os
import random
import numpy as np
import json
from tqdm import tqdm
from models.vit_roberta import RobertaVQAVLRouting, RobertaVQALoRA, RobertaVQAVLRoutingVisPre, count_parameters
import torch
from data.vqa_data import VQAFineTuneDataset, VQADataset, VQAEvaluator
from utils.task_utils import update_config
from omegaconf import OmegaConf 
import argparse

from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import ViTModel, AutoImageProcessor, RobertaTokenizer, AdamW, get_linear_schedule_with_warmup, BeitModel

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
        default="efficient_vl",
        help="project of expriment (used in wandb)"
    )

    parser.add_argument(
        "-n",
        "--experiment_name",
        type=str,
        default="lora_vit_roberta",
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

    parser.add_argument(
        "--offline_wandb",
        action="store_true"
    )

    ## args for inference -- sentence generation
    parser.add_argument(
        "--use_beam_search",
        action="store_true"
    )
    parser.add_argument("--beam_size", type=int, default=5)

    parser.add_argument(
        "--roberta_type",
        type=str,
        default="roberta-base"
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

    parser.add_argument("--vqa_dir", type=str, default="../VQAv2")
    parser.add_argument("--image_dir", type=str, default="../MSCOCO/Images")

    parser.add_argument("--max_length", type=int, default=20)


    parser.add_argument("--output_prefix", type=str, default="vit_roberta_lora")

    parser.add_argument("--use_classifier",action="store_true")
    parser.add_argument("--classifier_2layer",action="store_true")
    parser.add_argument("--use_vis_prefix",action="store_true")
    parser.add_argument("--vit_use_pooler",action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)



    parser.add_argument("--use_routing",action="store_true")
    parser.add_argument("--lora_split",action="store_true")
    parser.add_argument("--add_relu",action="store_true")

    parser.add_argument("--element_mul", action="store_true")
    parser.add_argument("--element_add", action="store_true")
    parser.add_argument("--element_mul_expand", action="store_true")

    # to be del
    parser.add_argument("--add_res", action="store_true")

    parser.add_argument("--freeze_ln", action="store_true")
    
    parser.add_argument("--use_noise_vis", action="store_true")
    parser.add_argument("--use_ones_vis", action="store_true")

    parser.add_argument("--r_lora", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--num_heads_lora", type=int, default=4)
    parser.add_argument("--split_type", type=str, default="split")
    parser.add_argument("--num_labels", type=int, default=3129)

    parser.add_argument("--train_split", type=str, default="train,nominival")
    parser.add_argument("--val_split", type=str, default="minival")
    parser.add_argument("--test_split", type=str, default="test")

    parser.add_argument("--use_adapter", action="store_true")


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

    optimizer = AdamW(lm_model.parameters(), lr=opt.lr)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=_config.run.params.warmup_steps * _config.run.params.epoch * train_size, num_training_steps=_config.run.params.epoch * train_size
    )
    
    return lm_model, vis_model, optimizer, scheduler


def train_epoch(train_dataloader, epoch, lm_model, vis_model, optimizer, scheduler, output_dir, device, output_prefix):
    lm_model.train()
    train_loss = 0
    step = 0
    bce_loss = torch.nn.BCEWithLogitsLoss()
    # img_mask = torch.ones(opt.train_batch_size, 1)
    for idx, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        lm_model.zero_grad()
        tokens, mask, img, target = batch["input_ids"], batch["padding_mask"], batch["img_tensors"], batch["target"]
        tokens = tokens.to(device)
        img = img.to(device)
        mask = mask.to(device)
        target = target.to(device)
        # img_mask = img_mask.to(mask.device)
        # mask = torch.cat((img_mask, mask), dim=1)

        if opt.vit_use_pooler:
            img_feat = vis_model(img)["pooler_output"]
            img_feat = img_feat.unsqueeze(1)
        else:
            img_feat = vis_model(img)["last_hidden_state"]


        outputs = lm_model(input_ids=tokens, attention_mask=mask, vision_hidden_states=img_feat)

        logits = outputs.logits.contiguous()
        loss = bce_loss(logits, target)

        train_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        step += 1
        
        wandb.log({"loss": loss})

        if (idx + epoch*len(train_dataloader)) % 500 == 0:
            torch.save(lm_model, os.path.join(output_dir, f"{output_prefix}_latest.pt"),)
    return train_loss/step


def val_epoch(val_dataloader, epoch, lm_model, vis_model, device, dset=None, evaluator=None):
    lm_model.eval()
    step = 0
    val_loss = 0
    # img_mask = torch.ones(opt.val_batch_size, 1)
    bce_loss = torch.nn.BCEWithLogitsLoss()
    
    quesid2ans = {}
    
    for idx, batch in enumerate(tqdm(val_dataloader, desc="Iteration")):
        lm_model.zero_grad()
        tokens, mask, img, target = batch["input_ids"], batch["padding_mask"], batch["img_tensors"], batch["target"]
        ques_id = batch["question_ids"]
        tokens = tokens.to(device)
        img = img.to(device)
        mask = mask.to(device)
        target = target.to(device)

        if opt.vit_use_pooler:
            img_feat = vis_model(img)["pooler_output"]
            img_feat = img_feat.unsqueeze(1)
        else:
            img_feat = vis_model(img)["last_hidden_state"]

        outputs = lm_model(input_ids=tokens, attention_mask=mask, vision_hidden_states=img_feat)
        logits = outputs.logits.contiguous()
        loss = bce_loss(logits, target)

        val_loss += loss.item()
        step += 1

        score, label = logits.max(1)
        for qid, l in zip(ques_id, label.cpu().numpy()):
            ans = dset.label2ans[l]
            quesid2ans[qid] = ans
        wandb.log({"val loss": loss})
    
    val_acc = evaluator.evaluate(quesid2ans)
    wandb.log({"val acc": val_acc})
    return val_loss / step, val_acc


def train(_config, train_dataloader, val_dataloader, lm_model, vis_model, optimizer, scheduler, output_dir, device, output_prefix, dset_val=None, evaluator=None):
    epochs = _config.run.params.epoch
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    lm_model = lm_model.to(device)

    wandb.watch(lm_model)

    min_val_loss = 999
    max_val_acc = 0
    for epoch in range(epochs):
        run.log({"Epochs" : epoch})

        loss = train_epoch(train_dataloader, epoch, lm_model, vis_model, optimizer, scheduler, output_dir, device, output_prefix)
        
        with torch.no_grad():
            val_loss, val_acc  = val_epoch(val_dataloader, epoch, lm_model, vis_model, device, dset=dset_val, evaluator=evaluator)
        if epoch % _config.run.params.save_every == 0 or epoch == epochs - 1:
            torch.save(
                lm_model,
                os.path.join(output_dir, f"{output_prefix}-{epoch:03d}.pt"),
            )
        if val_loss < min_val_loss:
            torch.save(lm_model, os.path.join(output_dir, f"{output_prefix}_best_val.pt"),)
            min_val_loss = val_loss

            wandb.log({"min val loss": min_val_loss})
        if val_acc > max_val_acc:
            torch.save(lm_model, os.path.join(output_dir, f"{output_prefix}_best_val_acc.pt"),)
            max_val_acc = val_acc

            wandb.log({"max val acc": max_val_acc})
            
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

    output_prefix = f"{opt.seed}_" + opt.output_prefix + f"_r{opt.r_lora}_alpha{opt.lora_alpha}_lr{opt.lr}"
    if opt.use_routing:
        output_prefix = output_prefix + "_routing"
    else:
        output_prefix = output_prefix + "_lora"
    if not opt.use_vis_prefix:
        output_prefix = output_prefix + "_no_vis_pre"

    if opt.vit_use_pooler:
        output_prefix = output_prefix + "_pool"
    if opt.lora_split:
        output_prefix = output_prefix + "_split"
    if opt.classifier_2layer:
        output_prefix = output_prefix + "_classifier2L"
    if opt.add_res:
        output_prefix = output_prefix + "_res"
    if not opt.freeze_ln:
        output_prefix = output_prefix + "_tuneLN"
    if opt.use_noise_vis:
        output_prefix = output_prefix + "_noisevis"
    if opt.use_ones_vis:
        output_prefix = output_prefix + "_onesvis"
    if opt.add_relu:
        output_prefix = output_prefix + "_relu"

    print(f"experiment:{output_prefix}")
    run = wandb.init(
        project=_config.run.params.project_name,
        name=output_prefix,
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

    if _config.lora.params.use_routing and not opt.use_vis_prefix:
        lm_model = RobertaVQAVLRouting.from_pretrained(_config.model.params.roberta_type, lora_config=_config.lora.params, use_adapter=opt.use_adapter, classifier_2layer=opt.classifier_2layer)
    elif _config.lora.params.use_routing and opt.use_vis_prefix:
        # to be merged
        lm_model = RobertaVQAVLRoutingVisPre.from_pretrained(_config.model.params.roberta_type, lora_config=_config.lora.params, use_adapter=opt.use_adapter, classifier_2layer=opt.classifier_2layer)
    else:
        lm_model = RobertaVQALoRA.from_pretrained(_config.model.params.roberta_type, lora_config=_config.lora.params, use_adapter=opt.use_adapter,classifier_2layer=opt.classifier_2layer)

    tokenizer = RobertaTokenizer.from_pretrained(_config.model.params.roberta_type)

    if "vit" in _config.model.params.vit_type:
        vis_model = ViTModel.from_pretrained(_config.model.params.vit_type)
    elif "beit" in _config.model.params.vit_type:
        vis_model = BeitModel.from_pretrained(_config.model.params.vit_type)
    else:
        print("wrong model type for image encoder")
    for params in vis_model.parameters():
        params.requires_grad = False
    
    img_processor = AutoImageProcessor.from_pretrained(_config.model.params.vit_type)

    run.log({"lm trainable param" : count_parameters(lm_model)})
    run.log({"classifier trainable param" : count_parameters(lm_model.classifier)})
    run.log({"vis trainable param" : count_parameters(vis_model)})

    train_config = _config.task.train_data_config.params
    train_data_raw = VQADataset(train_config.train_split, verbose=train_config.verbose)
    train_data = VQAFineTuneDataset(tokenizer, split=train_config.train_split, raw_dataset=train_data_raw, rank=train_config.rank, topk=train_config.topk, verbose=train_config.verbose, args=None, mode=train_config.mode, max_length=train_config.max_length, use_vis_prefix=train_config.use_vis_prefix, prefix_length=train_config.prefix_length, image_transforms=img_processor, vqa_dir=train_config.vqa_dir, image_dir=train_config.image_dir, use_classifier=train_config.use_classifier)
    train_sampler = DistributedSampler(dataset=train_data, rank=local_rank, shuffle=True)
    train_loader = DataLoader(train_data, opt.train_batch_size, num_workers=opt.num_workers, collate_fn=train_data.collate, sampler=train_sampler)


    val_config = _config.task.val_data_config.params
    val_data_raw = VQADataset(val_config.val_split, verbose=val_config.verbose)
    val_data = VQAFineTuneDataset(tokenizer, split=val_config.val_split, raw_dataset=val_data_raw, rank=val_config.rank, topk=val_config.topk, verbose=val_config.verbose, args=None, mode=val_config.mode, max_length=val_config.max_length, use_vis_prefix=val_config.use_vis_prefix, prefix_length=val_config.prefix_length, image_transforms=img_processor, vqa_dir=val_config.vqa_dir, image_dir=val_config.image_dir, use_classifier=val_config.use_classifier)
    val_sampler = DistributedSampler(dataset=val_data, rank=local_rank, shuffle=False)
    val_loader = DataLoader(val_data, opt.val_batch_size, num_workers=opt.num_workers, collate_fn=val_data.collate, sampler=val_sampler)

    val_evaluator = VQAEvaluator(val_data_raw)

    lm_model, vis_model, optimizer, scheduler = prep_for_training(lm_model, vis_model, _config, len(train_loader), DEVICE)
    
    if "vit" in _config.model.params.vit_type:
        output_dir = os.path.join(_config.run.params.out_dir, _config.model.params.roberta_type, _config.model.params.vit_type.replace("google/", ""))
    elif "beit" in _config.model.params.vit_type:
        output_dir = os.path.join(_config.run.params.out_dir, _config.model.params.roberta_type, _config.model.params.vit_type.replace("microsoft/", ""))
    else:
        print("wrong model type for vit")

    train(_config, train_loader, val_loader, lm_model, vis_model, optimizer, scheduler, output_dir, DEVICE, output_prefix, dset_val=val_data_raw, evaluator=val_evaluator)
