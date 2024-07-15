import os
import json
import torch
from tqdm import tqdm
from data.vqa_data import VQADataset, VQAEvaluator, VQAFineTuneDataset
from utils.task_utils import update_config
from omegaconf import OmegaConf 
import argparse
from torch.utils.data import DataLoader
from transformers import ViTModel, AutoImageProcessor, RobertaTokenizer


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

    parser.add_argument(
        "--offline_wandb",
        action="store_true"
    )

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

    parser.add_argument("--vqa_dir", type=str, default="/cw/liir_data/NoCsBack/VQAv2")
    parser.add_argument("--image_dir", type=str, default="/cw/liir_data/NoCsBack/MSCOCO/Images")

    parser.add_argument("--max_length", type=int, default=20)


    parser.add_argument("--output_prefix", type=str, default="vit_roberta_lora")

    parser.add_argument("--use_classifier",action="store_true")
    parser.add_argument("--use_vis_prefix",action="store_true")
    parser.add_argument("--vit_use_pooler",action="store_true")


    parser.add_argument("--use_routing",action="store_true")
    parser.add_argument("--lora_split",action="store_true")
    parser.add_argument("--add_relu",action="store_true")

    parser.add_argument("--r_lora", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--num_heads_lora", type=int, default=4)
    parser.add_argument("--split_type", type=str, default="split")
    parser.add_argument("--num_labels", type=int, default=3129)

    parser.add_argument("--model_name", type=str, default="vitL_robertaL_r128_routing_no_vis_pre_pool_split-010")

    parser.add_argument("--val_split", type=str, default="minival", help="split for the data, not the lora")

    return parser 


def predict(model, vis_model, dataloader, evaluator, dset, dump=None, device="cuda"):
    """
    Predict the answers to questions in a data split.

    :param eval_tuple: The data tuple to be evaluated.
    :param dump: The path of saved file to dump results.
    :return: A dict of question_id to answer.
    """
    model.eval()
    quesid2ans = {}
    for i, batch in enumerate(tqdm(dataloader)):
        ques_id, img, tokens, mask = batch["question_ids"], batch["img_tensors"], batch["input_ids"], batch["padding_mask"]
        with torch.no_grad():
            img = img.to(device)
            tokens = tokens.to(device)
            mask = mask.to(device)

            if opt.vit_use_pooler:
                img_feat = vis_model(img)["pooler_output"]
                img_feat = img_feat.unsqueeze(1)
            else:
                img_feat = vis_model(img)["last_hidden_state"]
            outputs = model(input_ids=tokens, attention_mask=mask, vision_hidden_states=img_feat)
            logit = outputs.logits

            score, label = logit.max(1)
            for qid, l in zip(ques_id, label.cpu().numpy()):
                ans = dset.label2ans[l]
                quesid2ans[qid] = ans
    print(evaluator.evaluate(quesid2ans))
    if dump is not None:
        evaluator.dump_result(quesid2ans, dump)
    return quesid2ans

# def evaluate_json(evaluator, json_dir):
#     with open(json_dir) as f:
#         quesid2ans = json.load(f)
#     result_dict = {}
#     for sample in quesid2ans:
#         result_dict.update(sample)
#         import ipdb
#         ipdb.set_trace()
#     print(evaluator.evaluate(result_dict))



if __name__ == "__main__":
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    configs = [OmegaConf.load(cfg) for cfg in opt.base]

    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    _config = update_config(opt=opt, config=config, ignore_args=[])

    local_rank = int(os.environ['LOCAL_RANK'])
    torch.distributed.init_process_group(backend="nccl")
    torch.distributed.barrier()
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
    DEVICE = torch.device("cuda", local_rank)


    tokenizer = RobertaTokenizer.from_pretrained(_config.model.params.roberta_type)
    img_processor = AutoImageProcessor.from_pretrained(_config.model.params.vit_type)

    val_config = _config.task.val_data_config.params
    val_data_raw = VQADataset(val_config.val_split, verbose=val_config.verbose)
    val_data = VQAFineTuneDataset(tokenizer, split=val_config.val_split, raw_dataset=val_data_raw, rank=val_config.rank, topk=val_config.topk, verbose=val_config.verbose, args=None, mode=val_config.mode, max_length=val_config.max_length, use_vis_prefix=val_config.use_vis_prefix, prefix_length=val_config.prefix_length, image_transforms=img_processor, vqa_dir=val_config.vqa_dir, image_dir=val_config.image_dir, use_classifier=val_config.use_classifier)

    val_loader = DataLoader(val_data, opt.val_batch_size, num_workers=opt.num_workers, collate_fn=val_data.collate,)

    val_evaluator = VQAEvaluator(val_data_raw)


    vis_model = ViTModel.from_pretrained(_config.model.params.vit_type)
    vis_model = vis_model.cuda()
    vis_model = vis_model.eval()

    model_dir = os.path.join(opt.out_dir, opt.roberta_type, opt.vit_type.replace("google/", ""))
    lm_model = torch.load(os.path.join(model_dir, opt.model_name+".pt"), map_location="cpu")
    lm_model = lm_model.to(DEVICE)
    lm_model = torch.nn.parallel.DistributedDataParallel(lm_model, device_ids=[local_rank], output_device=[local_rank])
    

    result_dir = os.path.join(model_dir, opt.model_name+f"{opt.val_split}.json")
    predict(lm_model, vis_model, val_loader, val_evaluator, val_data_raw, dump=result_dir, device="cuda")
    print(opt.model_name)
    # evaluate_json(val_evaluator, result_dir)


# def evaluate(self, eval_tuple: DataTuple, dump=None):
#     """Evaluate all data in data_tuple."""
#     quesid2ans = self.predict(eval_tuple, dump)
#     return eval_tuple.evaluator.evaluate(quesid2ans)