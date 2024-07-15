import numpy as np
import json 
import types
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
# from inference.coco import COCO

from pycocoevalcap.eval import COCOEvalCap

from pycocoevalcap.cider.cider_scorer import CiderScorer
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
# from inference.utils import set_dict2list,to_coco_annotation_format
from inference.utils import set_dict2list,to_coco_annotation_format
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', default='/export/home1/NoCsBack/working/tingyu/efficient_vl_freeze_head')
parser.add_argument("--out_dict_name", type=str, default="vit_gpt_vllora_best_val")
args = parser.parse_args()
import os


import ipdb 
# default test on karpathy test split 
KARPATHY_SPLIT = "/cw/liir_data/NoCsBack/MSCOCO/annotations/karpathy_split_coco.json"
GT_PATH = "/cw/liir_data/NoCsBack/MSCOCO/annotations/captions_test_karpathy.json"
# GT_PATH = "/staging/leuven/stg_00095/mingxiao/data/MSCOCO/annotations/captions_test_karpathy.json"

def _stat(self, hypothesis_str, reference_list):
    # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
    hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
    score_line = ' ||| '.join(
        ('SCORE', ' ||| '.join(reference_list), hypothesis_str))
    score_line = score_line.replace('\n', '').replace('\r', '')
    self.meteor_p.stdin.write('{}\n'.format(score_line).encode())
    self.meteor_p.stdin.flush()
    return self.meteor_p.stdout.readline().decode().strip()

class LangGenerationEvluator(object):
    
    def __init__(self,
                 data = "coco", 
                 metrics=['bleu','cider','meteor','rouge','spice'],
                 gt_data_path=KARPATHY_SPLIT,
                 n_bleu=4,
                 n_cider=4,
                 cider_sigma=6.0,
                 pred_data_root=None):
        
        self._get_evaluation_metrics(metrics, n_bleu)
        self.gt_data = self._get_gt_data(gt_data_path,data) 
        self.metrics_score = {}
    
    def _get_gt_data(self, path, data_name):
        with open(path, "r") as f:
            data = json.load(f)
        if data_name == "coco":
            test_data = []
            for d in data["images"]:
                if d["split"] == "test":
                    for s in d["sentences"]:
                        test_data.append(
                            {
                                "image_id":s['imgid'],
                                "id": s['sentid'], 
                                "caption": s['raw']
                            }
                        )
        else:
            raise NotImplementedError(f"Do not support data {data_name} yet")
        return test_data
    
    def _get_evaluation_metrics(self, metrics, n_cider=4, cider_sigma=6.0, n_bleu=4):          
        for m in metrics:
            if m == "bleu":
                setattr(self, m, BleuScorer(n_bleu))
            elif m == "cider":
                setattr(self, m, CiderScorer(n=n_cider, sigma=cider_sigma))
            elif m == "meteor":
                setattr(self, m , Meteor())
            elif m == "rouge":
                setattr(self, m, Rouge())
            elif m == "spice":
                setattr(self, m, Spice())
                
    @staticmethod
    def eval_offline(gt_sents, pred_sents, n_bleu=4, n_cider=4, cider_sigma=6.0):
        """
        gt_sents: list of ground truth sentences
        pred_sents: list of ground truth sentences
        """
        assert len(gt_sents) == len(pred_sents)
        bleu_scorer = BleuScorer(n=n_bleu)
        rouge_scorer = Rouge()
        cider_scorer = CiderScorer()
        meteor_scorer = Meteor()
        meteor_scorer._stat = types.MethodType(_stat,  meteor_scorer)
        
        rouge_scores = []
        meteor_scores = []
        
        eval_line = 'EVAL'
        meteor_scorer.lock.acquire()
        count = 0
        for i in range(len(gt_sents)):
            bleu_scorer += (pred_sents[i],gt_sents[i])
            cider_scorer += (pred_sents[i],gt_sents[i])
            rouge_score = rouge_scorer.calc_score([pred_sents[i]], gt_sents[i])
            rouge_scores.append(rouge_score)
            stat = meteor_scorer._stat(pred_sents[i],gt_sents[i])
            eval_line += ' ||| {}'.format(stat)
            count += 1 
        
        meteor_scorer.meteor_p.stdin.write('{}\n'.format(eval_line).encode())
        meteor_scorer.meteor_p.stdin.flush()
        for _ in range(count):
            meteor_scores.append(float(meteor_scorer.meteor_p.stdout.readline().strip()))
        meteor_score = float(meteor_scorer.meteor_p.stdout.readline().strip())
        meteor_scorer.lock.release()
        
        blue_score, _ = bleu_scorer.compute_score(option='closest')
        rouge_score = np.mean(np.array(rouge_scores))
        cider_score, _ = cider_scorer.compute_score()
        
        return {"bleu1":blue_score[0],
                "bleu2":blue_score[1],
                "bleu3":blue_score[2],
                "bleu4":blue_score[3],
                "rouge":rouge_score,
                "meteor":meteor_score, 
                "cider":cider_score}
        
    @staticmethod
    def coco_evaluator(gt_path,result_file_path):
        coco_gt, coco_pred = to_coco_annotation_format(gt_path, result_file_path) 
        evaluater = COCOEvalCap(coco_gt, coco_pred)
        evaluater.evaluate()


if __name__ == "__main__":
    file_path = os.path.join(args.out_dir, args.out_dict_name+".json")

    LangGenerationEvluator.coco_evaluator(gt_path=GT_PATH, result_file_path=file_path)
    
