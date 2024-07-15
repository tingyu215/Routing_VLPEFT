from torch.utils.data import DataLoader, Dataset, Sampler
from pathlib import Path
from collections import defaultdict
import json
import random
from multiprocessing import Pool
import h5py
import pickle
import math
from tqdm import tqdm
import torch
import numpy as np
from copy import deepcopy
import re
import os
from PIL import Image 

from torch.utils.data.distributed import DistributedSampler



class VQAFineTuneDataset(Dataset):
    def __init__(self, tokenizer, split='train', raw_dataset=None, rank=-1, topk=-1, verbose=True, args=None, mode='train', max_length=20, use_vis_prefix=False, prefix_length=1, image_transforms=None, vqa_dir="../VQAv2", image_dir="../MSCOCO/Images", use_classifier=True):
        super().__init__()
        self.tokenizer = tokenizer
        self.use_classifier = use_classifier
        self.prompt = "<s>"
        self.image_dir = image_dir
        self.transform = image_transforms
        self.max_length = max_length # max length for question tokens
        self.use_vis_prefix = use_vis_prefix # if use vis prefix, padding mask +1, maxlengh-1

        self.prefix_mask = torch.ones(1, prefix_length)


        self.raw_dataset = raw_dataset
        self.topk = topk
        self.verbose = verbose
        self.args = args

        self.mode = mode

        # Loading datasets to data
        self.sources = split.split(',')
        if self.verbose:
            print('Data sources: ', self.sources)


        self.answer_normalizer = VQAEvaluator()

        self.img_ids_to_source = {}
        data_info_dicts = []
        for source in self.sources:
            # data_info_path = dataset_dir.joinpath(f'vqa/{source}.json')
            data_info_path = os.path.join(vqa_dir, f"{source}.json")
            with open(data_info_path) as f:
                _data_info_dicts = json.load(f)
                for _d in _data_info_dicts:
                    if 'vg_qa_full' == source:
                        self.img_ids_to_source[_d['img_id']] = 'vg'
                    elif 'train2014' in _d['img_id']:
                        self.img_ids_to_source[_d['img_id']] = 'train2014'
                    elif 'val2014' in _d['img_id']:
                        self.img_ids_to_source[_d['img_id']] = 'val2014'
                    elif 'test' in _d['img_id']:
                        self.img_ids_to_source[_d['img_id']] = 'test2015'
                    else:
                        self.img_ids_to_source[_d['img_id']] = source
                        _d['source'] = source

                data_info_dicts.extend(_data_info_dicts)
            if self.verbose:
                print(f"Loaded {len(_data_info_dicts)} data from", source)

        data = data_info_dicts

        self.n_gpus = torch.cuda.device_count()

        self.rank = rank

        if isinstance(self.topk, float) and (0 < self.topk <= 1):
            used_samples = int(self.topk * len(data))
            data = random.sample(data, used_samples)
            if self.verbose:
                print(f"Use only {len(data)} data")

        elif self.topk > 0:
            data = data[:int(self.topk)]
            if self.verbose:
                print(f"Use only {len(data)} data")

        self.data = data

        if self.verbose:
            print("# all sentences:", len(self.data))

    def __len__(self):
        return len(self.data)

    def _get_image(self, img_path):
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img, return_tensors="pt")["pixel_values"]
        return img 

    def __getitem__(self, idx):

        out_dict = {}

        datum = self.data[idx]
        ###### Image ######
        img_id = datum['img_id']
        out_dict['img_id'] = img_id
        source = self.img_ids_to_source[img_id]
        image_path = os.path.join(self.image_dir, source, f"{img_id}.jpg")

        img = self._get_image(image_path)
        out_dict['images'] = img
        
        ###### Text #####
        if 'sent' in datum:
            sent = datum['sent']
        elif 'question' in datum:
            sent = datum['question']
        
        if self.use_vis_prefix:
            input_ids = self.tokenizer.encode(sent, max_length=self.max_length-1, truncation=True, return_tensors="pt")
            pad_tensor = torch.ones((1, self.max_length -1 - input_ids.size()[1])) * -100
            padding_mask = torch.zeros(1,self.max_length-1)

        else:
            input_ids = self.tokenizer.encode(sent, max_length=self.max_length, truncation=True, return_tensors="pt")
            pad_tensor = torch.ones((1, self.max_length - input_ids.size()[1])) * -100
            padding_mask = torch.zeros(1,self.max_length)

        input_ids = torch.cat((input_ids, pad_tensor.to(torch.long)),dim=1)

        sent_len = torch.sum(input_ids != -100)
        padding_mask[:,:sent_len] = 1
        # target_sen[:, cap_len-1:]=-100
        if self.use_vis_prefix:
            padding_mask = torch.cat((self.prefix_mask, padding_mask), dim=1)
        out_dict["padding_mask"] = padding_mask

        question_id = datum['question_id']
        out_dict['question_id'] = question_id


        out_dict['sent'] = sent
        out_dict['input_ids'] = torch.LongTensor(input_ids)
        out_dict['input_length'] = len(input_ids)

        if 'is_topk_optimal' in datum:
            out_dict['is_topk_optimal'] = datum['is_topk_optimal']

        if 'label' in datum:
            label = datum['label']
            out_dict['label'] = label

            # 3129 topk answers
            if self.use_classifier:
                target = torch.zeros(self.raw_dataset.num_answers)
                for ans, score in label.items():
                    target[self.raw_dataset.ans2label[ans]] = score
                out_dict['target'] = target

            elif self.args.raw_label:
                answers = datum['answers']
                answer = random.choice(answers)['answer']

                if self.args.answer_normalize:
                    answer = self.answer_normalizer.normalize_answer(answer)

                score = int(len(answers) > 0)

                out_dict['answer'] = answer
                out_dict['score'] = score
                out_dict['all_answers'] = [a['answer'] for a in answers]

                target_ids = self.tokenizer.encode(answer, max_length=10, truncation=True)

                out_dict['target_ids'] = torch.LongTensor(target_ids)
                out_dict['target_length'] = len(target_ids)

            else:
                answers = []
                scores = []
                for a, s in label.items():
                    answers.append(a)
                    scores.append(s)

                score_sum = sum(scores)

                if score_sum == 0:
                    answer = ''
                    score = 0.
                else:
                    prob = [score / score_sum for score in scores]
                    choice = np.random.multinomial(1, prob).argmax()
                    answer = answers[choice]
                    score = scores[choice]
                    assert len(answer) > 0, (sent, label, choice, answer)

                out_dict['answer'] = answer
                out_dict['score'] = score
                out_dict['all_answers'] = answers


                target_ids = self.tokenizer.encode(answer, max_length=10, truncation=True)

                out_dict['target_ids'] = torch.LongTensor(target_ids)
                out_dict['target_length'] = len(target_ids)

        return out_dict

    def collate(self, batch):
        # simplified collate function, which only handles cases with classification. to be modified.
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        img_id = dict_batch["img_id"]
        img_tensors = torch.stack(dict_batch["images"]).squeeze(1)

        input_ids = torch.stack(dict_batch["input_ids"]).squeeze(1)
        # padding id = 1 in roberta
        input_ids[input_ids==-100]=1
        padding_mask = torch.stack(dict_batch["padding_mask"]).squeeze(1)
        
        sources = "".join(self.sources)
        if not "test" in sources or "karpathy_test" in sources:
            target = torch.stack(dict_batch["target"]).squeeze(1)
            label = dict_batch["label"]
        else:
            target = None
            label = None

        question_text = dict_batch["sent"]
        question_ids = dict_batch["question_id"]
        # label = dict_batch["label"]
        input_length = dict_batch["input_length"]
        return {
            "img_id": img_id,
            "img_tensors": img_tensors,
            "input_ids": input_ids,
            "padding_mask": padding_mask,
            "target": target,
            "question_text": question_text,
            "question_ids": question_ids,
            "label": label,
            "input_length": input_length
        }



def get_loader(args, split='karpathy_train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0, topk=-1):

    verbose = (gpu == 0)

    _dset = VQADataset(split, verbose)

    dataset = VQAFineTuneDataset(
        split,
        raw_dataset=_dset,
        rank=gpu,
        topk=topk,
        verbose=verbose,
        args=args,
        mode=mode)

    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None

    if mode == 'train':
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=(sampler is None),
            num_workers=workers, pin_memory=True, sampler=sampler,
            collate_fn=dataset.collate_fn)
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers, pin_memory=True,
            sampler=sampler,
            shuffle=None if (sampler is not None) else False,
            collate_fn=dataset.collate_fn,
            drop_last=False)

    if verbose:
        loader.evaluator = VQAEvaluator(_dset)

    loader.task = 'vqa'

    return loader


class VQADataset:
    """
    A VQA data example in json file:
        {
            "answer_type": "other",
            "img_id": "COCO_train2014_000000458752",
            "label": {
                "net": 1
            },
            "question_id": 458752000,
            "question_type": "what is this",
            "sent": "What is this photo taken looking through?"
        }
    """

    def __init__(self, splits: str, verbose=True):
        self.name = splits
        self.splits = splits.split(',')
        
        dataset_dir = "../VQAv2"
        
        with open(os.path.join(dataset_dir, 'v2_mscoco_train2014_annotations.json')) as f:
            train2014_data = json.load(f)
        with open(os.path.join(dataset_dir, 'v2_mscoco_val2014_annotations.json')) as f:
            val2014_data = json.load(f)
        
        train2014_id2datum = {}
        
        for datum in train2014_data['annotations']:
            qid = datum['question_id']
            train2014_id2datum[qid] = datum
        val2014_id2datum = {}
        for datum in val2014_data['annotations']:
            qid = datum['question_id']
            val2014_id2datum[qid] = datum
        self.id2datum_gt = {**train2014_id2datum, **val2014_id2datum}

        # Loading datasets
        self.data = []
        for split in self.splits:
            self.data.extend(
                json.load(open(os.path.join(dataset_dir, f"{split}.json"))))

        if verbose:
            print("Load %d data from split(s) %s." %
                  (len(self.data), self.name))

        # Convert list to dict (for evaluation)
        self.id2datum = {
            datum['question_id']: datum
            for datum in self.data
        }

        # Topk Answers

        self.ans2label = json.load(
            open(os.path.join(dataset_dir, "trainval_ans2label.json")))
        self.label2ans = json.load(
            open(os.path.join(dataset_dir, "trainval_label2ans.json")))
        assert len(self.ans2label) == len(self.label2ans)

        if verbose:
            print('# Answers:', len(self.ans2label))

    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)


class VQAEvaluator:
    def __init__(self, dataset: VQADataset = None):
        self.dataset = dataset

        """https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py"""

        self.contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't", \
							 "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't", \
							 "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hed've": "he'd've", \
							 "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", \
							 "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's", \
							 "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've", \
							 "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't", \
							 "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've", \
							 "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", \
							 "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll", \
							 "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've", \
							 "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've", \
							 "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd", "thered've": "there'd've", \
							 "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", \
							 "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", \
							 "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're", \
							 "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've", \
							 "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", \
							 "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've", \
							 "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've", \
							 "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", \
							 "youll": "you'll", "youre": "you're", "youve": "you've"}

        self.manualMap    = { 'none': '0',
							  'zero': '0',
							  'one': '1',
							  'two': '2',
							  'three': '3',
							  'four': '4',
							  'five': '5',
							  'six': '6',
							  'seven': '7',
							  'eight': '8',
							  'nine': '9',
							  'ten': '10'
							}

        self.articles     = ['a',
							 'an',
							 'the'
							]

        self.periodStrip  = re.compile("(?!<=\d)(\.)(?!\d)")
        self.commaStrip   = re.compile("(\d)(\,)(\d)")
        self.punct        = [';', r"/", '[', ']', '"', '{', '}',
							 '(', ')', '=', '+', '\\', '_', '-',
							 '>', '<', '@', '`', ',', '?', '!']

        self.n = 2

    def evaluate(self, quesid2ans: dict):
        score = 0.
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['label']
            if ans in label:
                score += label[ans]
        return score / len(quesid2ans)

    def dump_result(self, quesid2ans: dict, path):
        """
        Dump results to a json file, which could be submitted to the VQA online evaluation.
        VQA json file submission requirement:
            results = [result]
            result = {
                "question_id": int,
                "answer": str
            }
        :param quesid2ans: dict of quesid --> ans
        :param path: The desired path of saved file.
        """
        with open(path, 'w') as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                result.append({
                    'question_id': ques_id,
                    'answer': ans
                })
            json.dump(result, f, indent=4, sort_keys=True)

    def evaluate_raw(self, quesid2ans: dict, is_topk_optimal=None):
        """https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py"""

        gts = self.dataset.id2datum_gt

        self.accuracy     = {}
        self.evalQA       = {}
        self.evalQuesType = {}
        self.evalAnsType  = {}

        accQA = []
        accQuesType = {}
        accAnsType = {}

        # print("Computing accuracy")

        for quesId, resAns in tqdm(quesid2ans.items(), total=len(quesid2ans), ncols=80):

            quesId = int(quesId)

            datum = self.dataset.id2datum[quesId]

            if is_topk_optimal is None:
                pass
            elif 'is_topk_optimal' in datum:
                if datum['is_topk_optimal'] != is_topk_optimal:
                    continue

            resAns      = resAns.replace('\n', ' ')
            resAns      = resAns.replace('\t', ' ')
            resAns      = resAns.strip()
            resAns      = self.processPunctuation(resAns)
            resAns      = self.processDigitArticle(resAns)

            gtAcc  = []
            gtAnswers = [ans['answer'] for ans in gts[quesId]['answers']]
            if len(set(gtAnswers)) > 1:
                for ansDic in gts[quesId]['answers']:
                    ansDic['answer'] = self.processPunctuation(ansDic['answer'])
            for gtAnsDatum in gts[quesId]['answers']:
                otherGTAns = [item for item in gts[quesId]['answers'] if item!=gtAnsDatum]
                matchingAns = [item for item in otherGTAns if item['answer']==resAns]
                acc = min(1, float(len(matchingAns))/3)
                gtAcc.append(acc)
            quesType    = gts[quesId]['question_type']
            ansType     = gts[quesId]['answer_type']
            avgGTAcc = float(sum(gtAcc))/len(gtAcc)
            accQA.append(avgGTAcc)
            if quesType not in accQuesType:
                accQuesType[quesType] = []
            accQuesType[quesType].append(avgGTAcc)
            if ansType not in accAnsType:
                accAnsType[ansType] = []
            accAnsType[ansType].append(avgGTAcc)

            self.setEvalQA(quesId, avgGTAcc)
            self.setEvalQuesType(quesId, quesType, avgGTAcc)
            self.setEvalAnsType(quesId, ansType, avgGTAcc)


        if len(accQA) == 0:
            return {
                'overall': 0,
                'perQuestionType': {},
                'perAnswerType': {}
            }
        else:
            self.setAccuracy(accQA, accQuesType, accAnsType)

        return self.accuracy

    def normalize_answer(self, resAns):
        resAns      = resAns.replace('\n', ' ')
        resAns      = resAns.replace('\t', ' ')
        resAns      = resAns.strip()
        resAns      = self.processPunctuation(resAns)
        resAns      = self.processDigitArticle(resAns)
        resAns = resAns.replace(',', '')
        return resAns

    def processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + ' ' in inText or ' ' + p in inText) or (re.search(self.commaStrip, inText) != None):
                outText = outText.replace(p, '')
            else:
                outText = outText.replace(p, ' ')
        outText = self.periodStrip.sub("",
                                        outText,
                                        re.UNICODE)
        return outText

    def processDigitArticle(self, inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = self.manualMap.setdefault(word, word)
            if word not in self.articles:
                outText.append(word)
            else:
                pass
        for wordId, word in enumerate(outText):
            if word in self.contractions:
                outText[wordId] = self.contractions[word]
        outText = ' '.join(outText)
        return outText

    def setEvalQA(self, quesId, acc):
        self.evalQA[quesId] = round(100*acc, self.n)

    def setEvalQuesType(self, quesId, quesType, acc):
        if quesType not in self.evalQuesType:
            self.evalQuesType[quesType] = {}
        self.evalQuesType[quesType][quesId] = round(100*acc, self.n)

    def setEvalAnsType(self, quesId, ansType, acc):
        if ansType not in self.evalAnsType:
            self.evalAnsType[ansType] = {}
        self.evalAnsType[ansType][quesId] = round(100*acc, self.n)

    def setAccuracy(self, accQA, accQuesType, accAnsType):
        self.accuracy['overall']         = round(100*float(sum(accQA))/len(accQA), self.n)
        self.accuracy['perQuestionType'] = {quesType: round(100*float(sum(accQuesType[quesType]))/len(accQuesType[quesType]), self.n) for quesType in accQuesType}
        self.accuracy['perAnswerType']   = {ansType:  round(100*float(sum(accAnsType[ansType]))/len(accAnsType[ansType]), self.n) for ansType in accAnsType}
