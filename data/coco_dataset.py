
import torch
from PIL import Image 
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import json
import copy
import random
import numpy as np
import clip


class COCOCLIPDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split,
        tokenizer,
        data_root,
        version,
        max_len,
        prefix_length,
        img_version = None,
        img_split = None,  # set for the karpathy test split which use image in val2014
        transform=None,
    ):
        super().__init__()
        self.split = split
        self.data_root = data_root
        self.version = version
        self.max_len = max_len
        self.load_img = True
        self.get_offline_feat = False
        self.feat_save_path = None
        self.offline_feat_model_name = None
        self.v_input_resolution = 224
        if self.feat_save_path is not None:
            assert self.offline_feat_model_name is not None 
            self.feat_save_path = f"{self.feat_save_path}/{self.offline_feat_model_name}/{self.split}"

        data_path = f"{self.data_root}/annotations/captions_{self.split}{self.version}.json"
        self.img_info, self.annotations = self._load_data(data_path)
        self.tokenizer = tokenizer
        self.vocab_size = len(self.tokenizer.decoder)

        self.prefix_mask = torch.ones(1, prefix_length)


        if img_split is None:
            self.img_split = split 
        else:
            self.img_split = img_split
        
        if version == "_karpathy":
            self.img_version = "2014"
        elif img_version is None:
            self.img_version = self.version
        else:
            self.img_version = img_version

        self.transform = transform

        # self.prefix = "Generated image caption:"
    def _get_image(self, img_path):
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img, return_tensors="pt")["pixel_values"]
        return img 

    def _load_data(self, path):
        with  open(path,"r") as f:
            data = json.load(f)
        data = copy.deepcopy(data)
        images_info = data["images"]
        annotations = data["annotations"]
        imginfo_dict = {}
        for img in images_info:
            if img["id"] not in imginfo_dict:
                imginfo_dict[img["id"]] = img
        if self.split == "test": # only pick one sample for one image id 
            random.shuffle(annotations)
            existed_imgid = []
            filtered_anno = []
            print("For test split, filter the data so that each image id only has one caption")
            for an in annotations:
                if an["image_id"] not in existed_imgid:
                    filtered_anno.append(an)
                    existed_imgid.append(an["image_id"])
            annotations = filtered_anno
        return imginfo_dict, annotations
    
    def _get_offline_feature_map(self, id):
        # for offline training of diffusion model
        assert self.feat_save_path is not None 
        path = f"{self.feat_save_path}/{self.split}_{str(id)}.npy"
        feat = torch.tensor(np.load(path))
        return feat 

    def __len__(self):
        return len(self.annotations) 

    def __getitem__(self, index):
        current_data = self.annotations[index] 
        current_img_info = self.img_info[current_data["image_id"]]
        img_id = current_data["image_id"]
        cur_img_path = f"{self.data_root}/Images/{self.img_split}{self.img_version}/{current_img_info['file_name']}"
        cap_text = current_data["caption"]
        sample_id = current_data["id"]
        
        cap_tensor = torch.tensor(self.tokenizer.encode(cap_text, max_length=self.max_len, truncation=True), dtype=torch.int64)
        # start + sentence + end 
        pad_tensor = torch.ones((self.max_len - cap_tensor.size()[0])) * -100
        cap_tensor = torch.cat((cap_tensor, pad_tensor.to(torch.long)),dim=0)
        cap_len = torch.sum(cap_tensor != -100)
        # target_sen = self.tokenizer.tokenize(cap_text,without_start=True)
        cap_tensor[cap_tensor==-100] = self.tokenizer.eos_token_id
        target_sen = cap_tensor.clone()

        padding_mask = torch.zeros(1,self.max_len)
        padding_mask[:,:cap_len] = 1
        padding_mask = torch.cat((self.prefix_mask, padding_mask), dim=1)  # adding prefix mask
        
        if self.get_offline_feat:
            feat_map = self._get_offline_feature_map(sample_id)
        else:
            feat_map = None 
        
        if self.load_img:
            img = self._get_image(cur_img_path)
        else:
            img = None
        
        caption_tensor_clip = clip.tokenize(cap_text)

        return {
            "caption_text": cap_text,   
            # "prefix_tensor": prefix_tensor,  
            "caption_tensor": cap_tensor,  #encoder input  decoder input
            "caption_tensor_clip": caption_tensor_clip,
            "caption_length": cap_len,     
            "target_sen": target_sen, # target sequence
            "padding_mask": padding_mask,
            "images": img,
            "image_id": img_id, 
            "id": sample_id,
            "feat_map": feat_map,
            # "prefix_len": prefix_len,
        }

    def collate(self, batch):

        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}
        
        image_id = torch.tensor(dict_batch["image_id"])
        sample_id = torch.tensor(dict_batch["id"])
        caption_text = dict_batch["caption_text"]

        caption_tensor = torch.stack(dict_batch["caption_tensor"]).squeeze(1)
        caption_length = torch.stack(dict_batch["caption_length"])
        target_sen = torch.stack(dict_batch["target_sen"]).squeeze(1)
        padding_mask = torch.stack(dict_batch["padding_mask"]).squeeze(1)
        caption_ori_tensor = copy.deepcopy(caption_tensor)

        caption_tensor_clip = torch.stack(dict_batch["caption_tensor_clip"]).squeeze(1)

        if self.get_offline_feat:
            feat_map_tensor = torch.stack(dict_batch["feat_map"]).squeeze(1)
        else:
            feat_map_tensor = None 
        
        if dict_batch["images"] is not None:
            try:
                context_input = torch.stack(dict_batch["images"]).squeeze(1)
            except:
                import ipdb 
                ipdb.set_trace()
        else:
            context_input = None 

        return {
            "text": caption_text,
            # "encoder_input_ids": prefix_tensor,
            "encoder_input_ids": caption_tensor,
            "caption_tensor_clip":caption_tensor_clip,
            "decoder_input_ids": caption_ori_tensor,
            "input_length": caption_length,
            "tgt_sen": target_sen,
            "encoder_padding_mask": padding_mask,
            "context_inputs": context_input,
            "context_attention_mask": None,
            "image_id": image_id,
            "sample_id": sample_id,
            "feat_map": feat_map_tensor
        }





class COCOCLIPDatasetITuning(torch.utils.data.Dataset):
    def __init__(
        self,
        split,
        tokenizer,
        data_root,
        version,
        max_len,
        prefix_length,
        img_version = None,
        img_split = None,  # set for the karpathy test split which use image in val2014
        transform=None,
    ):
        super().__init__()
        self.split = split
        self.data_root = data_root
        self.version = version
        self.max_len = max_len
        self.load_img = True
        self.get_offline_feat = False
        self.feat_save_path = None
        self.offline_feat_model_name = None
        self.v_input_resolution = 224
        if self.feat_save_path is not None:
            assert self.offline_feat_model_name is not None 
            self.feat_save_path = f"{self.feat_save_path}/{self.offline_feat_model_name}/{self.split}"

        data_path = f"{self.data_root}/annotations/captions_{self.split}{self.version}.json"
        self.img_info, self.annotations = self._load_data(data_path)
        self.tokenizer = tokenizer
        self.vocab_size = len(self.tokenizer.decoder)

        if img_split is None:
            self.img_split = split 
        else:
            self.img_split = img_split
        
        if version == "_karpathy":
            self.img_version = "2014"
        elif img_version is None:
            self.img_version = self.version
        else:
            self.img_version = img_version

        self.transform = transform

        self.prefix = "<s>"
    
    def _get_image(self, img_path):
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img, return_tensors="pt")["pixel_values"]
        return img 

    def _load_data(self, path):
        with  open(path,"r") as f:
            data = json.load(f)
        data = copy.deepcopy(data)
        images_info = data["images"]
        annotations = data["annotations"]
        imginfo_dict = {}
        for img in images_info:
            if img["id"] not in imginfo_dict:
                imginfo_dict[img["id"]] = img
        if self.split == "test": # only pick one sample for one image id 
            random.shuffle(annotations)
            existed_imgid = []
            filtered_anno = []
            print("For test split, filter the data so that each image id only has one caption")
            for an in annotations:
                if an["image_id"] not in existed_imgid:
                    filtered_anno.append(an)
                    existed_imgid.append(an["image_id"])
            annotations = filtered_anno
        return imginfo_dict, annotations
    
    def _get_offline_feature_map(self, id):
        # for offline training of diffusion model
        assert self.feat_save_path is not None 
        path = f"{self.feat_save_path}/{self.split}_{str(id)}.npy"
        feat = torch.tensor(np.load(path))
        return feat 

    def __len__(self):
        return len(self.annotations) 

    def __getitem__(self, index):
        current_data = self.annotations[index] 
        current_img_info = self.img_info[current_data["image_id"]]
        img_id = current_data["image_id"]
        cur_img_path = f"{self.data_root}/Images/{self.img_split}{self.img_version}/{current_img_info['file_name']}"
        cap_text = current_data["caption"]
        sample_id = current_data["id"]

        cap_text = self.prefix + cap_text
        cap_tensor = torch.tensor(self.tokenizer.encode(cap_text, max_length=self.max_len, truncation=True), dtype=torch.int64)
        # start + sentence + end 
        pad_tensor = torch.ones((self.max_len - cap_tensor.size()[0])) * -100
        cap_tensor = torch.cat((cap_tensor, pad_tensor.to(torch.long)),dim=0)
        cap_len = torch.sum(cap_tensor != -100)

        cap_tensor[cap_tensor==-100] = self.tokenizer.eos_token_id
        target_sen = cap_tensor.clone()

        padding_mask = torch.zeros(1,self.max_len)
        padding_mask[:,:cap_len] = 1
        
        if self.get_offline_feat:
            feat_map = self._get_offline_feature_map(sample_id)
        else:
            feat_map = None 
        
        if self.load_img:
            img = self._get_image(cur_img_path)
        else:
            img = None
        
        caption_tensor_clip = clip.tokenize(cap_text)

        return {
            "caption_text": cap_text,   
            # "prefix_tensor": prefix_tensor,  
            "caption_tensor": cap_tensor,  #encoder input  decoder input
            "caption_tensor_clip": caption_tensor_clip,
            "caption_length": cap_len,     
            "target_sen": target_sen, # target sequence
            "padding_mask": padding_mask,
            "images": img,
            "image_id": img_id, 
            "id": sample_id,
            "feat_map": feat_map,
            # "prefix_len": prefix_len,
        }

    def collate(self, batch):

        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}
        
        image_id = torch.tensor(dict_batch["image_id"])
        sample_id = torch.tensor(dict_batch["id"])
        caption_text = dict_batch["caption_text"]

        caption_tensor = torch.stack(dict_batch["caption_tensor"]).squeeze(1)
        caption_length = torch.stack(dict_batch["caption_length"])
        target_sen = torch.stack(dict_batch["target_sen"]).squeeze(1)
        padding_mask = torch.stack(dict_batch["padding_mask"]).squeeze(1)
        caption_ori_tensor = copy.deepcopy(caption_tensor)

        caption_tensor_clip = torch.stack(dict_batch["caption_tensor_clip"]).squeeze(1)

        if self.get_offline_feat:
            feat_map_tensor = torch.stack(dict_batch["feat_map"]).squeeze(1)
        else:
            feat_map_tensor = None 
        
        if dict_batch["images"] is not None:
            try:
                context_input = torch.stack(dict_batch["images"]).squeeze(1)
            except:
                import ipdb 
                ipdb.set_trace()
        else:
            context_input = None 

        return {
            "text": caption_text,
            # "encoder_input_ids": prefix_tensor,
            "encoder_input_ids": caption_tensor,
            "caption_tensor_clip":caption_tensor_clip,
            "decoder_input_ids": caption_ori_tensor,
            "input_length": caption_length,
            "tgt_sen": target_sen,
            "encoder_padding_mask": padding_mask,
            "context_inputs": context_input,
            "context_attention_mask": None,
            "image_id": image_id,
            "sample_id": sample_id,
            "feat_map": feat_map_tensor
        }