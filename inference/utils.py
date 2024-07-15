
import torch 
import json 
import copy 
# from inference.coco import COCO
from inference.coco import COCO

COCO_DATA_INFO = {
        "description" : "COCO 2014 Dataset",
        "url": "http://cocodataset.org",
        "version": "1.0",
        "year": 2014,
        "contributor": "COCO Consortium",
        "data_created": "2017/09/01"
    }
COCO_DATA_LICENSES = [
     {"url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
      "id": 1,
      "name": "Attribution-NonCommercial-ShareAlike License"} ,
     {
      "url": "http://creativecommons.org/licenses/by-nc/2.0/",
      "id": 2,
      "name": "Attribution-NonCommercial License"
     },
     {
      "url":"http://creativecommons.org/licenses/by-nc-nd/2.0/",
      "id": 3,
      "name": "Attribution-NonCommercial-NoDerivs License"
     },
     {
      "url":"http://creativecommons.org/licenses/by/2.0/",
      "id": 4,
      "name": "Attribution License"
     },
     {
     "url": "http://creativecommons.org/licenses/by-sa/2.0/",
     "id": 5,
     "name": "Attribution-ShareAlike License"
     },
     {
      "url":"http://creativecommons.org/licenses/by-nd/2.0/",
      "id": 6,
      "name": "Attribution-NoDerivs License"
     },
     {
      "url":"http://flickr.com/commons/usage/",
      "id": 7,
      "name":"No known copyright restrictions"
     },
     {
      "url":"http://www.usa.gov/copyright.shtml",
      "id": 8,
      "name": "United States Government Work"
     }
]

def to_device(batch, device):
    for k,v in batch.items():
        if isinstance(v, torch.Tensor):
            v = v.to(device)
            batch[k] = v
    return batch 


def to_coco_annotation_format(gt_path, result_path, test_ae=True):
    with open(result_path, "r") as f:
        data =json.load(f)
    with open(gt_path, "r") as f:
        gt_data = json.load(f)
    
    coco_anno = {
               "type": "caption",
               "info": COCO_DATA_INFO,
               "licenses": COCO_DATA_LICENSES,
               "images": gt_data["images"],
               "annotations": gt_data["annotations"]
              }
    
    coco_ref = {"type": "caption",
                "info": COCO_DATA_INFO,
                "licenses": COCO_DATA_LICENSES,
                "images": copy.deepcopy(gt_data["images"]),
                } # predict coco

    coco_ref["annotations"] = data["pred_sent"]

    coco_gt = COCO(annotation_data=coco_anno)
    coco_pred = COCO(annotation_data=coco_ref)
    return coco_gt, coco_pred
    
    

def update_dict(result_dict, new_dict, key):
    """ 
     updating result_dict by merging new_dict 
    key = 'gt_sent' or 'pred_sent'
    """
    for k in new_dict[key].keys():
        if k not in result_dict[key]:
            result_dict[key][k] = new_dict[key][k]
        else:
            result_dict[key][k].extend(new_dict[key][k])
    return result_dict


def set_dict2list(data_path):
    """
    data_path: json file of output resutls
        results = {gt_sent: {img_id: [s1, s2, s3 ..]}, {[]},
                   pred_sent: {img_id: [s1, s2, s3, ..]}, {[]}
    return: dict = {
        one2one: [
                  [pred_s1,pred_s2,....] 
                  [[gt_s1],[gt_s2], ...]
                  ],
        one2many: [
                  [pred_s1,pred_s2,....]
                  [[gt_s11,gt_s12,..],[gt_s21,gt_s22,...]]
                  ]
    }
    """
    output_dict = {
        "one2one" : [[],[]],
        "one2many": [[],[]]
    }
    with open(data_path,"r") as f:
        result_data = json.load(f)
    all_img_ids = list(result_data["gt_sent"].keys())
    for img_id in all_img_ids:
        # set one to one
        output_dict["one2one"][0].extend(result_data["pred_sent"][img_id])
        output_dict["one2many"][0].extend(result_data["pred_sent"][img_id])
        
        #output_dict["one2one"][1].extend(result_data["gt_sent"][img_id])
        
        for i in range(len(result_data["pred_sent"][img_id])):
            output_dict["one2many"][1].append(result_data["gt_sent"][img_id])
            output_dict["one2one"][1].append([result_data["gt_sent"][img_id][i]])
    return output_dict

def load_model_from_ckpt(model, ckpt_path):
    pass 