from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig


def update_config(opt, config, ignore_args):
    def get_cfg_recursive(config_dict):
        all_attr_dotlist = []
        for k,v in config_dict.items():
            if not isinstance(v,DictConfig):
                all_attr_dotlist.extend([k])
            else:
                all_attr_dotlist.extend([f"{k}.{x}" for x in get_cfg_recursive(v)])
        return all_attr_dotlist
        
    cfg_content = config.__dict__['_content']
    all_cfg_dotlist = get_cfg_recursive(cfg_content)
    update_cfg_dot_list = []
    for attr in all_cfg_dotlist:
        if "." in attr:
            map_attr = attr.split(".")[-1]
        else:
            map_attr = attr 
        if hasattr(opt, map_attr):
            if map_attr in ignore_args:
                continue 
            value = getattr(opt,map_attr)
            if value != None:
                update_cfg_dot_list.append(f"{attr}={value}") 
    
    conf_dict = OmegaConf.from_dotlist(update_cfg_dot_list)
    config = OmegaConf.merge(config, conf_dict)
    return config