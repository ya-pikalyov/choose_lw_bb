import torch
from src.utils.check_bb import count_params_flops_thop, measure_fps

from src.bb.bisenet.model import BiSeNetV2
from src.bb.edgeformer.edgeformer_model import edgeformer
from src.bb.edgenext.edgenext_model import edgenext_x_small
from src.bb.efficient_unet.model import EfficientUNetPP
from src.bb.efficientnet2.efficient_net_models import efficientnet_v2_t_gc
from src.bb.fasternet.fasternet_model import create_fasternet
from src.bb.faster_vit.faster_vit_model import faster_vit_0_224
from src.bb.flexi_vit.flexi_vit_model import flexi_vit_tiny
from src.bb.inception_next.inception_next_model import inceptionnext_tiny
from src.bb.mob_vit_v3.mobvit_model import MobileViTv3
from src.bb.mobileone.mobileone import mobileone
from src.bb.pcpe_yolo.model import PCPE_YOLO_Light
from src.bb.sbcformer.model import SBCFormerBackbone
from src.bb.seaformer.model import SeaFormerBackbone
from src.bb.sema_yolo.model import SEMA_YOLO_Backbone
from src.bb.stripnet.stripnet_model import get_stripnet_tiny_tiny
from src.bb.tiny_vit.tvit_model import tiny_vit_5m_224
from src.bb.wlusnet.model import WLUSNet


import yaml
import argparse

import transformers
import collections
import torch.nn as nn


def _patch_sequential():
    if not hasattr(nn.Sequential, "total_ops"):
        nn.Sequential.total_ops = 0
    if not hasattr(nn.Sequential, "total_params"):
        nn.Sequential.total_params = 0



def flatten_yaml_as_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten_yaml_as_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def load_cfg(cfg):
    hyp = None
    if isinstance(cfg, str):
        with open(cfg, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    return argparse.Namespace(**hyp)

def load_config_file(name, opts):
    with open(name, 'r') as yaml_file:
        try:
            cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)
            flat_cfg = flatten_yaml_as_dict(cfg)
            for k, v in flat_cfg.items():
                #if hasattr(opts, k):
                if 'model' in k:
                    setattr(opts, k, v)
        except yaml.YAMLError as exc:
            print('Error while loading config file: {}'.format(name))
            print('Error message: {}'.format(str(exc)))
    return opts

def yaml2argparse(path2yaml, description='default arguments'):
    parser = argparse.ArgumentParser(description=description, add_help=True)
    opts = parser.parse_args()
    opts = load_config_file(path2yaml, opts)
    return opts

def make_dict_models():
    '''
    bise_model = BiSeNetV2()
    edgeformer_model = edgeformer()
    edgenext_model = edgenext_x_small()
    effunet_pp = EfficientUNetPP()
    cfg = load_cfg('/home/servml/Документы/code/Yaroslav/choose_bb_/in/models/backbone/cfg/fasternet_t0.yaml')
    fasternet = create_fasternet(cfg)
    effnet = efficientnet_v2_t_gc()
    fastervit = faster_vit_0_224()
    flexi_vit = flexi_vit_tiny()
    inceptionnext = inceptionnext_tiny()
    mobvit = MobileViTv3()
    mobone = MobileOne()
    pcpe = PCPE_YOLO()
    sbcformer = SBCFormerBackbone()
    seaformer = SeaFormerBackbone()
    semayolo = SEMA_YOLO_Backbone()
    stripnet = get_stripnet_tiny_tiny()
    tinyvit = tiny_vit_5m_224()
    wlusnet = WLUSNet()
    '''

    cfg_fasternet = load_cfg('/home/servml/Документы/code/Yaroslav/choose_bb_/in/models/backbone/cfg/fasternet_t0.yaml')
    fasternet = create_fasternet(cfg_fasternet)

    cfg_edgeformer = yaml2argparse('/home/servml/Документы/code/Yaroslav/choose_bb_/in/models/backbone/cfg/edgeformer_s.yaml')
    setattr(cfg_edgeformer, "model.activation.inplace", getattr(cfg_edgeformer, "model.classification.activation.inplace", False))
    setattr(cfg_edgeformer, "model.activation.neg_slope", getattr(cfg_edgeformer, "model.classification.activation.neg_slope", 0.1))
    edgeformer_model= edgeformer(cfg_edgeformer)

    cfg_mob_vit = yaml2argparse('/home/servml/Документы/code/Yaroslav/choose_bb_/in/models/backbone/cfg/mobvit3_s_l2.yaml')
    mobvit = MobileViTv3(cfg_mob_vit)



    dict_nn = {
                #'bise_model' : BiSeNetV2(),
                #'edgeformer_model' : edgeformer_model,
                #'edgenext_model' : edgenext_x_small(),
                #'effunet_pp' : EfficientUNetPP(),
                #'fasternet' : fasternet,
                #'effnet' : efficientnet_v2_t_gc(),
                #'fastervit' : faster_vit_0_224(),
                #'flexi_vit' : flexi_vit_tiny(),
                #'inceptionnext' : inceptionnext_tiny(),
                #'mobvit' : mobvit,
                #'mobone' : mobileone(),
                #'pcpe' : PCPE_YOLO_Light(),
                #'sbcformer' : SBCFormerBackbone(),
                #'seaformer' : SeaFormerBackbone(),
                'semayolo' : SEMA_YOLO_Backbone(),
                'stripnet' : get_stripnet_tiny_tiny(),
                'tinyvit' : tiny_vit_5m_224(),
                'wlusnet' : WLUSNet()
    }

    return dict_nn


def func_models(B, H, W):


    dict_nn = make_dict_models()


    for i, kv in enumerate(dict_nn.items()):
        name_model, model = kv
        print('Testing {}...'.format(name_model))
        try:
            #count_params_flops_thop(model, B, H, W)
            measure_fps(model, B, H, W, num_warmup=10, num_iters=1000, device="cuda")
        except:
            #count_params_flops_thop(model, B, H=240, W=240)
            measure_fps(model, B, H=240, W=240, num_warmup=10, num_iters=1000, device="cuda")


if __name__ == '__main__':
    func_models(1, 224, 224)