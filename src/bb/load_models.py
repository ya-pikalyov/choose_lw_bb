"""
Load pretrained models
"""

from torchvision import models
import torch
import torch.nn as nn
from src.models.backbone.convnext.convnext_model import convnextv2_nano_1k_224_ema
from src.models.backbone.davit.davit_model import davit_tiny
from src.models.backbone.edgeformer.edgeformer_model import edgeformer
from src.models.backbone.edgenext.edgenext_model import edgenext_base
from src.models.backbone.efficient_former.efficient_former_model import efficientformerv2_l
from src.models.backbone.efficientnet2.efficient_net_models import efficientnet_v2_t_gc
from src.models.backbone.mob_vit_v3.mobvit_model import MobileViTv3
from src.models.backbone.nas_vit.nas_vit_model import create_nas_model
from src.models.backbone.next_vit.next_vit_model import nextvit_small
from src.models.backbone.swift_former.swift_former_model import SwiftFormer_L1
from src.models.backbone.fasternet.fasternet_model import create_fasternet
from src.models.backbone.mobileone.mobileone import mobileone, reparameterize_model
from src.models.backbone.cas_vit.cas_vit_model import rcvit_xs
from src.models.backbone.faster_vit.faster_vit_model import faster_vit_0_224
from src.models.backbone.rep_vit.rep_vit_model import repvit_m0_9
from src.models.backbone.hiera.hiera_model import hiera_tiny_224
from src.models.backbone.flexi_vit.flexi_vit_model import flexi_vit_tiny
from src.models.backbone.inception_next.inception_next_model import get_inception_next_model
from src.models.backbone.tiny_vit.tvit_model import tiny_vit_21m_224
from src.models.backbone.moganet.moganet_model import moganet_xtiny
#from src.models.backbone.nas_vit.nas_vit_model import create_nas_model
from src.models.backbone.fix_st_weights import fix_sd_davit, fix_sd_convnext, fix_sd_tvit, fix_sd_swift
from src.utils.utils_prepro import yaml2argparse, load_cfg

# path to current project directory
PATH2PROJ = r'//home//servml//Документы//code//choose_bb//'

# form strok for chekpont holder directory
_tmp_path2ckpt = r'{}in//models//backbone//weights//{}.pth'
_tmp_path2cfgs = r'{}in//models//backbone//cfg//{}.yaml'


###
_dict_models = {
                'convnextv2_nano_224_ema' : convnextv2_nano_1k_224_ema,
                'davit_t' : davit_tiny,
                'edgeformer_s' : edgeformer,
                'edgenext_base_usi' : edgenext_base,
                'efficientformerv2_l' : efficientformerv2_l,
                'efficientnetv2_t_gc' : efficientnet_v2_t_gc,
                'mobvit3_s_l2' : MobileViTv3,
                'next_vit_small' : nextvit_small,
                'resnet50' : models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
                'tiny_vit_21m' : tiny_vit_21m_224,
                'swiftformer_l1' : SwiftFormer_L1,
                'mobileone_s0_unfused': mobileone,
                'fasternet_t0' : create_fasternet,
                'cas-vit-xs' : rcvit_xs,
                'fastervit_0_224_1k' : faster_vit_0_224,
                'repvit_m0_9_distill_450e' : repvit_m0_9,
                'hiera_tiny_224' : hiera_tiny_224,
                'flexivit_small.1200ep_in1k' : flexi_vit_tiny,
                'inceptionnext_tiny' : get_inception_next_model,
                #'nas_vit_ckpt_360.pth' : create_nas_model,
                }
###
models_with_cfgs = ['edgeformer_s', 'mobvit3_s_l2', 'fasternet_t0']
###


def fix_sd(model_name, state_dict, model_ft):
    if "convnext" in model_name:
        state_dict = fix_sd_convnext(state_dict, model_ft)
    elif "davit" in model_name:
        state_dict = fix_sd_davit(state_dict)
    elif "tiny_vit" in model_name:
        state_dict = fix_sd_tvit(state_dict, model_ft)
    elif "swiftformer" in model_name:
        state_dict = fix_sd_swift(state_dict)
    return state_dict


def load_weights_from_ckpt(model_name, model_ft, device=torch.device('cpu')):
    """
    :param model_name:
    :param model_ft:
    :param device:
    :return:
    """
    path2ckpt = _tmp_path2ckpt.format(PATH2PROJ, model_name)
    try:
        ckpt = torch.load(path2ckpt, map_location=device)
    except:
        ckpt = torch.load(f'{path2ckpt}.tar', map_location=device)
    state_dict = None
    if type(ckpt) == dict:
        if 'model' in ckpt.keys():
            state_dict = ckpt['model']
        elif 'state_dict' in ckpt.keys():
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt
    state_dict = fix_sd(model_name, state_dict, model_ft)
    try:
        model_ft.load_state_dict(state_dict)
    except:
        missing_keys, unexpected_keys = model_ft.load_state_dict(torch.load(path2ckpt), False)
    return model_ft


def load_cfgs(model_name, feat_ext=False):
    """
    :param model_name:
    :return:
    """
    model_ft = _dict_models[model_name]
    if model_name in models_with_cfgs:
        path2cfg = _tmp_path2cfgs.format(PATH2PROJ, model_name)
        if 'edgeformer' in model_name:
            cfg = yaml2argparse(path2cfg)
            setattr(cfg, "model.activation.inplace", getattr(cfg, "model.classification.activation.inplace", False))
            setattr(cfg, "model.activation.neg_slope", getattr(cfg, "model.classification.activation.neg_slope", 0.1))
            model_ft = model_ft(cfg)
        elif 'mobvit3' in model_name:
            cfg = yaml2argparse(path2cfg)
            model_ft = model_ft(cfg)
        elif 'fasternet' in model_name:
            cfg = load_cfg(path2cfg)
            print(cfg)
            model_ft = model_ft(cfg)
    else:
        model_ft = model_ft()
    return model_ft


def load_models(model_name):
    """
    :param model_name:
    :return:
    """
    if 'resnet' in model_name:
        model_ft = _dict_models[model_name]
    else:
        model_ft = load_cfgs(model_name)
        if not 'flexivit' in model_name:
            model_ft = load_weights_from_ckpt(model_name, model_ft)
    return model_ft


def test1(name_model):
    print(name_model)
    model = load_models(name_model).to('cuda')
    print(model)
    tensor = torch.Tensor(2, 3, 224, 224).to('cuda')
    try:
        res = model(tensor)
    except:
        tensor = torch.Tensor(2, 3, 240, 240).to('cuda')
        res = model(tensor)
    print(res)
    del model


def test2():
    for name_model in _dict_models.keys():
        test1(name_model)

if __name__ == "__main__":
    test2()