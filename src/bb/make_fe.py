'''
format nn from classification task of ImageNet to feature extractors for new task
'''

import torch
import torch.nn as nn
from src.models.backbone.load_models import load_models
from functools import partial

class MLP(nn.Module):
    """ MLP classification head
    """
    def __init__(self, dim, num_classes=1000, mlp_ratio=3, act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), drop=0.2, bias=True, use_pooling=False):
        super().__init__()
        self.use_pooling = use_pooling
        hidden_features = int(mlp_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden_features, bias=bias)
        self.act = act_layer()
        self.norm = norm_layer(hidden_features)
        self.fc2 = nn.Linear(hidden_features, num_classes, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        if self.use_pooling:
            x = x.mean((2, 3)) # global average pooling
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def make_class_layer(num_ftrs, num_class, drop_rate=0.2):
    return nn.Sequential(nn.Linear(num_ftrs, num_class), nn.Dropout(drop_rate), )


def get_num_feats(model_name, model_ft, input_tensor):
    res = ext_features(model_name, model_ft, input_tensor)
    num_feats = res.shape[-1]
    return num_feats

def ext_features(model_name, model_ft, input_tensor):

    if 'convnextv2' in model_name:
        temp = model_ft.forward_features(input_tensor)
        res = temp.flatten(2).mean(-1)
    elif 'davit_t' in model_name:
        temp = model_ft.forward_features(input_tensor)
        res = temp.flatten(2).mean(-1)
    elif 'edgeformer_s' in model_name:
        model_ft.classifier.fc = nn.Identity()
        #model_ft.conv_1x1_exp = nn.Identity()
        #model_ft.conv_1x1_exp = nn.Identity()
        #res = model_ft(input_tensor)
        #!todo?
        #res = torch.mean(res, dim=[-2, -1], keepdim=False)
        #res = model_ft.extract_features(input_tensor)
        res = model_ft.extract_end_points_l4(input_tensor)['out_l1']
        res = torch.mean(res, dim=[-2, -1], keepdim=False)
    elif 'edgenext' in model_name:
        res = model_ft.forward_features(input_tensor)
    elif 'efficientformerv2' in model_name:
        temp = model_ft.patch_embed(input_tensor)
        temp = model_ft.forward_tokens(temp)
        temp = model_ft.norm(temp)
        res = temp.flatten(2).mean(-1)
        #!todo
        #temp = model_ft.norm(temp)
        #res = model_ft.forward(input_tensor)    #temp.flatten(2).mean(-1)
    elif 'efficientnetv2' in model_name:
        temp = model_ft.forward_features(input_tensor)
        #temp = model_ft.global_pool(temp)
        res = temp.flatten(2).mean(-1)
    elif 'mobvit3' in model_name:
        #temp = model_ft._extract_features(input_tensor)
        #res = temp.flatten(2).mean(-1)
        model_ft.classifier.fc = nn.Identity()
        #model_ft.conv_1x1_exp = nn.Identity()
        res = model_ft(input_tensor)
    elif 'next_vit' in model_name:
        res = model_ft.forward_features(input_tensor)
    elif 'tiny_vit' in model_name:
        model_ft.norm_head = nn.Identity()
        model_ft.head = nn.Identity()
        #!todo
        #res = model_ft(input_tensor)
        res = model_ft.forward_features(input_tensor)
    elif "resnet" in model_name:
        model_ft.fc = nn.Identity()
        res = model_ft(input_tensor)
    elif "swiftformer" in model_name:
        temp = model_ft.patch_embed(input_tensor)
        temp = model_ft.forward_tokens(temp)
        #res = res.mean(-2)
        res = temp.flatten(2).mean(-1)
    elif "mobileone_s0_unfused" in model_name:
        model_ft.linear = nn.Identity()
        res = model_ft(input_tensor)
        #temp = model_ft.forward_tokens(temp)
        #res = res.mean(-2)
        #res = temp.flatten(2).mean(-1)
    elif "fasternet_t0" in model_name:
        model_ft.head = nn.Identity()
        res = model_ft(input_tensor)
    elif "cas-vit-xs" in model_name:
        model_ft.dist_head = nn.Identity()
        model_ft.head = nn.Identity()
        res = model_ft(input_tensor)
    elif "fastervit_0_224_1k" in model_name:
        model_ft.head = nn.Identity()
        #res = model_ft.patch_embed(input_tensor)
        res = model_ft(input_tensor)
    elif "repvit_m0_9_distill_450e" in model_name:
        model_ft.classifier = nn.Identity()
        res = model_ft(input_tensor)
    elif "hiera_tiny_224" in model_name:
        model_ft.head.projection = nn.Identity()
        res = model_ft(input_tensor)
    elif "flexivit_small.1200ep_in1k" in model_name:
        res = model_ft(input_tensor)
    elif 'inceptionnext_tiny' in model_name:
        model_ft.head = nn.Identity()
        temp = model_ft(input_tensor)
        res = temp.mean((2,3), keepdim=False)
        #t_max = temp.amax((2,3), keepdim=False)
        #print(t_max.shape)
        #res = torch.cat((t_avg, t_max), -1)
    return res

def make_fe_from_classification_model(model_name, model_ft, num_class, feat_ext=False, drop_rate=0.2):
    """
    :param model_name:
    :param model_ft:
    :param num_class:
    :return:
    """
    if 'convnextv2' in model_name:
        set_parameter_requires_grad(model_ft, feat_ext)
        model_ft.head.reset(num_classes=num_class, global_pool='avg')
    elif 'davit_t' in model_name:
        set_parameter_requires_grad(model_ft, feat_ext)
        model_ft.head.reset(num_classes=num_class, global_pool='avg')
    elif 'edgeformer_s' in model_name:
        set_parameter_requires_grad(model_ft, feat_ext)
        num_ftrs = model_ft.classifier.fc.in_features
        model_ft.classifier.fc = make_class_layer(num_ftrs, num_class, drop_rate=drop_rate)
    elif 'edgenext' in model_name:
        set_parameter_requires_grad(model_ft, feat_ext)
        num_ftrs = model_ft.head.in_features
        model_ft.head = make_class_layer(num_ftrs, num_class, drop_rate=drop_rate)
    elif 'efficientformerv2' in model_name:
        set_parameter_requires_grad(model_ft, feat_ext)
        num_ftrs = model_ft.head.in_features
        model_ft.dist = False
        model_ft.head_dist = nn.Identity()
        #model_ft.head_dist = nn.Linear(num_ftrs, num_class)
        model_ft.head = make_class_layer(num_ftrs, num_class, drop_rate=drop_rate)
    elif 'efficientnetv2' in model_name:
        set_parameter_requires_grad(model_ft, feat_ext)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = make_class_layer(num_ftrs, num_class, drop_rate=drop_rate)
    elif 'mobvit3' in model_name:
        set_parameter_requires_grad(model_ft, feat_ext)
        num_ftrs = model_ft.classifier.fc.in_features
        model_ft.classifier.fc = make_class_layer(num_ftrs, num_class, drop_rate=drop_rate)
    elif 'next_vit' in model_name:
        set_parameter_requires_grad(model_ft, feat_ext)
        num_ftrs = model_ft.proj_head[0].in_features
        model_ft.proj_head = make_class_layer(num_ftrs, num_class, drop_rate=drop_rate)
    elif 'tiny_vit_21m' in model_name:
        set_parameter_requires_grad(model_ft, feat_ext)
        num_ftrs = model_ft.head.in_features
        model_ft.head = make_class_layer(num_ftrs, num_class, drop_rate=drop_rate)
    elif "resnet" in model_name:
        set_parameter_requires_grad(model_ft, feat_ext)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = make_class_layer(num_ftrs, num_class, drop_rate=drop_rate)
    elif "swiftformer" in model_name:
        set_parameter_requires_grad(model_ft, feat_ext)
        num_ftrs = model_ft.head.in_features
        model_ft.head = make_class_layer(num_ftrs, num_class, drop_rate=drop_rate)
        model_ft.dist_head = nn.Identity()
    elif "mobileone_s0_unfused" in model_name:
        set_parameter_requires_grad(model_ft, feat_ext)
        num_ftrs = model_ft.in_planes
        model_ft.linear = make_class_layer(num_ftrs, num_class, drop_rate=drop_rate)
    elif "fasternet_t0" in model_name:
        set_parameter_requires_grad(model_ft, feat_ext)
        num_ftrs = model_ft.head.in_features
        model_ft.head = make_class_layer(num_ftrs, num_class, drop_rate=drop_rate)
    elif "cas-vit-xs" in model_name:
        set_parameter_requires_grad(model_ft, feat_ext)
        num_ftrs = model_ft.head.in_features
        model_ft.head = make_class_layer(num_ftrs, num_class, drop_rate=drop_rate)
        #model_ft.dist_head = make_class_layer(num_ftrs, num_class, drop_rate=drop_rate)
    elif "fastervit_0_224_1k" in model_name:
        set_parameter_requires_grad(model_ft, feat_ext)
        num_ftrs = model_ft.head.in_features
        model_ft.head = make_class_layer(num_ftrs, num_class, drop_rate=drop_rate)
    elif "repvit_m0_9_distill_450e" in model_name:
        set_parameter_requires_grad(model_ft, feat_ext)
        num_ftrs = model_ft.classifier.classifier.l.in_features
        model_ft.classifier.classifier.l = make_class_layer(num_ftrs, num_class, drop_rate=drop_rate)
    elif "hiera_tiny_224" in model_name:
        set_parameter_requires_grad(model_ft, feat_ext)
        num_ftrs = model_ft.head.projection.in_features
        model_ft.head.projection = make_class_layer(num_ftrs, num_class, drop_rate=drop_rate)
    elif "flexivit_small.1200ep_in1k" in model_name:
        num_ftrs = model_ft.num_features
        model_ft.head = make_class_layer(num_ftrs, num_class, drop_rate=drop_rate)
    elif "inceptionnext_tiny" in model_name:
        num_ftrs = model_ft.num_features
        model_ft.head = MLP(num_ftrs, num_class, drop=drop_rate, use_pooling=True)
    return model_ft

def test():
    from src.utils.utils_img import tensor_from_img
    num_class = 10
    model_name = 'tiny_vit_21m'
    model = load_models(model_name)
    print(model)
    model.eval()
    model.to('cuda')
    img_tensor = tensor_from_img(r'/home/servml/Документы/code/choose_bb/in/data/objectdet_crop/images/train/alarm_clock/0acee8f2d41849d.png')
    print(img_tensor.shape)

    img_tensor = torch.rand((16, 3, 224, 224))
    #img_tensor = torch.rand((16, 3, 240, 240))
    #input = img_tensor[None, :, :, :]
    img_tensor = img_tensor.to('cuda')
    print(img_tensor.shape)
    res = ext_features(model_name, model, img_tensor)
    #model = make_fe_from_classification_model(model_name, model, num_class)
    #model.to('cuda')
    #res = model(img_tensor)
    print(res.shape)
    print(res)

if __name__ == "__main__":
    test()