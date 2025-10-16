import timm


def flexi_vit_tiny(model_name='flexivit_small.1200ep_in1k', pretrained=False, **kwargs):
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, )
    return model




'''

output = model(transforms(img).unsqueeze(0))  # output is (batch_size, num_features) shaped tensor
# or equivalently (without needing to set num_classes=0)
output = model.forward_features(transforms(img).unsqueeze(0))
# output is unpooled, a (1, 226, 384) shaped tensor
output = model.forward_head(output, pre_logits=True)
# output is a (1, num_features) shaped tensor

'''


