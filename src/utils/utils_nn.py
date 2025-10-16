import torch
import torch.nn.functional
from torch import Tensor
from torch.nn import Module as BaseModule
from typing import Sequence
from functools import reduce
from operator import mul
import warnings
import math
import time
import torch.nn as nn
from src.utils.utils_types import Union, Tuple

from src.utils.utils_types import to_2tuple
from thop import profile, clever_format
from enum import Enum
from typing import Union

from fvcore.nn import FlopCountAnalysis, parameter_count_table



class Format(str, Enum):
    NCHW = 'NCHW'
    NHWC = 'NHWC'
    NCL = 'NCL'
    NLC = 'NLC'

FormatT = Union[str, Format]

def get_spatial_dim(fmt: FormatT):
    fmt = Format(fmt)
    if fmt is Format.NLC:
        dim = (1,)
    elif fmt is Format.NCL:
        dim = (2,)
    elif fmt is Format.NHWC:
        dim = (1, 2)
    else:
        dim = (2, 3)
    return dim


def get_channel_dim(fmt: FormatT):
    fmt = Format(fmt)
    if fmt is Format.NHWC:
        dim = 3
    elif fmt is Format.NLC:
        dim = 2
    else:
        dim = 1
    return dim

def count_params_flops_fvcore(model, input_size=(1, 3, 256, 256)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    dummy_input = torch.randn(input_size).to(device)

    # FLOPs
    flops = FlopCountAnalysis(model, dummy_input)
    print(f"Input size: {input_size}")
    print(f"Total FLOPs: {flops.total() / 1e9:.3f} GFLOPs")

    # Parameters
    print(parameter_count_table(model))


def measure_fps(model, inputs, num_warmup=10, num_iters=50, device="cuda"):
    """
    Measure inference FPS of a PyTorch model.

    Args:
        model: torch.nn.Module
        inputs: example input tensor (B, C, H, W)
        num_warmup: number of warm-up runs (ignored in timing)
        num_iters: number of iterations to average
        device: "cuda" or "cpu"
    """
    model = model.to(device).eval()
    inputs = inputs.to(device)

    # Warm-up (for stable GPU clocks)
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(inputs)
    if device == "cuda":
        torch.cuda.synchronize()

    # Timing loop
    start = time.time()
    with torch.no_grad():
        for _ in range(num_iters):
            _ = model(inputs)
            if device == "cuda":
                torch.cuda.synchronize()
    end = time.time()

    elapsed = end - start
    fps = num_iters / elapsed
    return fps


def set_enn_plain_profiling(module: nn.Module, enabled: bool = True):
    """
    Recursively enable/disable 'use_plain' on ENN wrappers so THOP can profile.
    """
    for m in module.modules():
        if hasattr(m, "use_plain"):
            m.use_plain = enabled

def count_params_flops_thop(model, input_size=(1, 3, 256, 256)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dummy_input = torch.randn(input_size).to(device)

    # FLOPs and params
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")

    print(f"Model: {model.__class__.__name__}")
    print(f"Params: {params}")
    print(f"FLOPs : {flops} for input {input_size}")

    return params, flops

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

def linear_expectation(probs, values):
    assert(len(values) == probs.ndimension() - 2)
    expectation = []

    for i in range(2, probs.ndimension()):
        # Marginalise probabilities
        marg = probs
        for j in range(probs.ndimension() - 1, 1, -1):
            if i != j:
                marg = marg.sum(j, keepdim=False)
        # Calculate expectation along axis `i`
        expectation.append((marg * values[len(expectation)]).sum(-1, keepdim=False))
    return torch.stack(expectation, -1)


TORCH_VERSION = torch.__version__

def get_norm(name_norm):

    from torch.nn.modules.batchnorm import _BatchNorm
    from torch.nn.modules.instancenorm import _InstanceNorm


    assert name_norm in ['BN', 'SyncBN', 'InNorm'], 'Wrong type of norm'

    if name_norm == 'SyncBN':
        return torch.nn.SyncBatchNorm
    elif name_norm == 'InNorm':
        return _InstanceNorm
    elif name_norm == 'BN':
        return _BatchNorm



def build_norm_layer(name_norm, requires_grad,
                     num_features: int,
                     postfix: Union[int, str] = '') -> Tuple[str, nn.Module]:
    """Build normalization layer.

    Args:
        cfg (dict): The norm layer config, which should contain:

            - type (str): Layer type.
            - layer args: Args needed to instantiate a norm layer.
            - requires_grad (bool, optional): Whether stop gradient updates.
        num_features (int): Number of input channels.
        postfix (int | str): The postfix to be appended into norm abbreviation
            to create named layer.

    Returns:
        tuple[str, nn.Module]: The first element is the layer name consisting
        of abbreviation and postfix, e.g., bn1, gn. The second element is the
        created norm layer.
    """

    eps = 1e-5

    norm_layer = get_norm(name_norm=name_norm)
    layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return layer


def normalized_linspace(length, dtype=None, device=None):
    """Generate a vector with values ranging from -1 to 1.

    Note that the values correspond to the "centre" of each cell, so
    -1 and 1 are always conceptually outside the bounds of the vector.
    For example, if length = 4, the following vector is generated:

    ```text
     [ -0.75, -0.25,  0.25,  0.75 ]
     ^              ^             ^
    -1              0             1
    ```

    Args:
        length: The length of the vector

    Returns:
        The generated vector
    """
    if isinstance(length, torch.Tensor):
        length = length.to(device, dtype)
    first = -(length - 1.0) / length
    return torch.arange(length, dtype=dtype, device=device) * (2.0 / length) + first


def soft_argmax(heatmaps, normalized_coordinates=True):
    if normalized_coordinates:
        values = [normalized_linspace(d, dtype=heatmaps.dtype, device=heatmaps.device)
                  for d in heatmaps.size()[2:]]
    else:
        values = [torch.arange(0, d, dtype=heatmaps.dtype, device=heatmaps.device)
                  for d in heatmaps.size()[2:]]
    coords = linear_expectation(heatmaps, values)
    # We flip the tensor like this instead of using `coords.flip(-1)` because aten::flip is not yet
    # supported by the ONNX exporter.

    coords = torch.cat(tuple(reversed(coords.split(1, -1))), -1)
    return coords


def dsnt(heatmaps, **kwargs):
    """Differentiable spatial to numerical transform.

    Args:
        heatmaps (torch.Tensor): Spatial representation of locations

    Returns:
        Numerical coordinates corresponding to the locations in the heatmaps.
    """
    return soft_argmax(heatmaps, **kwargs)


def sharpen_heatmaps(heatmaps, alpha):
    """Sharpen heatmaps by increasing the contrast between high and low probabilities.

    Example:
        Approximate the mode of heatmaps using the approach described by Equation 1 of
        "FlowCap: 2D Human Pose from Optical Flow" by Romero et al.)::

            coords = soft_argmax(sharpen_heatmaps(heatmaps, alpha=6))

    Args:
        heatmaps (torch.Tensor): Heatmaps generated by the model
        alpha (float): Sharpness factor. When ``alpha == 1``, the heatmaps will be unchanged. Use
        ``alpha > 1`` to actually sharpen the heatmaps.

    Returns:
        The sharpened heatmaps.
    """
    sharpened_heatmaps = heatmaps ** alpha
    sharpened_heatmaps /= sharpened_heatmaps.flatten(2).sum(-1)
    return sharpened_heatmaps


def flat_softmax(inp):
    """Compute the softmax with all but the first two tensor dimensions combined."""

    orig_size = inp.size()
    flat = inp.view(-1, reduce(mul, orig_size[2:]))
    flat = torch.nn.functional.softmax(flat, -1)
    return flat.view(*orig_size)




def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def adjust_channels(channels: int, width_mult: float):
    return _make_divisible(channels * width_mult, 8)


def count_parameters(model, requires_grad=False):
    if requires_grad:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

def concatenate_tensors_with_diff_size_kernels(tensor1, tensor2):
    # Create example tensors
    k1 = tensor1.shape[2]
    k2 = tensor2.shape[2]
    # Calculate padding for tensor_b to match tensor_a's spatial dimensions
    pad_height = (k1 - k2) // 2
    pad_width = (k1 - k2) // 2
    # Pad tensor_b
    padded_tensor_b = torch.nn.functional.pad(tensor2, (pad_width, pad_width, pad_height, pad_height), mode='constant', value=0)
    # Concatenate along the channel dimension (dim=1)
    concatenated_tensor = torch.cat((tensor1, padded_tensor_b), dim=1)
    return concatenated_tensor

def concatenate_tensors_with_diff_size_kernels_v2(tensor1, tensor2):
    # Create example tensors
    k1 = tensor1.shape[2]
    k2 = tensor2.shape[2]

    # Resize tensor_b to match tensor_a's spatial dimensions
    resized_tensor_b = F.interpolate(tensor2, size=(k1, k1), mode='bilinear', align_corners=False)
    # Concatenate along the channel dimension (dim=1)
    concatenated_tensor = torch.cat((tensor1, resized_tensor_b), dim=1)
    return concatenated_tensor


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p



def trunc_normal_init(
        module: nn.Module,
        mean: float = 0,
        std: float = 1,
        a: float = -2,
        b: float = 2,
        bias: float = 0,
) -> None:
    if hasattr(module, "weight") and module.weight is not None:
        # trunc_normal_(module.weight, mean, std, a, b)  # type: ignore
        _no_grad_trunc_normal_(module.weight, mean, std, a, b)  # type: ignore
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)  # type: ignore


def _no_grad_trunc_normal_(
        tensor: Tensor, mean: float, std: float, a: float, b: float
) -> Tensor:
    # Method based on
    # https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    # Modified from
    # https://github.com/pytorch/pytorch/blob/master/torch/nn/init.py
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [lower, upper], then translate
        # to [2lower-1, 2upper-1].
        tensor.uniform_(2 * lower - 1, 2 * upper - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(
        tensor: Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0
) -> Tensor:
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    Modified from
    https://github.com/pytorch/pytorch/blob/master/torch/nn/init.py

    Args:
        tensor (``torch.Tensor``): an n-dimensional `torch.Tensor`.
        mean (float): the mean of the normal distribution.
        std (float): the standard deviation of the normal distribution.
        a (float): the minimum cutoff value.
        b (float): the maximum cutoff value.
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def constant_init(module, val, bias=0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def uniform_init(module, a=0, b=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.uniform_(module.weight, a, b)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def build_norm_layer(norm_cfg, embed_dims):
    assert norm_cfg["type"] == "LN"
    norm_layer = nn.LayerNorm(embed_dims)
    return norm_cfg["type"], norm_layer


class GELU(nn.Module):
    r"""Applies the Gaussian Error Linear Units function:

    .. math::
        \text{GELU}(x) = x * \Phi(x)
    where :math:`\Phi(x)` is the Cumulative Distribution Function for
    Gaussian Distribution.

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    .. image:: scripts/activation_images/GELU.png

    Examples::

        >>> m = nn.GELU()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def forward(self, input):
        return F.gelu(input)


def build_activation_layer(act_cfg):
    if act_cfg["type"] == "ReLU":
        act_layer = nn.ReLU(inplace=act_cfg["inplace"])
    elif act_cfg["type"] == "GELU":
        act_layer = GELU()
    return act_layer


def build_conv_layer(
        conv_cfg, in_channels, out_channels, kernel_size, stride, padding, dilation, bias
):
    conv_layer = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
    )
    return conv_layer


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of
    residual blocks).

    We follow the implementation
    https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py  # noqa: E501

    Args:
        drop_prob (float): Probability of the path to be zeroed. Default: 0.1
    """

    def __init__(self, drop_prob=0.1):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def drop_path(x, drop_prob=0.0, training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of
    residual blocks).

    We follow the implementation
    https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py  # noqa: E501
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    # handle tensors with different dimensions, not just 4D tensors.
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    output = x.div(keep_prob) * random_tensor.floor()
    return output


def build_dropout(drop_cfg):
    drop_layer = DropPath(drop_cfg["drop_prob"])
    return drop_layer

