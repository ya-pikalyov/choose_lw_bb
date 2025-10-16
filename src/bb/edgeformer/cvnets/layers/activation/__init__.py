import os
import importlib
import argparse

SUPPORTED_ACT_FNS = []


def register_act_fn(name):
    def register_fn(fn):
        if name in SUPPORTED_ACT_FNS:
            raise ValueError("Cannot register duplicate activation function ({})".format(name))
        SUPPORTED_ACT_FNS.append(name)
        return fn
    return register_fn


def arguments_activation_fn(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Non-linear functions", description="Non-linear functions")

    group.add_argument('--model.activation.name', default='relu', type=str, help='Non-linear function type')
    group.add_argument('--model.activation.inplace', action='store_true', help='Inplace non-linear functions')
    group.add_argument('--model.activation.neg-slope', default=0.1, type=float, help='Negative slope in leaky relu')

    return parser


# import later to avoid circular loop
from src.bb.edgeformer.cvnets.layers.activation.gelu import GELU
from src.bb.edgeformer.cvnets.layers.activation.hard_sigmoid import Hardsigmoid
from src.bb.edgeformer.cvnets.layers.activation.hard_swish import Hardswish
from src.bb.edgeformer.cvnets.layers.activation.leaky_relu import LeakyReLU
from src.bb.edgeformer.cvnets.layers.activation.prelu import PReLU
from src.bb.edgeformer.cvnets.layers.activation.relu import ReLU
from src.bb.edgeformer.cvnets.layers.activation.relu6 import ReLU6
from src.bb.edgeformer.cvnets.layers.activation.sigmoid import Sigmoid
from src.bb.edgeformer.cvnets.layers.activation.swish import Swish


__all__ = [
    'GELU',
    'Hardsigmoid',
    'Hardswish',
    'LeakyReLU',
    'PReLU',
    'ReLU',
    'ReLU6',
    'Sigmoid',
    'Swish',
]
