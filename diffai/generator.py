import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import multinomial, categorical
import torch.optim as optim

import math

try:
    from . import helpers as h
    from . import ai
    from . import scheduling as S
    from . import components as n
except:
    import helpers as h
    import ai
    import scheduling as S
    import components as n

import math
import abc

from torch.nn.modules.conv import _ConvNd
from enum import Enum

def SemanticGenerator(latent_dim, bias = False, normal=False):
    conv_layers = [(latent_dim*2, 5, 5, 2, 'ReLU'), (1, 4, 4, 2, 'tanh')]
    def transfer(tp):
        if isinstance(tp, InferModule):
            return tp
        if isinstance(tp[0], str):
            return MaxPool2D(*tp[1:])
        return Seq(ConvTranspose2D(out_channels = tp[0], kernel_size = tp[1],
                                    stride = tp[-2] if len(tp) == 5 else 1,
                                    bias=bias, normal=normal, padding=0),
                    activation(tp[-1]))
    conv = [transfer(s) for s in conv_layers]
    return n.Seq(
            #input_shape=(Z_DIM,)
            n.FFNN([latent_dim * 7*7*4], batch_norm=True, bias=bias),
                Unflatten2d((7, 7, LATENT_DIM*4)),
                *conv)

def MyLenet():
    return n.Seq(
        n.Flatten((28, 28)),
        n.PrintActivation(),
        n.Linear(784, 128),
        n.activation('ReLU'),
        n.Dropout(128, p=0.2),
        n.Linear(128, 10),
        n.activation('softmax'))
