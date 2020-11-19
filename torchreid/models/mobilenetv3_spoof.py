"""
 Copyright (c) 2020 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.

 The initial implementation is taken from  https://github.com/d-li14/mobilenetv3.pytorch (MIT License)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchreid.losses import AngleSimpleLinear


class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0):

        super().__init__()
        self.theta = theta
        self.bias = bias or None
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        if self.groups > 1:
            self.weight = nn.Parameter(kaiming_init(out_channels, in_channels//in_channels, kernel_size))
        else:
            self.weight = nn.Parameter(kaiming_init(out_channels, in_channels, kernel_size))
        self.padding = padding
        self.i = 0

    def forward(self, x):
        out_normal = F.conv2d(input=x, weight=self.weight, bias=self.bias, dilation=self.dilation,
                              stride=self.stride, padding=self.padding, groups=self.groups)
        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            kernel_diff = self.weight.sum(dim=(2,3), keepdim=True)
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.bias, dilation=self.dilation,
                                stride=self.stride, padding=0, groups=self.groups)
            return out_normal - self.theta * out_diff

def kaiming_init(c_out, c_in, k):
    return torch.randn(c_out, c_in, k, k)*math.sqrt(2./c_in)


class Dropout(nn.Module):
    DISTRIBUTIONS = ['bernoulli', 'gaussian', 'none']

    def __init__(self, p=0.5, mu=0.5, sigma=0.3, dist='bernoulli', linear=False):
        super().__init__()

        self.dist = dist
        assert self.dist in Dropout.DISTRIBUTIONS

        self.p = float(p)
        assert 0. <= self.p <= 1.

        self.mu = float(mu)
        self.sigma = float(sigma)
        assert self.sigma > 0.
        # need to distinct 2d and 1d dropout
        self.linear = linear

    def forward(self, x):
        if self.dist == 'bernoulli' and not self.linear:
            out = F.dropout2d(x, self.p, self.training)
        elif self.dist == 'bernoulli' and self.linear:
            out = F.dropout(x, self.p, self.training)
        elif self.dist == 'gaussian':
            if self.training:
                with torch.no_grad():
                    soft_mask = x.new_empty(x.size()).normal_(self.mu, self.sigma).clamp_(0., 1.)

                scale = 1. / self.mu
                out = scale * soft_mask * x
            else:
                out = x
        else:
            out = x

        return out


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.ReLU = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.ReLU(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, make_divisible(channel // reduction, 8)),
                nn.PReLU(),
                nn.Linear(make_divisible(channel // reduction, 8), channel),
                h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_in(inp, oup, stride, theta):
    return nn.Sequential(
        Conv2d_cd(inp, oup, 3, stride, 1, bias=False, theta=theta),
        nn.InstanceNorm2d(oup),
        h_swish()
    )

def conv_3x3_bn(inp, oup, stride, theta):
    return nn.Sequential(
        Conv2d_cd(inp, oup, 3, stride, 1, bias=False, theta=theta),
        nn.BatchNorm2d(oup),
        h_swish()
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )

def conv_1x1_in(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.InstanceNorm2d(oup),
        h_swish()
    )

def make_divisible(v, divisor, min_value=None):
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


class MobileNet(nn.Module):
    """parent class for mobilenets"""
    def __init__(self, num_classes, width_mult=1., prob_dropout=0.1, type_dropout="bernoulli",
                 prob_dropout_linear=0, embeding_dim=256, mu=0.5, sigma=0.3,
                 theta=0, multi_heads=False, feature=False, loss='softmax', contrastive = False,
                 classification = False,  **kwargs):
        super().__init__()
        self.prob_dropout = prob_dropout
        self.type_dropout = type_dropout
        self.num_classes = num_classes
        self.classification = classification
        self.contrastive = contrastive
        self.width_mult = width_mult
        self.prob_dropout_linear = prob_dropout_linear
        self.embeding_dim = embeding_dim
        self.mu = mu
        self.loss = loss
        self.sigma = sigma
        self.theta = theta
        self.multi_heads = multi_heads
        self.features = nn.Identity
        self.feature = feature

        # building last several layers
        self.conv_last = nn.Identity
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(embeding_dim, 2)

    def forward(self, x, return_featuremaps=False, get_embeddings=False):
        x = self.features(x)
        if return_featuremaps:
            return x
        x = self.conv_last(x)
        x = self.avgpool(x)

        if self.feature or not self.training:
            return x

        x = x.view(x.size(0), -1)
        y = self.classifier(x)

        if get_embeddings:
            return y, x

        if self.loss in ['softmax', 'am_softmax']:
            return y
        elif self.loss in ['triplet', ]:
            return y, x
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

        x = self.features(x)
        x = self.conv_last(x)
        x = self.avgpool(x)
        return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride,
                 use_se, use_hs, prob_dropout, type_dropout, sigma, mu):
        super().__init__()
        assert stride in [1, 2]
        self.identity = stride == 1 and inp == oup
        self.dropout2d = Dropout(dist=type_dropout, mu=mu ,
                                 sigma=sigma,
                                 p=prob_dropout)
        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride,
                         (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.PReLU(),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.PReLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride,
                         (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                h_swish() if use_hs else nn.PReLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.dropout2d(self.conv(x))
        else:
            return self.dropout2d(self.conv(x))


class MobileNetV3(MobileNet):
    def __init__(self, cfgs, mode='large', **kwargs):
        super().__init__(**kwargs)
        self.cfgs = cfgs
        # setting of inverted residual blocks
        assert mode in ['large', 'small']
        # building first layer
        input_channel = make_divisible(16 * self.width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2, theta=self.theta)]
        # building inverted residual blocks
        block = InvertedResidual
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = make_divisible(c * self.width_mult, 8)
            exp_size = make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs,
                                                                prob_dropout=self.prob_dropout,
                                                                mu=self.mu,
                                                                sigma=self.sigma,
                                                                type_dropout=self.type_dropout))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)
        self.conv_last = conv_1x1_bn(input_channel, self.embeding_dim)

        if not self.feature:
            classifier_block = nn.Linear if self.loss not in ['am_softmax'] else AngleSimpleLinear
            self.classifier = classifier_block(self.embeding_dim, self.num_classes)


def mobilenetv3_large_spoof(**kwargs):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s
        [3,   1,  16, 0, 0, 1],
        [3,   4,  24, 0, 0, 2],
        [3,   3,  24, 0, 0, 1],
        [5,   3,  40, 1, 0, 2],
        [5,   3,  40, 1, 0, 1],
        [5,   3,  40, 1, 0, 1],
        [3,   6,  80, 0, 1, 2],
        [3, 2.5,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [5,   6, 160, 1, 1, 2],
        [5,   6, 160, 1, 1, 1],
        [5,   6, 160, 1, 1, 1]
    ]
    return MobileNetV3(cfgs, mode='large', **kwargs)

def mobilenetv3_small(**kwargs):
    """
    Constructs a MobileNetV3-Small model
    """
    cfgs = [
        # k, t, c, SE, HS, s
        [3,    1,  16, 1, 0, 2],
        [3,  4.5,  24, 0, 0, 2],
        [3, 3.67,  24, 0, 0, 1],
        [5,    4,  40, 1, 1, 2],
        [5,    6,  40, 1, 1, 1],
        [5,    6,  40, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    6,  96, 1, 1, 2],
        [5,    6,  96, 1, 1, 1],
        [5,    6,  96, 1, 1, 1],
    ]

    return MobileNetV3(cfgs, mode='small', **kwargs)
