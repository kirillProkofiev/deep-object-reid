"""
 MIT License

 Copyright (c) 2018 Kaiyang Zhou

 Copyright (c) 2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from __future__ import absolute_import
from __future__ import division

import warnings
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import grad
import numpy as np

from torchreid.models.osnet import OSNet, ConvLayer, LightConv3x3, Conv1x1Linear, \
                                   ChannelGate, Conv1x1, pretrained_urls
from torchreid.losses import AngleSimpleLinear
from torchreid.ops import Dropout, FPN, GeneralizedMeanPooling


__all__ = ['fpn_osnet_x1_0', 'fpn_osnet_x0_75', 'fpn_osnet_x0_5', 'fpn_osnet_x0_25', 'fpn_osnet_ibn_x1_0']

pretrained_urls_fpn = {
    'fpn_osnet_x1_0': pretrained_urls['osnet_x1_0'],
    'fpn_osnet_x0_75': pretrained_urls['osnet_x0_75'],
    'fpn_osnet_x0_5': pretrained_urls['osnet_x0_5'],
    'fpn_osnet_x0_25': pretrained_urls['osnet_x0_25'],
    'fpn_osnet_ibn_x1_0': pretrained_urls['osnet_ibn_x1_0']
}


__RSC_MODES__ = ['overall', 'channelwise', 'spatial']

def rsc(features, scores, labels, retain_p=0.77, mode='overall'):
    """Representation Self-Challenging module (RSC).
       Based on the paper: https://arxiv.org/abs/2007.02454
    """
    assert mode in __RSC_MODES__
    batch_range = torch.arange(scores.size(0), device=scores.device)
    gt_scores = scores[batch_range, labels.view(-1)]
    z_grads = grad(outputs=gt_scores,
                   inputs=features,
                   grad_outputs=torch.ones_like(gt_scores),
                   create_graph=True)[0]
    with torch.no_grad():
        z_grads_cpu = z_grads.cpu().numpy()
        if mode == __RSC_MODES__[0]:
            axes = (1, 2, 3)
        elif mode == __RSC_MODES__[1]:
            axes = (2, 3)
        elif mode == __RSC_MODES__[2]:
            axes = (1,)
        z_grad_thresholds_cpu = np.quantile(z_grads_cpu, retain_p, axis=axes, keepdims=True)
        zero_mask = z_grads > torch.from_numpy(z_grad_thresholds_cpu).to(z_grads.device)
        unchanged_mask = torch.randint(2, [z_grads.size(0)], dtype=torch.bool, device=z_grads.device)
        unchanged_mask = unchanged_mask.view(-1, 1, 1, 1)

    scale = 1.0 / float(retain_p)
    filtered_features = scale * torch.where(zero_mask, torch.zeros_like(features), features)
    out_features = torch.where(unchanged_mask, features, filtered_features)

    return out_features


class LCTGate(nn.Module):
    def __init__(self, channels, groups=16):
        super(LCTGate, self).__init__()
        assert channels > 0
        assert groups > 0
        while channels % groups != 0:
            groups //= 2
        self.gn = nn.GroupNorm(groups, channels, affine=True)
        nn.init.ones_(self.gn.bias)
        nn.init.zeros_(self.gn.weight)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.gate_activation = nn.Sigmoid()

    def forward(self, x):
        input = x
        x = self.global_avgpool(x)
        x = self.gn(x)
        x = self.gate_activation(x)
        return input * x


class OSBlock(nn.Module):
    """Omni-scale feature learning block."""

    def __init__(self, in_channels, out_channels, IN=False, bottleneck_reduction=4,
                 dropout_cfg=None, channel_gate=ChannelGate, **kwargs):
        super(OSBlock, self).__init__()
        mid_channels = out_channels // bottleneck_reduction
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2a = LightConv3x3(mid_channels, mid_channels)
        self.conv2b = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2c = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2d = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.gate = channel_gate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)
        self.IN = None
        if IN:
            self.IN = nn.InstanceNorm2d(out_channels, affine=True)
        self.dropout = None
        if dropout_cfg is not None:
            self.dropout = Dropout(**dropout_cfg)

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2a = self.conv2a(x1)
        x2b = self.conv2b(x1)
        x2c = self.conv2c(x1)
        x2d = self.conv2d(x1)
        x2 = self.gate(x2a) + self.gate(x2b) + self.gate(x2c) + self.gate(x2d)
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        if self.dropout:
            x3 = self.dropout(x3)
        out = x3 + identity
        if self.IN is not None:
            out = self.IN(out)
        return F.relu(out)


class OSNetFPN(OSNet):
    """Omni-Scale Network.

    Reference:
        - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
    """

    def __init__(self, num_classes, blocks, layers, channels,
                 feature_dim=256,
                 loss='softmax',
                 instance_norm=False,
                 dropout_cfg=None,
                 fpn_cfg=None,
                 pooling_type='avg',
                 input_size=(256, 128),
                 IN_first=False,
                 extra_blocks=False,
                 lct_gate=False,
                 rsc_cfg=None,
                 **kwargs):
        self.dropout_cfg = dropout_cfg
        self.rsc_cfg = rsc_cfg
        self.extra_blocks = extra_blocks
        self.channel_gate = LCTGate if lct_gate else ChannelGate
        if self.extra_blocks:
            for i, l in enumerate(layers):
                layers[i] = l + 1
        super(OSNetFPN, self).__init__(num_classes, blocks, layers, channels, feature_dim, loss, instance_norm)

        self.feature_scales = (4, 8, 16, 16)
        if fpn_cfg is not None:
            self.fpn_enable = fpn_cfg.enable
            self.fpn_dim = fpn_cfg.dim
            self.fpn_process = fpn_cfg.process
            assert self.fpn_process in ['concatenation', 'max_pooling', 'elementwise_sum']
        else:
            self.fpn_enable = False
        self.feature_dim = feature_dim

        self.use_IN_first = IN_first
        if IN_first:
            self.in_first = nn.InstanceNorm2d(3, affine=True)
            self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3, IN=self.use_IN_first)

        if self.fpn_enable:
            self.fpn = FPN(channels, self.feature_scales, self.fpn_dim, self.fpn_dim)
            fpn_out_dim = self.fpn_dim if self.fpn_process in ['max_pooling', 'elementwise_sum'] \
                          else feature_dim
            self.fc = self._construct_fc_layer(feature_dim, fpn_out_dim, dropout_cfg)
        else:
            self.fpn = None
            self.fc = self._construct_fc_layer(feature_dim, channels[3], dropout_cfg)

        if self.loss not in ['am_softmax', ]:
            self.classifier = nn.Linear(self.feature_dim, num_classes)
        else:
            self.classifier = AngleSimpleLinear(self.feature_dim, num_classes)

        if 'conv' in pooling_type:
            kernel_size = (input_size[0] // self.feature_scales[-1], input_size[1] // self.feature_scales[-1])
            if self.fpn_enable:
                self.global_avgpool = nn.Conv2d(fpn_out_dim, fpn_out_dim, kernel_size, groups=fpn_out_dim)
            else:
                self.global_avgpool = nn.Conv2d(channels[3], channels[3], kernel_size, groups=channels[3])
        elif 'avg' in pooling_type:
            self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        elif 'gmp' in pooling_type:
            self.global_avgpool = GeneralizedMeanPooling()
        else:
            raise ValueError('Incorrect pooling type')

        if self.fpn_enable and self.fpn_process == 'concatenation':
            self.fpn_extra_conv = ConvLayer(self.fpn_dim * len(self.fpn.dims_out),
                                            feature_dim, 3, stride=1, padding=1, IN=False)
        else:
            self.fpn_extra_conv = None

        self._init_params()

    def _make_layer(self, block, layer, in_channels, out_channels, reduce_spatial_size, IN=False):
        layers = []

        layers.append(block(in_channels, out_channels, IN=IN))
        for i in range(1, layer):
            layers.append(block(out_channels, out_channels, IN=IN,
                                dropout_cfg=self.dropout_cfg, channel_gate=self.channel_gate))

        if reduce_spatial_size:
            layers.append(
                nn.Sequential(
                    Conv1x1(out_channels, out_channels),
                    nn.AvgPool2d(2, stride=2)
                )
            )

        return nn.Sequential(*layers)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        if fc_dims is None or fc_dims < 0:
            self.feature_dim = input_dim
            return None

        if isinstance(fc_dims, int):
            fc_dims = [fc_dims]

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            if self.loss not in ['am_softmax', ]:
                layers.append(nn.ReLU(inplace=True))
            else:
                layers.append(nn.PReLU())
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p.p))
            input_dim = dim

        self.feature_dim = fc_dims[-1]

        return nn.Sequential(*layers)

    def featuremaps(self, x):
        out = []
        if self.use_IN_first:
            x = self.in_first(x)
        x = self.conv1(x)
        x1 = self.maxpool(x)
        out.append(x1)
        x2 = self.conv2(x1)
        out.append(x2)
        x3 = self.conv3(x2)
        out.append(x3)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        out.append(x5)
        return x5, out

    def process_feature_pyramid(self, feature_pyramid):
        feature_pyramid = feature_pyramid[:-1]
        target_shape = feature_pyramid[-1].shape[2:]
        for i in range(len(feature_pyramid) - 1):
            kernel_size = int(feature_pyramid[i].shape[2] // target_shape[0])
            feature_pyramid[i] = nn.functional.max_pool2d(feature_pyramid[i], kernel_size=kernel_size)
            if self.fpn_process == 'max_pooling':
                feature_pyramid[-1] = torch.max(feature_pyramid[i], feature_pyramid[-1])
            elif self.fpn_process == 'elementwise_sum':
                feature_pyramid[-1] = torch.add(feature_pyramid[i], feature_pyramid[-1])
            else:
                feature_pyramid[-1] = torch.cat((feature_pyramid[i], feature_pyramid[-1]), dim=1)
        if self.fpn_process == 'concatenation':
            output = self.fpn_extra_conv(feature_pyramid[-1])
        else:
            output = feature_pyramid[-1]
        return output

    def head_forward(self, features, get_embeddings=False):
        v = self.global_avgpool(features)
        if isinstance(self.fc[0], nn.Linear):
            v = v.view(v.size(0), -1)

        if self.fc is not None:
            if self.training:
                v = self.fc(v)
            else:
                v = self.fc[0](v).view(v.size(0), -1, 1)
                v = self.fc[1](v)
                v = self.fc[2](v)
        v = v.view(v.size(0), -1)

        if not self.training:
            return v

        y = self.classifier(v)

        if get_embeddings:
            return y, v

        if self.loss in ['softmax', 'adacos', 'd_softmax', 'am_softmax']:
            return y
        elif self.loss in ['triplet', ]:
            return y, v
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

    def forward(self, x, trg_labels=None, return_featuremaps=False, get_embeddings=False):
        x, feature_pyramid = self.featuremaps(x)
        if self.fpn is not None:
            feature_pyramid = self.fpn(feature_pyramid)
            x = self.process_feature_pyramid(feature_pyramid)

        if return_featuremaps:
            return x

        if self.train and trg_labels is not None and self.rsc_cfg.enable:
            y, _ = self.head_forward(x, get_embeddings=True)
            x = rsc(x, y, trg_labels, 1. - self.rsc_cfg.drop_percentage, self.rsc_cfg.mode)

        return self.head_forward(x, get_embeddings)


    def load_pretrained_weights(self, pretrained_dict):
        model_dict = self.state_dict()
        new_state_dict = OrderedDict()
        matched_layers, discarded_layers = [], []

        for k, v in pretrained_dict.items():
            if k.startswith('module.'):
                k = k[7:]  # discard module.

            if k in model_dict and model_dict[k].size() == v.size():
                new_state_dict[k] = v
                matched_layers.append(k)
            else:
                discarded_layers.append(k)

        model_dict.update(new_state_dict)
        self.load_state_dict(model_dict)

        if len(matched_layers) == 0:
            warnings.warn(
                'The pretrained weights cannot be loaded, '
                'please check the key names manually '
                '(** ignored and continue **)'
            )
        else:
            print('Successfully loaded pretrained weights')
            if len(discarded_layers) > 0:
                print(
                    '** The following layers are discarded '
                    'due to unmatched keys or layer size: {}'.
                    format(discarded_layers)
                )


def init_pretrained_weights(model, key=''):
    """Initializes model with pretrained weights.

    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    import os
    import errno
    import gdown

    def _get_torch_home():
        ENV_TORCH_HOME = 'TORCH_HOME'
        ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
        DEFAULT_CACHE_DIR = '~/.cache'
        torch_home = os.path.expanduser(
            os.getenv(
                ENV_TORCH_HOME,
                os.path.join(
                    os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), 'torch'
                )
            )
        )
        return torch_home

    torch_home = _get_torch_home()
    model_dir = os.path.join(torch_home, 'checkpoints')
    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    filename = key + '_imagenet.pth'
    cached_file = os.path.join(model_dir, filename)

    if not os.path.exists(cached_file):
        gdown.download(pretrained_urls_fpn[key], cached_file, quiet=False)

    state_dict = torch.load(cached_file)
    model.load_pretrained_weights(state_dict)


def fpn_osnet_x1_0(num_classes, pretrained=False, download_weights=False, **kwargs):
    model = OSNetFPN(
        num_classes,
        blocks=[OSBlock, OSBlock, OSBlock],
        layers=[2, 2, 2],
        channels=[64, 256, 384, 512],
        **kwargs
    )

    if pretrained and download_weights:
        init_pretrained_weights(model, key='fpn_osnet_x1_0')

    return model


def fpn_osnet_x0_75(num_classes, pretrained=False, download_weights=False, **kwargs):
    model = OSNetFPN(
        num_classes,
        blocks=[OSBlock, OSBlock, OSBlock],
        layers=[2, 2, 2],
        channels=[48, 192, 288, 384],
        **kwargs
    )

    if pretrained and download_weights:
        init_pretrained_weights(model, key='fpn_osnet_x0_75')

    return model


def fpn_osnet_x0_5(num_classes, pretrained=False, download_weights=False, **kwargs):
    model = OSNetFPN(
        num_classes,
        blocks=[OSBlock, OSBlock, OSBlock],
        layers=[2, 2, 2],
        channels=[32, 128, 192, 256],
        **kwargs
    )

    if pretrained and download_weights:
        init_pretrained_weights(model, key='fpn_osnet_x0_5')

    return model


def fpn_osnet_x0_25(num_classes, pretrained=False, download_weights=False, **kwargs):
    model = OSNetFPN(
        num_classes,
        blocks=[OSBlock, OSBlock, OSBlock],
        layers=[2, 2, 2],
        channels=[16, 64, 96, 128],
        **kwargs
    )

    if pretrained and download_weights:
        init_pretrained_weights(model, key='fpn_osnet_x0_25')

    return model


def fpn_osnet_ibn_x1_0(num_classes, pretrained=False, download_weights=False, **kwargs):
    # standard size (width x1.0) + IBN layer
    model = OSNetFPN(
        num_classes,
        blocks=[OSBlock, OSBlock, OSBlock],
        layers=[2, 2, 2],
        channels=[64, 256, 384, 512],
        IN=True, **kwargs
    )

    if pretrained and download_weights:
        init_pretrained_weights(model, key='fpn_osnet_ibn_x1_0')

    return model
