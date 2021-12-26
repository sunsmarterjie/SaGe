import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm
from openselfsup.utils import get_root_logger
# from ..registry import DECODERS
from .utils import build_conv_layer, build_norm_layer
import torch
import torch.nn.functional as F


class UPBlock_V1(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 up_scale=4,
                 stride=2,
                 padding=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='SyncBN')):
        super(UPBlock_V1, self).__init__()
        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = torch.nn.ConvTranspose2d(inplanes, planes, up_scale, stride, padding, bias=False)
        self.add_module(self.norm1_name, norm1)
        # self.conv2 = build_conv_layer(
        #     conv_cfg, planes, planes, 3, padding=1, bias=False
        # )
        # self.add_module(self.norm2_name, norm2)
        self.relu = nn.ReLU(True)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        # out = self.conv2(out)
        # out = self.norm2(out)
        # out = self.relu(out)
        return out


class Decoder(nn.Module):
    def __init__(self,
                 in_channels=2048,
                 style='pytorch',
                 conv_cfg=None,
                 norm_cfg=dict(type='SyncBN', requires_grad=True),
                 norm_eval=False,
                 zero_init_residual=False):
        super(Decoder, self).__init__()
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.zero_init_residual = zero_init_residual

        self.res_layers = []
        # outplanes = [2048, 1024, 512, 256, 128]
        # outplanes = [1024, 512, 256, 128, 64]
        outplanes = [512, 256, 128, 64, 32]
        # outplanes = [256, 128, 64, 32, 16]
        # outplanes = [360, 196, 96, 32, 24]
        strides = [2, 2, 2, 2, 2]
        paddings = [1, 1, 1, 1, 1]
        # paddings = [1, 1, 0, 0, 0]

        self.conv1 = nn.Conv2d(2048, 42, 1)
        self.add_module('conv1', self.conv1)
        self.bn1 = nn.BatchNorm2d(42)
        self.add_module('bn1', self.bn1)
        self.relu1 = nn.ReLU()

        neck = 512
        self.conv7 = nn.Conv2d(2048, neck, 7)
        self.add_module('conv7', self.conv7)
        self.bn7 = nn.BatchNorm2d(neck)
        self.add_module('bn7', self.bn7)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Conv2d(2048, 8192, 1)
        layer_name = 'linear'
        self.add_module(layer_name, self.linear)

        self.conv1_7 = nn.Conv2d(2048, 4, 1)
        self.add_module('conv1_7', self.conv1_7)
        self.conv1_14 = nn.Conv2d(1024, 1, 1)
        self.add_module('conv1_14', self.conv1_14)

        in_channels = 256
        self.stem = nn.Sequential(
            nn.ConvTranspose2d(neck, in_channels, kernel_size=7, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))
        layer_name = 'stem'
        self.add_module(layer_name, self.stem)

        # self.stem = nn.Conv2d(in_channels, )
        self.layer0 = UPBlock_V1(inplanes=in_channels, planes=outplanes[0], up_scale=4, stride=strides[0],
                                 padding=paddings[0])
        layer_name = 'layer{}'.format(0)
        self.add_module(layer_name, self.layer0)
        self.layer1 = UPBlock_V1(inplanes=outplanes[0], planes=outplanes[1], up_scale=4, stride=strides[1],
                                 padding=paddings[1])
        layer_name = 'layer{}'.format(1)
        self.add_module(layer_name, self.layer1)
        self.layer2 = UPBlock_V1(inplanes=outplanes[1], planes=outplanes[2], up_scale=4, stride=strides[2],
                                 padding=paddings[2])
        layer_name = 'layer{}'.format(2)
        self.add_module(layer_name, self.layer2)
        self.layer3 = UPBlock_V1(inplanes=outplanes[2], planes=outplanes[3], up_scale=4, stride=strides[3],
                                 padding=paddings[3])
        layer_name = 'layer{}'.format(3)
        self.add_module(layer_name, self.layer3)
        self.layer4 = UPBlock_V1(inplanes=outplanes[3], planes=outplanes[4], up_scale=4, stride=strides[4],
                                 padding=paddings[4])
        layer_name = 'layer{}'.format(4)
        self.add_module(layer_name, self.layer4)

        self.final = torch.nn.Conv2d(outplanes[4], 3, 1)
        self.tanh = torch.nn.Tanh()
        self.add_module('layer5', self.final)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=True, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m, mode='fan_in', nonlinearity='relu')
                elif isinstance(m, nn.ConvTranspose2d):
                    kaiming_init(m, mode='fan_in', nonlinearity='relu')
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Decoder_V2):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, Decoder_V2):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        outs = []
        # out = self.stem(self.linear(self.avg_pool(x[-1])))
        out = self.stem(self.conv7(x[-1]))
        # out = self.layer0(torch.cat([out, self.conv1_7(x[-1])], dim=1))
        out = self.layer0(out)
        # out = self.layer1(torch.cat([out, self.conv1_14(x[-2])], dim=1))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.tanh(self.final(out))
        outs.append(out)

        return tuple(outs)


if __name__ == '__main__':
    model = Decoder_V1()
    x = torch.randn(2, 3, 128, 128)
    y = model(x)
    print(len(y))
