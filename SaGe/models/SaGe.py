import torch
import torch.nn as nn
from torchvision.models import resnet18
from .decoder import Decoder
from .resnet import ResNet
from .necks import NonLinearNeckSimCLR
from math import cos, pi
from openselfsup.utils import print_log, get_root_logger
from ..apis.checkpoint import *
from mmcv import Config
import torch.nn.functional as F

from . import builder
from .registry import MODELS


@MODELS.register_module
class SaGe_Net(nn.Module):

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 base_momentum=0.99,
                 wtl2=0.0,
                 **kwargs):
        super(SaGe_Net, self).__init__()
        self.wtl2 = wtl2
        self.online_net = nn.Sequential(
            builder.build_backbone(backbone), builder.build_neck(neck))
        self.target_net = nn.Sequential(
            builder.build_backbone(backbone), builder.build_neck(neck))
        self.backbone = self.online_net[0]
        self.decoder = Decoder()
        # assert False
        for param in self.target_net.parameters():
            param.requires_grad = False
        self.head = builder.build_head(head)

        cfg_rf = Config.fromfile('./SaGe/models/rf_config.py')
        self.r_function = nn.Sequential(
            builder.build_backbone(cfg_rf['rf']['backbone']), builder.build_neck(cfg_rf['rf']['neck']))

        for param in self.r_function.parameters():
            param.requires_grad = False

        self.init_weights(pretrained=pretrained)
        self.base_momentum = base_momentum
        self.momentum = base_momentum

        self.mse = nn.MSELoss()

    def init_weights(self, pretrained=None):
        """Initialize the weights of model.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
        """
        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
        self.online_net[0].init_weights(pretrained=pretrained)  # backbone
        self.online_net[1].init_weights(init_linear='kaiming')  # projection
        for param_ol, param_tgt in zip(self.online_net.parameters(),
                                       self.target_net.parameters()):
            param_tgt.data.copy_(param_ol.data)
        # init the predictor in the head
        self.head.init_weights()
        self.decoder.init_weights()

        try:
            logger = get_root_logger()
            load_checkpoint(self.r_function, './resnet.pth.tar', strict=False, logger=logger, tmp=True)
            print('try load_checkpoint success')
        except:
            print('try load_checkpoint error')
            state_dict = torch.load('./resnet.pth.tar')['state_dict']
            print(state_dict.keys())
            if list(state_dict.keys())[0].startswith('module'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            if list(state_dict.keys())[0].startswith('online'):
                state_dict = {k[11:]: v for k, v in state_dict.items()}
            try:
                self.r_function.load_state_dict(state_dict)
            except Exception as e:
                print('r_function load ...')
                print(e)

    @torch.no_grad()
    def _momentum_update(self):
        """Momentum update of the target network."""
        for param_ol, param_tgt in zip(self.online_net.parameters(),
                                       self.target_net.parameters()):
            param_tgt.data = param_tgt.data * self.momentum + \
                             param_ol.data * (1. - self.momentum)

    @torch.no_grad()
    def momentum_update(self):
        self._momentum_update()

    def forward_train(self, img, **kwargs):
        # compute query features
        assert img.dim() == 5, \
            "Input must have 5 dims, got: {}".format(img.dim())

        img1 = img[:, 0, ...].contiguous()
        img2 = img[:, 1, ...].contiguous()
        en_online_v1 = self.online_net[0](img1)
        en_online_v2 = self.online_net[0](img2)
        proj_online_v1 = self.online_net[1]([en_online_v1[-1]])[-1]
        proj_online_v2 = self.online_net[1]([en_online_v2[-1]])[-1]
        with torch.no_grad():
            proj_target_v1 = self.target_net(img1)[-1].clone().detach()
            proj_target_v2 = self.target_net(img2)[-1].clone().detach()

        loss1 = self.head(proj_online_v1, proj_target_v2)['loss'] + \
                self.head(proj_online_v2, proj_target_v1)['loss']

        rf_img_v1, rf_recon_img1 = self.r_function(img1), self.r_function(recon_img1)
        rf_img_v2, rf_recon_img2 = self.r_function(img2), self.r_function(recon_img2)

        loss_mse = self.mse(img1, recon_img1) + self.mse(img2, recon_img2)
        loss_f28 = self.mse(rf_img_v1[-4], rf_recon_img1[-4]) + self.mse(rf_img_v2[-4], rf_recon_img2[-4])
        loss_f14 = self.mse(rf_img_v1[-3], rf_recon_img1[-3]) + self.mse(rf_img_v2[-3], rf_recon_img2[-3])
        loss_f7 = self.mse(rf_img_v1[-2], rf_recon_img1[-2]) + self.mse(rf_img_v2[-2], rf_recon_img2[-2])
        loss_f1 = self.mse(rf_img_v1[-1], rf_recon_img1[-1]) + self.mse(rf_img_v2[-1], rf_recon_img2[-1])

        assert img1.size(2) == 224
        assert recon_img1.size(2) == 224
        assert rf_img_v1[-2].size(2) == 7
        assert rf_img_v1[-3].size(2) == 14
        assert rf_img_v1[-4].size(2) == 28

        loss_mse /= 10
        loss_f1 /= 10
        loss_f7 /= 30
        loss_f14 /= 30
        loss_f28 /= 30

        return dict(loss1=loss1, loss_mse=loss_mse, loss_f1=loss_f1, loss_f7=loss_f7, loss_f14=loss_f14,
                    loss_f28=loss_f28)

    def forward_test(self, img, **kwargs):
        pass

    def forward(self, img, mode='train', **kwargs):
        if mode == 'train':
            return self.forward_train(img, **kwargs)
        elif mode == 'test':
            return self.forward_test(img, **kwargs)
        elif mode == 'extract':
            return self.backbone(img)
        else:
            raise Exception("No such mode: {}".format(mode))
