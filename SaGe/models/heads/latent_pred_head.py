import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from ..registry import HEADS
from .. import builder

@HEADS.register_module
class LatentPredictHead(nn.Module):
    """Head for contrastive learning.
    """

    def __init__(self, predictor, size_average=True, cat_batch=False):
        super(LatentPredictHead, self).__init__()
        self.predictor = builder.build_neck(predictor)
        self.size_average = size_average
        self.cat_batch = cat_batch

    def init_weights(self, init_linear='normal'):
        self.predictor.init_weights(init_linear=init_linear)

    def forward(self, input, target):
        """Forward head.

        Args:
            input (Tensor): NxC input features.
            target (Tensor): NxC target features.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        bs = input.size(0)
        pred = self.predictor([input])[-1]
        pred_norm = nn.functional.normalize(pred, dim=1)
        target_norm = nn.functional.normalize(target, dim=1)
        if self.cat_batch:
            bs_half = bs // 2
            loss = -2 * (pred_norm[:bs_half] * target_norm[bs_half:]).sum() - 2 * (pred_norm[bs_half:] * target_norm[:bs_half]).sum() 
            if self.size_average:
                loss /= bs_half
                loss += 4
        else:
            loss = -2 * (pred_norm * target_norm).sum()
            if self.size_average:
                loss /= bs
                loss += 2
        return dict(loss=loss)


@HEADS.register_module
class LatentClsHead(nn.Module):
    """Head for contrastive learning.
    """

    def __init__(self, predictor):
        super(LatentClsHead, self).__init__()
        self.predictor = nn.Linear(predictor.in_channels,
                                   predictor.num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def init_weights(self, init_linear='normal'):
        normal_init(self.predictor, std=0.01)

    def forward(self, input, target):
        """Forward head.

        Args:
            input (Tensor): NxC input features.
            target (Tensor): NxC target features.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        pred = self.predictor(input)
        with torch.no_grad():
            label = torch.argmax(self.predictor(target), dim=1).detach()
        loss = self.criterion(pred, label)
        return dict(loss=loss)


@HEADS.register_module
class DenseMatchHead(nn.Module):
    """Head for contrastive learning.
    """

    def __init__(self, predictor, size_average=True):
        super(DenseMatchHead, self).__init__()
        self.predictor = builder.build_neck(predictor)
        self.size_average = size_average

    def init_weights(self, init_linear='normal'):
        self.predictor.init_weights(init_linear=init_linear)

    def forward(self, input, target):
        """Forward head.

        Args:
            input (Tensor): NxC input features.
            target (Tensor): NxC target features.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        pred = self.predictor(input)[0]
        cos_sim = F.cosine_similarity(pred, target.unsqueeze(-1).unsqueeze(-1), dim=1)
        max_sim = F.adaptive_max_pool2d(cos_sim, (1, 1)).squeeze()
        loss = -2 * max_sim.sum()
        if self.size_average:
            loss /= pred.size(0)
            loss += 2
        return dict(loss=loss)


@HEADS.register_module
class DenseCLHead(nn.Module):
    """Head for contrastive learning.
    """

    def __init__(self, predictor, size_average=True, swap_dense_loss=False, soft_loss=False):
        super(DenseCLHead, self).__init__()
        self.predictor = builder.build_neck(predictor)
        self.size_average = size_average
        self.swap_dense_loss = swap_dense_loss
        self.soft_loss = soft_loss

    def init_weights(self, init_linear='normal'):
        self.predictor.init_weights(init_linear=init_linear)

    def comp_dense_loss(self, feat1, feat2, dense1, dense2, soft_loss=False):
        b, c, h, w = feat1.shape
        feat1 = feat1.view(b, c, -1).unsqueeze(-1)
        feat2 = feat2.view(b, c, -1).unsqueeze(-2)
        cos_sim_feat = F.cosine_similarity(feat1, feat2, dim=1).detach().clone()

        b, c, h, w = dense1.shape
        dense1 = dense1.view(b, c, -1)
        dense2 = dense2.view(b, c, -1)
        if soft_loss:
            cos_sim = F.cosine_similarity(
                dense1.unsqueeze(-1), dense2.unsqueeze(-2), dim=1)
            cos_sim_loss = (cos_sim * F.softmax(cos_sim_feat, dim=-1)).sum()
            if self.swap_dense_loss:
                cos_sim_loss += (cos_sim * F.softmax(cos_sim_feat, dim=-2)).sum()
        else:
            argmax = torch.argmax(cos_sim_feat, dim=-1)
            cos_sim_loss = F.cosine_similarity(
                dense1, torch.gather(dense2, -1, argmax.unsqueeze(1).repeat(1, c, 1)), dim=1).sum()
            if self.swap_dense_loss:
                argmax = torch.argmax(cos_sim_feat, dim=-2)
                cos_sim_loss += F.cosine_similarity(
                    dense2, torch.gather(dense1, -1, argmax.unsqueeze(1).repeat(1, c, 1)), dim=1).sum()
        return cos_sim_loss, cos_sim_feat


    def forward(self, feats_on, feats_targ):
        pred = self.predictor((feats_on['dense_proj'],))[0]

        feat_on, feat_targ = feats_on['feat'].detach(), feats_targ['feat']
        target = feats_targ['dense_proj']
        b, c, h, w = feat_on.shape

        cos_sim, cos_sim_feat = self.comp_dense_loss(feat_on, feat_targ, pred, target, self.soft_loss)
        if self.swap_dense_loss:
        #     cos_sim_t2o, cos_sim_feat_t2o = self.comp_dense_loss(feat_targ, feat_on, target, pred)
        #     loss = -2 * (cos_sim_o2t.sum() + cos_sim_t2o.sum())
            loss_bias = 4
        else:
            loss_bias = 2
        loss = -2 * cos_sim

        if self.size_average:
            loss /= b * h * w
            loss += loss_bias
        # if self.swap_dense_loss:
        #     return dict(loss=loss, cos_sim_o2t=cos_sim_feat_o2t, cos_sim_t2o=cos_sim_feat_t2o)
        # else:
        return dict(loss=loss)



@HEADS.register_module
class TransPredictHead(LatentPredictHead):
    """Head for contrastive learning.
    """

    def __init__(self, predictor, size_average=True):
        super(TransPredictHead, self).__init__(predictor, size_average)

    def forward(self, input, target):
        """Forward head.

        Args:
            input (Tensor): NxHWxC input features.
            target (Tensor): NxHWxC target features.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        bs = input.shape[0]
        pred, attn = self.predictor(input)
        pred_norm = nn.functional.normalize(pred[:, 0], dim=1)
        target_norm = nn.functional.normalize(target[:, 0], dim=1)
        loss = -2 * (pred_norm * target_norm).sum()
        if self.size_average:
            loss /= bs
            loss += 2
        return dict(loss=loss, attn=attn)
