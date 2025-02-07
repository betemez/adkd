# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import InstanceList, OptInstanceList, reduce_mean

from ..utils import multi_apply, unpack_gt_instances
from .crosskd_single_stage import CrossKDSingleStageDetector

class ABF(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, fuse):
        super(ABF, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channel, out_channel,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channel),
        )
        if fuse:
            self.att_conv = nn.Sequential(
                    nn.Conv2d(mid_channel*2, 2, kernel_size=1),
                    nn.Sigmoid(),
                )
        else:
            self.att_conv = None
        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)  # pyre-ignore
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)  # pyre-ignore

    def forward(self, x, y=None, shape=None):
        n,_,h,w = x.shape
        # transform student features
        x = self.conv1(x)
        if self.att_conv is not None:
            # upsample residual features
            shape = x.shape[-2:]
            y = F.interpolate(y, shape, mode="nearest")
            # fusion
            z = torch.cat([x, y], dim=1)
            z = self.att_conv(z)
            x = (x * z[:,0].view(n,1,h,w) + y * z[:,1].view(n,1,h,w))
        # output
        y = self.conv2(x)
        return y, x



@MODELS.register_module()
class CrossKDGFL(CrossKDSingleStageDetector):
    def __init__(
        self, in_channels, out_channels, mid_channel,
            backbone: ConfigType,
            neck: ConfigType,
            bbox_head: ConfigType,
            teacher_config: Union[ConfigType, str, Path],
            teacher_ckpt: Optional[str] = None,
            kd_cfg: OptConfigType = None,
            train_cfg: OptConfigType = None,
            test_cfg: OptConfigType = None,
            data_preprocessor: OptConfigType = None,
    ):
        super(CrossKDGFL, self).__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            teacher_ckpt=teacher_ckpt,
            teacher_config=teacher_config,
            kd_cfg=kd_cfg,
            data_preprocessor=data_preprocessor)

        abfs = nn.ModuleList()

        for idx, in_channel in enumerate(in_channels):
            abfs.append(ABF(in_channel, mid_channel, out_channels[idx], idx < len(in_channels)-1))

        self.abfs = abfs[::-1]

    def abf(self, x, y=None, shape=None):
        n,_,h,w = x.shape
        # transform student features
        x = self.conv1(x)
        if self.att_conv is not None:
            # upsample residual features
            shape = x.shape[-2:]
            y = F.interpolate(y, shape, mode="nearest")
            # fusion
            z = torch.cat([x, y], dim=1)
            z = self.att_conv(z)
            x = (x * z[:,0].view(n,1,h,w) + y * z[:,1].view(n,1,h,w))
        # output
        y = self.conv2(x)
        return y, x

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        tea_x = self.teacher.extract_feat(batch_inputs)
        alpha = 0.5
        """for t in tea_x:
            i = 0
            i += 1
            print("i", t)"""
        tea_cls_scores, tea_bbox_preds, tea_cls_hold, tea_reg_hold = \
            multi_apply(self.forward_crosskd_single, tea_x,
                        self.teacher.bbox_head.scales, module=self.teacher)
        stu_x = self.extract_feat(batch_inputs)

        # 进行特征融合
        teacher_features = tea_x
        student_features = stu_x

        fusion_features = []
        for teacher_feat, student_feat in zip(teacher_features, student_features):
            fused_feat = alpha * teacher_feat + (1 - alpha) * student_feat
            fusion_features.append(fused_feat)


        stu_cls_scores, stu_bbox_preds, stu_cls_hold, stu_reg_hold = \
            multi_apply(self.forward_crosskd_single, stu_x,
                        self.bbox_head.scales, module=self)
        reused_cls_scores, reused_bbox_preds = multi_apply(
            self.reuse_teacher_head, tea_cls_hold, tea_reg_hold, stu_cls_hold,
            stu_reg_hold, self.teacher.bbox_head.scales)

        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs
        losses = self.loss_by_feat(tea_cls_scores, tea_bbox_preds, tea_x,
                                   stu_cls_scores, stu_bbox_preds, stu_x,
                                   reused_cls_scores, reused_bbox_preds,
                                   batch_gt_instances, batch_img_metas,
                                   batch_gt_instances_ignore)
        return losses

    def forward_crosskd_single(self, x, scale, module):
        cls_feat, reg_feat = x, x
        cls_feat_hold, reg_feat_hold = x, x
        for i, cls_conv in enumerate(module.bbox_head.cls_convs):
            cls_feat = cls_conv(cls_feat, activate=False)
            if i + 1 == self.reused_teacher_head_idx:
                cls_feat_hold = cls_feat
            cls_feat = cls_conv.activate(cls_feat)
        for i, reg_conv in enumerate(module.bbox_head.reg_convs):
            reg_feat = reg_conv(reg_feat, activate=False)
            if i + 1 == self.reused_teacher_head_idx:
                reg_feat_hold = reg_feat
            reg_feat = reg_conv.activate(reg_feat)
        cls_score = module.bbox_head.gfl_cls(cls_feat)
        bbox_pred = scale(module.bbox_head.gfl_reg(reg_feat)).float()
        return cls_score, bbox_pred, cls_feat_hold, reg_feat_hold

    def reuse_teacher_head(self, tea_cls_feat, tea_reg_feat, stu_cls_feat,
                           stu_reg_feat, scale):
        reused_cls_feat = self.align_scale(stu_cls_feat, tea_cls_feat)
        reused_reg_feat = self.align_scale(stu_reg_feat, tea_reg_feat)
        if self.reused_teacher_head_idx != 0:
            reused_cls_feat = F.relu(reused_cls_feat)
            reused_reg_feat = F.relu(reused_reg_feat)

        module = self.teacher.bbox_head
        for i in range(self.reused_teacher_head_idx, module.stacked_convs):
            reused_cls_feat = module.cls_convs[i](reused_cls_feat)
            reused_reg_feat = module.reg_convs[i](reused_reg_feat)
        reused_cls_score = module.gfl_cls(reused_cls_feat)
        reused_bbox_pred = scale(module.gfl_reg(reused_reg_feat)).float()
        return reused_cls_score, reused_bbox_pred

    def align_scale(self, stu_feat, tea_feat):
        N, C, H, W = stu_feat.size()
        # normalize student feature
        stu_feat = stu_feat.permute(1, 0, 2, 3).reshape(C, -1)
        stu_mean = stu_feat.mean(dim=-1, keepdim=True)
        stu_std = stu_feat.std(dim=-1, keepdim=True)
        stu_feat = (stu_feat - stu_mean) / (stu_std + 1e-6)
        #
        tea_feat = tea_feat.permute(1, 0, 2, 3).reshape(C, -1)
        tea_mean = tea_feat.mean(dim=-1, keepdim=True)
        tea_std = tea_feat.std(dim=-1, keepdim=True)
        stu_feat = stu_feat * tea_std + tea_mean
        return stu_feat.reshape(C, N, H, W).permute(1, 0, 2, 3)

    def loss_by_feat(
            self,
            tea_cls_scores: List[Tensor],
            tea_bbox_preds: List[Tensor],
            tea_feats: List[Tensor],
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            feats: List[Tensor],
            reused_cls_scores: List[Tensor],
            reused_bbox_preds: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Cls and quality scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_preds (list[Tensor]): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.bbox_head.prior_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.bbox_head.get_anchors(
            featmap_sizes, batch_img_metas, device=device)

        cls_reg_targets = self.bbox_head.get_targets(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, avg_factor) = cls_reg_targets

        avg_factor = reduce_mean(
            torch.tensor(avg_factor, dtype=torch.float, device=device)).item()

        losses_cls, losses_bbox, losses_dfl,\
            new_avg_factor = multi_apply(
                self.bbox_head.loss_by_feat_single,
                anchor_list,
                cls_scores,
                bbox_preds,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                self.bbox_head.prior_generator.strides,
                avg_factor=avg_factor)

        new_avg_factor = sum(new_avg_factor)
        new_avg_factor = reduce_mean(new_avg_factor).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / new_avg_factor, losses_bbox))
        losses_dfl = list(map(lambda x: x / new_avg_factor, losses_dfl))
        losses = dict(
            loss_cls=losses_cls, loss_bbox=losses_bbox, loss_dfl=losses_dfl)

        losses_cls_kd, losses_reg_kd, kd_avg_factor = multi_apply(
            self.pred_mimicking_loss_single,
            tea_cls_scores,
            tea_bbox_preds,
            reused_cls_scores,
            reused_bbox_preds,
            label_weights_list,
            avg_factor=avg_factor)
        kd_avg_factor = sum(kd_avg_factor)
        losses_reg_kd = list(map(lambda x: x / kd_avg_factor, losses_reg_kd))
        losses.update(
            dict(loss_cls_kd=losses_cls_kd, loss_reg_kd=losses_reg_kd))

        if self.with_feat_distill:
            losses_feat_kd = [
                self.loss_feat_kd(feat, tea_feat)
                for feat, tea_feat in zip(feats, tea_feats)
            ]
            losses.update(loss_feat_kd=losses_feat_kd)
        return losses

    def pred_mimicking_loss_single(self, tea_cls_score, tea_bbox_pred,
                                   reused_cls_score, reused_bbox_pred,
                                   label_weights, avg_factor):
        # classification branch distillation
        tea_cls_score = tea_cls_score.permute(0, 2, 3, 1).reshape(
            -1, self.bbox_head.cls_out_channels)
        reused_cls_score = reused_cls_score.permute(0, 2, 3, 1).reshape(
            -1, self.bbox_head.cls_out_channels)
        label_weights = label_weights.reshape(-1)
        loss_cls_kd = self.loss_cls_kd(
            reused_cls_score,
            tea_cls_score,
            label_weights,
            avg_factor=avg_factor)

        # regression branch distillation
        reg_max = self.bbox_head.reg_max
        tea_bbox_pred = tea_bbox_pred.permute(0, 2, 3,
                                              1).reshape(-1, reg_max + 1)
        reused_bbox_pred = reused_bbox_pred.permute(0, 2, 3, 1).reshape(
            -1, reg_max + 1)
        reg_weights = tea_cls_score.max(dim=1)[0].sigmoid()
        reg_weights[label_weights == 0] = 0
        loss_reg_kd = self.loss_reg_kd(
            reused_bbox_pred,
            tea_bbox_pred,
            weight=reg_weights[:, None].expand(-1, 4).reshape(-1),
            avg_factor=4.0)

        return loss_cls_kd, loss_reg_kd, reg_weights.sum()