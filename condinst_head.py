import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
from mmcv.cnn import (ConvModule, Scale, kaiming_init, normal_init, 
    bias_init_with_prob)
from mmcv.runner import force_fp32
from mmcv.ops.nms import batched_nms

from mmdet.core import (distance2bbox, multi_apply, reduce_mean, bbox2result, bbox_overlaps)
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from mmdet.models.dense_heads.paa_head import levels_to_images 
from sfl import SoftFocalLoss
INF = 1e8

def multiclass_nms(multi_bboxes,
                   multi_scores,
                   multi_kernels,
                   multi_points,
                   multi_strides,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None,
                   with_nms=True):
    num_classes = multi_scores.size(1) - 1
    bboxes = multi_bboxes[:, None].expand(
        multi_scores.size(0), num_classes, 4)
    kernels = multi_kernels[:, None].expand(
        multi_scores.size(0), num_classes, 169)
    bboxes = multi_bboxes[:, None].expand(
        multi_scores.size(0), num_classes, 4)
    points = multi_points[:, None].expand(
        multi_scores.size(0), num_classes, 2)
    strides = multi_strides[:, None].expand(
        multi_scores.size(0), num_classes)
    scores = multi_scores[:, :-1]

    labels = torch.arange(num_classes, dtype=torch.long)
    labels = labels.view(1, -1).expand_as(scores)

    bboxes = bboxes.reshape(-1, 4)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)
    kernels = kernels.reshape(-1, 169)
    points = points.reshape(-1, 2)
    strides = strides.reshape(-1, 1)

    # remove low scoring boxes
    valid_mask = scores > score_thr
    if score_factors is not None:
        # expand the shape to match original shape of score
        score_factors = score_factors.view(-1, 1).expand(
            multi_scores.size(0), num_classes)
        score_factors = score_factors.reshape(-1)
        scores = scores * score_factors
    inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
    bboxes, scores, labels, kernels, points, strides = \
        bboxes[inds], scores[inds], labels[inds], kernels[inds], points[inds], strides[inds]
    if inds.numel() == 0:
       return bboxes, labels, kernels, points, strides
    if with_nms:
        dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)
        if max_num > 0:
            dets = dets[:max_num]
            keep = keep[:max_num]
        return dets, labels[keep], kernels[keep], points[keep], strides[keep]

    else:
        dets = torch.cat([bboxes, scores[:,None]], -1)
        keep = scores.argsort(descending=True)
        keep = keep[:max_num]
        
            
        return dets[keep], labels[keep], kernels[keep], points[keep], strides[keep]


def aligned_bilinear(tensor, factor):
    assert tensor.dim() == 4
    assert factor >= 1
    assert int(factor) == factor 

    if factor == 1:
        return tensor 
    
    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode='replicate')
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(
        tensor, size=(oh, ow),
        mode='bilinear',
        align_corners=True)
    tensor = F.pad(
        tensor, pad=(factor//2, 0, factor//2, 0),
        mode='replicate')

    return tensor[:, :, :oh - 1, :ow - 1]


def compute_bbox_quality(pred, target):
    pred_left = pred[:, 0]
    pred_top = pred[:, 1]
    pred_right = pred[:, 2]
    pred_bottom = pred[:, 3]

    target_left = target[:, 0]
    target_top = target[:, 1]
    target_right = target[:, 2]
    target_bottom = target[:, 3]

    target_area = (target_left + target_right) * \
                  (target_top + target_bottom)
    pred_area = (pred_left + pred_right) * \
                (pred_top + pred_bottom)

    w_intersect = torch.min(pred_left, target_left) + \
                  torch.min(pred_right, target_right)
    h_intersect = torch.min(pred_bottom, target_bottom) + \
                  torch.min(pred_top, target_top)

    # g_w_intersect = torch.max(pred_left, target_left) + \
    #                 torch.max(pred_right, target_right)
    # g_h_intersect = torch.max(pred_bottom, target_bottom) + \
    #                 torch.max(pred_top, target_top)
    # ac_uion = g_w_intersect * g_h_intersect

    area_intersect = w_intersect * h_intersect
    area_union = target_area + pred_area - area_intersect

    ious = (area_intersect + 1.0) / (area_union + 1.0)
    # gious = ious - (ac_uion - area_union) / ac_uion

    return ious


def dice_coefficient(pred, target):
    eps = 1e-5
    n_inst = pred.size(0)
    pred = pred.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (pred * target).sum(dim=1)
    union = (pred ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps 
    dice = 1. - (2 * intersection / union)
    return dice



@HEADS.register_module()
class E2ECondInstHead(AnchorFreeHead):
    """Anchor-free head used in `CondInst <https://arxiv.org/abs/2003.05664>`_.

    The CondInst head does not use anchor boxes. 
    Here norm_on_bbox and centerness_on_reg are set as default, thus removed 
    from init arguments. New arguments are introduced:

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        strides (list[int] | list[tuple[int, int]]): Strides of points
            in multiple feature levels. Default: (4, 8, 16, 32, 64).
        regress_ranges (tuple[tuple[int, int]]): Regress range of multiple
            level points.
        center_sampling (bool): If true, use center sampling. 
            The center sampling strategy is changed if gt_masks are provided as 
            annotations. Default: False.
        center_sample_radius (float): Radius of center sampling. Default: 1.5.
        box_quality (str): The box quality target type, choose between ["centerness", 
            "iou"]. Default: "centerness".
        num_dynamic_layers (int): The number of conditional convolution layers as 
            described in paper.  
        conv_bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias of conv will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        dynamic_channels (int): The number of channels in dynamic convolutions. This 
            is different from the number of bases in the mask branch.
        mask_inputs (tuple): The indices to the input layers from FPN outputs.
        num_mask_layers (int): The number of conv blocks in the mask branch.
        num_bases (int): The output channels in the mask branch.
        sem_loss_on (bool): Whether to include auxiliary segmentation loss 
            during training. The results will be disregarded during inference.
        max_proposals (int): If not set to -1, randomly sample N number of proposals 
            for mask training. Default: "-1"
        topk_proposals_per_im (int): If not set to -1, sample topk highest ranked 
            proposals per image. Default: "-1". Note that the acutal number of proposals 
            could be less, because the operation caps the number of proposals for each 
            ground truth instance.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_quality (dict): Config of quality loss. The official implementation now 
            supports proposal IoU as centerness targets. Thus change the name to 
            loss_quality. 
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN', num_groups=32, requires_grad=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> self = CondInst(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, centerness = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """  # noqa: E501

    def __init__(self,
                 num_classes,
                 in_channels,
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 center_sampling=True,
                 center_sample_radius=1.5,
                 box_quality="centerness",
                 num_dynamic_layers=3,
                 dynamic_channels=8,
                 mask_inputs=(0, 1, 2),
                 mask_channels=128,
                 num_mask_layers=4,
                 num_bases=8,
                 sem_loss_on=False,
                 max_proposals=-1,
                 topk_proposals_per_im=64,
                 mask_out_stride=4,
                 sizes_of_interest=[64,128,256,512],
                 max_epoch=12,
                 o2f_topk=7,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
                 loss_quality=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='conv_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.num_dynamic_layers = num_dynamic_layers
        self.dynamic_channels = dynamic_channels
        self.num_bases = num_bases 
        loss_cls['activated'] = True
        super().__init__(
            num_classes,
            in_channels,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            **kwargs)
        assert box_quality in ["centerness", "iou"]
        self.box_quality = 'centerness'
        self.loss_quality = build_loss(loss_quality)
        self.mask_inputs = mask_inputs
        self.sem_loss_on = True
        self.mask_channels = mask_channels
        self.num_mask_layers = num_mask_layers
        soi = sizes_of_interest
        self.register_buffer("sizes_of_interest", torch.tensor(soi + [soi[-1] * 2]))
        self._init_mask_branch()

        self.max_proposals = max_proposals
        self.topk_proposals_per_im = topk_proposals_per_im
        self.mask_out_stride = mask_out_stride
        # cannot be both switched on
        assert (self.max_proposals == -1) or (self.topk_proposals_per_im == -1)

        
        self.mask_loss_weight = 2
        self.o2o_topk = o2f_topk
        self.soft_weight = 0.4
        self.max_soft_weight = 1.
        self.qfl_loss = SoftFocalLoss(activated=True)
        self.epoch = 0
        self.iter = 0
        self.max_epoch = max_epoch
        max_t = 0.6
        min_t = 0.2
        self.ff = lambda x: (min_t - max_t) / (self.max_epoch-1) * x + max_t 
        
    def init_weights(self):
        super().init_weights()
        normal_init(self.conv_quality, std=0.01)
        normal_init(self.controller, std=0.01)
        # follow official implementation, use kaiming uniform for mask_branch
        kaiming_init(self.refine, a=1, distribution='uniform')
        kaiming_init(self.mask_tower, a=1, distribution='uniform')
        if self.sem_loss_on:
            assert hasattr(self, "seg_head")
            assert hasattr(self, "seg_out")
            normal_init(self.seg_head, std=0.01)
            bias_init = bias_init_with_prob(0.01)
            normal_init(self.seg_out, std=0.01, bias=bias_init)

    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        self.conv_quality = nn.Conv2d(self.feat_channels//4, 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
        weight_nums, bias_nums = [], []
        for l in range(self.num_dynamic_layers):
            if l == 0:
                weight_nums.append((self.num_bases + 2) * self.dynamic_channels)
                bias_nums.append(self.dynamic_channels)
            elif l == self.num_dynamic_layers - 1:
                weight_nums.append(self.dynamic_channels)
                bias_nums.append(1)
            else:
                weight_nums.append(self.dynamic_channels * self.dynamic_channels)
                bias_nums.append(self.dynamic_channels)
        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)
        self.controller = nn.Conv2d(self.in_channels, self.num_gen_params,3,1,1)
        
        self.centerness_convs = nn.ModuleList()
        for i in range(3):
            if i == 0:
                in_channels = self.feat_channels 
                out_channels = self.feat_channels // 4
            else:
                in_channels = self.feat_channels // 4
                out_channels = self.feat_channels // 4
            self.centerness_convs.append(ConvModule(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    dilation=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))
    
    def _init_mask_branch(self):
        self.refine = nn.ModuleList()
        for _ in range(len(self.mask_inputs)):
            conv_block = []     # or simply use mmcv ConvModule() as conv_block
            conv_block.append(
                nn.Conv2d(
                    self.in_channels,
                    self.mask_channels,
                    3, 1, 1, bias=False))
            conv_block.append(nn.BatchNorm2d(self.mask_channels))
            conv_block.append(nn.ReLU(inplace=True))
            conv_block = nn.Sequential(*conv_block)
            self.refine.append(conv_block)

        tower = []
        for _ in range(self.num_mask_layers):
            conv_block = []
            conv_block.append(
                nn.Conv2d(
                    self.mask_channels,
                    self.mask_channels,
                    3, 1, 1, bias=False))
            conv_block.append(nn.BatchNorm2d(self.mask_channels))
            conv_block.append(nn.ReLU(inplace=True))
            conv_block = nn.Sequential(*conv_block)
            tower.append(conv_block)

        tower.append(nn.Conv2d(self.mask_channels, self.num_bases, 1, 1, 0))
        self.mask_tower = nn.Sequential(*tower)
        
        # auxiliary semantic seg head
        if self.sem_loss_on:
            num_classes = self.num_classes
            seg_head = []
            seg_head.append(
                nn.Conv2d(self.in_channels, self.mask_channels, 3, 1, 1, bias=False))
            seg_head.append(nn.BatchNorm2d(self.mask_channels))
            seg_head.append(nn.ReLU(inplace=True))
            seg_head.append(
                nn.Conv2d(self.mask_channels, self.mask_channels, 3, 1, 1, bias=False))
            seg_head.append(nn.BatchNorm2d(self.mask_channels))
            seg_head.append(nn.ReLU(inplace=True))
            self.seg_head = nn.Sequential(*seg_head)
            self.seg_out = nn.Conv2d(self.mask_channels, self.num_classes, 1, 1, 0)

    def parse_dynamic_params(self, top_feats):
        n_inst = top_feats.size(0)
        n_layers = len(self.weight_nums)
        params_splits = list(torch.split_with_sizes(
            top_feats, self.weight_nums + self.bias_nums, dim=1))
        weight_splits = params_splits[:n_layers]
        bias_splits = params_splits[n_layers:]
        for l in range(n_layers):
            if l < n_layers - 1:
                weight_splits[l] = weight_splits[l].reshape(
                    n_inst * self.dynamic_channels, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(n_inst * self.dynamic_channels)
            else:
                weight_splits[l] = weight_splits[l].reshape(n_inst, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(n_inst)

        return weight_splits, bias_splits

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_masks,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """ re-define to support gt_masks as inputs. Maybe a godd idea to include
             gt_masks support in the parient class."""
        outs = self(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, gt_masks, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Box scores for each scale level, \
                    each is a 4D-tensor, the channel number is \
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each \
                    scale level, each is a 4D-tensor, the channel number is \
                    num_points * 4.
                centernesses (list[Tensor]): centerness for each scale level, \
                    each is a 4D-tensor, the channel number is num_points * 1.
        """
        mask_feat, sem_preds = self.forward_mask_branch(feats)
        outs = (mask_feat, sem_preds) + multi_apply(self.forward_single, feats, self.scales,
                           self.strides) 
        return outs

    def forward_mask_branch(self, feats):
        for i in self.mask_inputs:
            if i == 0:
                x = self.refine[i](feats[i])
            else:
                x_p = self.refine[i](feats[i])
                target_h, target_w = x.size()[2:]
                h, w = x_p.size()[2:]
                assert target_h % h == 0
                assert target_w % w == 0
                factor_h, factor_w = target_h // h, target_w // w 
                assert factor_h == factor_w 
                x_p = aligned_bilinear(x_p, factor_h)
                x = x + x_p 
        mask_feat = self.mask_tower(x)
        sem_logits = None
        
        # add sem_preds
        if self.sem_loss_on: 
            sem_logits = self.seg_out(self.seg_head(feats[0]))
        
        return mask_feat, sem_logits

    def forward_single(self, x, scale, stride):
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.

        Returns:
            tuple: scores for each class, bbox predictions and centerness \
                predictions of input feature maps.
        """
        cls_score, bbox_pred, cls_feat, reg_feat = super().forward_single(x)
        
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = scale(bbox_pred).float()
        bbox_pred = F.relu(bbox_pred)
        if not self.training:
            bbox_pred *= stride
        top_feat = self.controller(reg_feat)
        centerness_feat = reg_feat
        for centerness_conv in self.centerness_convs:
            centerness_feat = centerness_conv(centerness_feat)
        qlty_pred = self.conv_quality(centerness_feat)
        return cls_score, bbox_pred, qlty_pred, top_feat

    def get_mask_head_inputs(self,
                             mask_feat,
                             points,
                             strides):
        n_inst = len(points)
        _, h, w = mask_feat.shape
        locations = self._get_points_single(
            (h, w), self.strides[0], torch.float32, mask_feat.device)
        relative_coords = points.reshape(-1, 1, 2) - locations.reshape(1, -1, 2)
        relative_coords = relative_coords.permute(0, 2, 1).float()
        relative_coords = relative_coords / (strides.float().reshape(-1, 1, 1) * 8)
        relative_coords = relative_coords.to(dtype=mask_feat.dtype)

        mask_head_inputs = torch.cat([
            relative_coords.reshape(n_inst, 2, h, w),
            mask_feat.repeat(n_inst, 1, 1, 1)], dim=1)
        mask_head_inputs = mask_head_inputs.reshape(1, -1, h, w)
        
        return mask_head_inputs

    def forward_mask_decode(self, features, weights, biases, n_inst):
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(x, w, bias=b, stride=1, padding=0, groups=n_inst)
            if i < n_layers - 1:
                x = F.relu(x)
        x = x.reshape(n_inst, 1, *x.size()[2:])
        assert self.strides[0] >= self.mask_out_stride
        assert self.strides[0] % self.mask_out_stride == 0
        x = aligned_bilinear(x, int(self.strides[0] / self.mask_out_stride))
        return x

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'qlty_preds'))
    def loss(self,
             mask_feat,  
             sem_preds,  # None or tensor
             cls_scores,
             bbox_preds,
             qlty_preds,
             top_feats,
             gt_bboxes,
             gt_labels,
             gt_masks,
             img_metas,
             gt_bboxes_ignore=None):

        # TODO rename all centerness into quality
        b = cls_scores[0].size(0)
        assert len(cls_scores) == len(bbox_preds) == len(qlty_preds)
        device = bbox_preds[0].device
        
        # FCOS LOSSES
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        points_list, strides_list = self.get_points(featmap_sizes, 
            bbox_preds[0].dtype, device)
        with torch.no_grad():
            labels_list, bbox_targets_list, soft_labels_list, soft_labels_weights_list, gt_inds_list = self.get_targets(
                points_list, gt_bboxes, gt_labels, gt_masks, cls_scores, bbox_preds, qlty_preds, img_metas, strides_list)
        #labels_list, bbox_targets_list, gt_inds_list = self.get_targets(
            #points_list, gt_bboxes, gt_labels, gt_masks)
        
        # completely overhaul to image-index first. This is to make sure mask 
        # predictions are now aligned with class and bbox prediction. Level-index 
        # can also be done, only need to shift the values in `gt_inds_list` by the 
        # number of ground truth instances in the previous images.  

        num_imgs = len(img_metas)
        batch_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        batch_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        batch_qlty_preds = [
            qlty_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 1)
            for qlty_pred in qlty_preds
        ]
        batch_top_feats = [
            top_feat.permute(0, 2, 3, 1).reshape(
                num_imgs, -1, self.num_gen_params)
            for top_feat in top_feats]

        # repeat points & strides to align with bbox_preds
        batch_points = torch.cat(points_list)[None].repeat(b, 1, 1)
        batch_strides = torch.cat(strides_list)[None].repeat(b, 1)

        # concat all fpn lvls
        batch_cls_scores = torch.cat(batch_cls_scores, dim=1)
        batch_bbox_preds = torch.cat(batch_bbox_preds, dim=1)
        batch_qlty_preds = torch.cat(batch_qlty_preds, dim=1)
        batch_top_feats = torch.cat(batch_top_feats, dim=1)
        batch_final_cls_scores = batch_cls_scores.sigmoid() * batch_qlty_preds.sigmoid()

        batch_labels = torch.stack(labels_list, dim=0)
        batch_bbox_targets = torch.stack(bbox_targets_list, dim=0)
        batch_soft_labels = torch.stack(soft_labels_list, dim=0)
        batch_soft_labels_weights = torch.stack(soft_labels_weights_list, dim=0)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        losses = {}
        bg_class_ind = self.num_classes
        _, _, mh, mw = mask_feat.shape
        pos_inds = []           # to align quality scores for gfl-like loss
        pos_decoded_bbox_preds, pos_decoded_bbox_targets = [], []
        pos_qlty_preds, pos_qlty_targets = [], []
        mask_preds, mask_targets = [], []

        for img_id in range(num_imgs):
            pos_inds_i = ((batch_labels[img_id] >= 0)
                    & (batch_labels[img_id] < bg_class_ind)).nonzero().reshape(-1)
            gt_inds_i = gt_inds_list[img_id]
            assert len(pos_inds_i) == len(gt_inds_i)
            pos_inds.append(pos_inds_i)
            mask_feat_i = mask_feat[img_id]

            if len(pos_inds_i) > 0:
                bbox_preds_i = batch_bbox_preds[img_id][pos_inds_i]
                bbox_targets_i = batch_bbox_targets[img_id][pos_inds_i]
                points_i = batch_points[img_id][pos_inds_i]
                strides_i = batch_strides[img_id][pos_inds_i]
                bbox_targets_i = bbox_targets_i / strides_i[:, None]
                
                cls_scores_i = batch_cls_scores[img_id]
                qlty_preds_i = batch_qlty_preds[img_id]
                final_cls_scores_i = batch_final_cls_scores[img_id]
                top_feats_i = batch_top_feats[img_id][pos_inds_i]
                pos_qlty_preds.append(qlty_preds_i[pos_inds_i])

                decoded_bbox_preds_i = distance2bbox(points_i, bbox_preds_i)
                decoded_bbox_targets_i = distance2bbox(points_i, bbox_targets_i)
                
                pos_decoded_bbox_preds.append(decoded_bbox_preds_i)
                pos_decoded_bbox_targets.append(decoded_bbox_targets_i)                
                
                if self.box_quality == "centerness":
                    bbox_quality_i = self.centerness_target(bbox_targets_i)
                elif self.box_quality == "iou":
                    bbox_quality_i = compute_bbox_quality(bbox_preds_i, 
                        bbox_targets_i)
                else:
                    raise NotImplementedError
                pos_qlty_targets.append(bbox_quality_i)

                # prepare masks 
                mask_head_input_i = self.get_mask_head_inputs(
                    mask_feat_i,
                    points_i,
                    strides_i)
                weights, biases = self.parse_dynamic_params(
                    top_feats_i)
                mask_logits_i = self.forward_mask_decode(
                    mask_head_input_i,
                    weights,
                    biases,
                    len(pos_inds_i))

                mask_preds_i = mask_logits_i.sigmoid()
                tmp_stride = self.mask_out_stride
                mask_targets_i = gt_masks[img_id][:, tmp_stride // 2::tmp_stride, 
                                                tmp_stride // 2::tmp_stride]
                mask_targets_i = mask_targets_i.gt(0.5).float()
                mask_targets_i = torch.index_select(
                    mask_targets_i, 0, gt_inds_i).contiguous()
                mask_targets_i = mask_targets_i.unsqueeze(1)

                mask_preds.append(mask_preds_i)
                mask_targets.append(mask_targets_i)

                kept_pos_inds = []
                if self.topk_proposals_per_im != -1:
                    unique_gt_inds = gt_inds_i.unique()
                    num_inst_per_gt = max(
                        int(self.topk_proposals_per_im / len(unique_gt_inds)), 1)
                    
                    for gt_ind in unique_gt_inds:
                        per_inst_pos_inds = pos_inds_i[gt_inds_i == gt_ind]
                        if len(per_inst_pos_inds) > num_inst_per_gt:
                            per_inst_scores = cls_scores_i[
                                per_inst_pos_inds].sigmoid().max(dim=1)[0]
                            per_inst_qlty = qlty_preds_i[
                                per_inst_pos_inds].sigmoid()
                            keep = (per_inst_scores * per_inst_qlty).topk(
                                k=num_inst_per_gt, dim=0)[1]
                            per_inst_pos_inds = per_inst_pos_inds[keep]
                        kept_pos_inds.append(per_inst_pos_inds)
                            
                    kept_pos_inds = torch.cat(kept_pos_inds).sort()[0]
                else:
                    kept_pos_inds = pos_inds_i
                kept_inds = (pos_inds_i[..., None] == kept_pos_inds).nonzero(
                    as_tuple=True)[0]
                mask_preds.append(mask_preds_i[kept_inds])
                mask_targets.append(mask_targets_i[kept_inds])
            else:
                pos_inds.append(batch_labels.new_empty((0,) ))
                pos_decoded_bbox_preds.append(batch_bbox_preds.new_empty((0, 4)))
                pos_decoded_bbox_targets.append(batch_bbox_targets.new_empty((0, 4)))
                pos_qlty_preds.append(batch_qlty_preds.new_empty((0)))
                pos_qlty_targets.append(batch_bbox_preds.new_empty((0)))
                mask_preds.append(mask_feat.new_empty((0, 1, 2*mh, 2*mw)))
                mask_targets.append(mask_feat.new_empty((0, 1, 2*mh, 2*mw)))

        # concat at image level
        pos_inds = torch.cat(pos_inds)
        num_pos = torch.tensor(len(pos_inds), dtype=torch.float, device=device)
        num_pos = max(reduce_mean(num_pos), 1.0)

        pos_decoded_bbox_preds = torch.cat(pos_decoded_bbox_preds)
        pos_decoded_bbox_targets = torch.cat(pos_decoded_bbox_targets)
        pos_qlty_preds = torch.cat(pos_qlty_preds)
        pos_qlty_targets = torch.cat(pos_qlty_targets)

        mask_preds = torch.cat(mask_preds)
        mask_targets = torch.cat(mask_targets)
        if self.max_proposals != -1 and len(pos_inds) > self.max_proposals:
            inds = torch.randperm(len(pos_inds), device=device).long()
            mask_preds = mask_preds[inds[:self.max_proposals]]
            mask_targets = mask_targets[inds[:self.max_proposals]]

        loss_cls = self.qfl_loss(batch_final_cls_scores.reshape(-1, self.cls_out_channels), (batch_labels.reshape(-1), batch_soft_labels.reshape(-1), batch_soft_labels_weights.reshape(-1)), avg_factor=num_pos)
        losses['loss_cls'] = loss_cls

        # TODO: This feels a bit redundant. TO simplify
        if len(pos_inds) > 0:
            if self.box_quality == "centerness":
                quality_denorm = max(
                    reduce_mean(pos_qlty_targets.sum().detach()), 1e-6)
            
                loss_bbox = self.loss_bbox(
                    pos_decoded_bbox_preds,
                    pos_decoded_bbox_targets,
                    weight=pos_qlty_targets,
                    avg_factor=quality_denorm)
            elif self.box_quality == "iou":
                loss_bbox = self.loss_bbox(
                    pos_decoded_bbox_preds,
                    pos_decoded_bbox_targets,
                    weight=None,
                    avg_factor=num_pos)
            else:
                raise NotImplementedError()

            loss_mask = dice_coefficient(mask_preds, mask_targets) * self.mask_loss_weight
        else:
            loss_bbox = batch_bbox_preds.sum() * 0
            loss_mask = mask_feat.sum() * 0 + batch_top_feats.sum() * 0 

        losses['loss_bbox'] = loss_bbox
        losses['loss_mask'] = loss_mask

        if self.sem_loss_on:
            # TODO DEBUG 
            assert sem_preds is not None
            semantic_targets = []
            for i, gt_mask in enumerate(gt_masks):
                h, w = gt_mask.size()[-2:]
                areas = gt_mask.sum(dim=-1).sum(dim=-1)
                areas = areas[:, None, None].repeat(1, h, w)
                areas[gt_mask == 0] = INF
                areas = areas.permute(1, 2, 0).reshape(h*w, -1)
                min_areas, inds = areas.min(dim=1)
                per_im_semantic_targets = gt_labels[i][inds]
                per_im_semantic_targets[min_areas == INF] = self.num_classes
                per_im_semantic_targets = per_im_semantic_targets.reshape(h, w)
                semantic_targets.append(per_im_semantic_targets)
            semantic_targets = torch.stack(semantic_targets, dim=0)
            tmp_stride = self.strides[0]
            semantic_targets = semantic_targets[:, tmp_stride // 2::tmp_stride, 
                                                tmp_stride // 2::tmp_stride]
            #semantic_targets = semantic_targets[:, 4::8, 4::8]
            pos_denorm = ((semantic_targets >= 0) * (semantic_targets < 
                self.num_classes)).sum().float().clamp(min=1.0)
            sem_preds = sem_preds.permute(0, 2, 3, 1).reshape(-1, self.num_classes).sigmoid()
            semantic_targets = semantic_targets.reshape(-1)
            loss_sem = self.loss_cls(
                sem_preds, semantic_targets, 
                reduction_override='sum') / pos_denorm
            losses['loss_sem'] = loss_sem
        return losses

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def get_bboxes(self,
                   mask_feat,
                   sem_preds,
                   cls_scores,
                   bbox_preds,
                   qlty_preds,
                   top_feats,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                with shape (N, num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W).
            qlty_preds (list[Tensor]): Centerness for each scale level with
                shape (N, num_points * 1, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used. Default: None.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """
        cfg = self.test_cfg if cfg is None else cfg 
        assert len(cls_scores) == len(bbox_preds) == len(qlty_preds)
        num_levels = len(cls_scores)
        device = cls_scores[0].device
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        
        mlvl_points, mlvl_strides = self.get_points(
            featmap_sizes, bbox_preds[0].dtype, bbox_preds[0].device)

        det_results_list, mask_results_list = [], []

        # split img, lvl idx first
        for i in range(len(img_metas)):
            cls_scores_i = [
                cls_scores[l][i].detach() for l in range(num_levels)]
            bbox_preds_i = [
                bbox_preds[l][i].detach() for l in range(num_levels)]
            qlty_preds_i = [
                qlty_preds[l][i].detach() for l in range(num_levels)]
            top_feats_i = [
                top_feats[l][i].detach() for l in range(num_levels)]
            mask_feat_i = mask_feat[i]
            img_meta_i = img_metas[i]

            det_bboxes, det_labels, det_masks = self._get_bboxes_single(
                cls_scores_i,
                bbox_preds_i,
                qlty_preds_i,
                top_feats_i,
                mask_feat_i,
                mlvl_points,
                mlvl_strides,
                img_meta_i,
                cfg,
                rescale)

            mask_results = [[] for _ in range(self.num_classes)]
            if det_bboxes.shape[0] == 0:
                det_results_list.append(
                    [np.zeros((0, 5), dtype=np.float32) for _ in range(self.num_classes)])
                mask_results_list.append(mask_results)
                continue

            bbox_results = bbox2result(det_bboxes, det_labels, self.num_classes)
            det_results_list.append(bbox_results)

            for idx in range(det_bboxes.shape[0]):
                label = det_labels[idx]
                mask = det_masks[idx].cpu().numpy()
                mask_results[label].append(mask)
            mask_results_list.append(mask_results)

        return det_results_list, mask_results_list 

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           qlty_preds,
                           top_feats,
                           mask_feat,
                           mlvl_points,
                           mlvl_strides,
                           img_meta,
                           cfg,
                           rescale=False):
        with_nms = cfg.get('with_nms', False)
        img_shape = img_meta['img_shape']
        scale_factor = img_meta['scale_factor']
        ori_shape = img_meta['ori_shape']

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_qlties = []
        mlvl_top_feats = []
        flatten_points = []
        flatten_strides = []
        for (cls_score, 
             bbox_pred, 
             qlty_pred, 
             top_feat, 
             points, 
             strides) in zip(
            cls_scores, 
            bbox_preds, 
            qlty_preds, 
            top_feats, 
            mlvl_points, 
            mlvl_strides):

            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            qlty_pred = qlty_pred.permute(1, 2, 0).reshape(-1).sigmoid()
            top_feat = top_feat.permute(1, 2, 0).reshape(-1, self.num_gen_params)

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                if self.box_quality == "centerness":
                    max_scores, _ = (scores * qlty_pred[:, None]).max(dim=1)
                elif self.box_quality == "iou":
                    max_scores, _ = scores.max(dim=1)
                else:
                    raise NotImplementedError
                _, topk_inds = max_scores.topk(nms_pre)

                points = points[topk_inds, :]
                strides = strides[topk_inds]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds]
                qlty_pred = qlty_pred[topk_inds]
                top_feat = top_feat[topk_inds, :]

            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_qlties.append(qlty_pred)
            mlvl_top_feats.append(top_feat)
            flatten_points.append(points)
            flatten_strides.append(strides)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_top_feats = torch.cat(mlvl_top_feats)
        if rescale:
            mlvl_bboxes = mlvl_bboxes / mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        mlvl_qlties = torch.cat(mlvl_qlties)
        flatten_points = torch.cat(flatten_points)
        flatten_strides = torch.cat(flatten_strides)

        det_bboxes, det_labels, det_top_feats, det_points, det_strides = \
            multiclass_nms(
                mlvl_bboxes,
                mlvl_scores,
                mlvl_top_feats,
                flatten_points,
                flatten_strides,
                cfg.score_thr,
                cfg.nms,
                cfg.max_per_img,
                score_factors=mlvl_qlties,
                with_nms=with_nms
            )
            
        mask_pred = []
        if det_bboxes.shape[0] > 0:
            n_inst = len(det_points)
            mask_head_inputs = self.get_mask_head_inputs(
                mask_feat,
                det_points,
                det_strides)
            weights, biases = self.parse_dynamic_params(det_top_feats)
            mask_logits = self.forward_mask_decode(
                mask_head_inputs,
                weights,
                biases,
                n_inst)

            mask_pred = mask_logits.sigmoid()
            mask_pred = aligned_bilinear(mask_pred, 4)
            mask_pred = mask_pred[:, :, :img_shape[0], :img_shape[1]]
            
            if rescale:
                mask_pred = F.interpolate(
                    mask_pred,
                    size=(ori_shape[0], ori_shape[1]),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)
            
            mask_pred = mask_pred.gt(0.5).float()
        
        return det_bboxes, det_labels, mask_pred

    def get_points(self, featmap_sizes, dtype, device):
        mlvl_points = []
        mlvl_strides = []
        for i in range(len(featmap_sizes)):
            points, strides = self._get_points_single(
                featmap_sizes[i],
                self.strides[i],
                dtype,
                device,
                return_stride=True)
            mlvl_points.append(points)
            mlvl_strides.append(strides)

        return mlvl_points, mlvl_strides

    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           return_stride=False):
        """Get points according to feature map sizes."""
        y, x = super()._get_points_single(featmap_size, stride, dtype, device)
        points = torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride),
                             dim=-1) + stride // 2
        if return_stride:
            strides = points[:, 0] * 0 + stride
            return points, strides
        return points


    def get_targets(self, points, gt_bboxes_list, gt_labels_list, gt_masks_list,
                    cls_scores, bbox_preds, centernesses, img_metas, strides_list):
        assert len(points) == len(self.regress_ranges)
        cls_scores = levels_to_images(cls_scores)
        bbox_preds = levels_to_images(bbox_preds)
        centernesses = levels_to_images(centernesses)
        
        num_imgs = len(gt_bboxes_list)
        num_levels = len(points)
        
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)]

        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)
        concat_strides = torch.cat(strides_list, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]
        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list, soft_labels_list, soft_labels_weights, gt_inds_list = multi_apply(
                self._get_target_single,
                gt_bboxes_list,
                gt_labels_list,
                gt_masks_list,
                cls_scores,
                bbox_preds,
                centernesses,
                img_metas,
                points=concat_points,
                strides=concat_strides,
                regress_ranges=concat_regress_ranges,
                num_points_per_lvl=num_points)

        # split to per img, per level
        # labels_list = [labels.split(num_points, 0) for labels in labels_list]
        # bbox_targets_list = [
        #     bbox_targets.split(num_points, 0)
        #     for bbox_targets in bbox_targets_list]

        return labels_list, bbox_targets_list, soft_labels_list, soft_labels_weights, gt_inds_list

    def _get_target_single(self, 
                           gt_bboxes, 
                           gt_labels, 
                           gt_masks, 
                           cls_scores,
                           bbox_preds,
                           centernesses,
                           img_metas,
                           points, 
                           strides,
                           regress_ranges,
                           num_points_per_lvl):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)

        # TODO make compatible
        if num_gts == 0:
            #raise NotImplementedError
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4)), gt_labels.new_full((num_points, ), -1)

        cls_scores = cls_scores.sigmoid()
        centernesses = centernesses.sigmoid()
        prob = (cls_scores * centernesses)[:, gt_labels]
        decode_bbox = self.bbox_coder.decode(points, bbox_preds.detach()*strides[:,None])
        iou = bbox_overlaps(decode_bbox, gt_bboxes)
        quality = prob ** (1 - 0.8) * iou ** 0.8

        
        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1])

        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            
            # use masks to determine center region
            _, h, w = gt_masks.size()
            yys = torch.arange(0, h, dtype=torch.float32, device=gt_masks.device)
            xxs = torch.arange(0, w, dtype=torch.float32, device=gt_masks.device)
            
            m00 = gt_masks.sum(dim=-1).sum(dim=-1).clamp(min=1e-6)
            m10 = (gt_masks * xxs).sum(dim=-1).sum(dim=-1)
            m01 = (gt_masks * yys[:, None]).sum(dim=-1).sum(dim=-1)
            center_xs = m10 / m00
            center_ys = m01 / m00
            center_xs = center_xs[None].expand(num_points, num_gts)
            center_ys = center_ys[None].expand(num_points, num_gts)

            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0],
                                             x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1],
                                             y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2],
                                             gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3],
                                             gt_bboxes[..., 3], y_maxs)

            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            # condition1: inside a gt bbox
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        quality[~inside_gt_bbox_mask] = 0
        iou[~inside_gt_bbox_mask] = 0
        valid_mask = inside_gt_bbox_mask.sum(1) > 0
        quality = quality[valid_mask]
        matching_matrix = torch.zeros_like(quality, dtype=torch.float)
        iou = iou[valid_mask]
        candidate_topk = (iou > 0).sum(0).int()
        for gt_idx in range(num_gts):
            _, pos_idx = torch.topk(quality[:, gt_idx], k=min(candidate_topk[gt_idx], self.o2o_topk), largest=True)
            matching_matrix[pos_idx, gt_idx] = quality[pos_idx, gt_idx]
        
        soft_point_labels = gt_labels.new_full((num_points,), 0, dtype=torch.float)
        soft_weights = gt_labels.new_full((num_points,), 0, dtype=torch.float)
        point_labels = gt_labels.new_full((num_points,), self.num_classes)
        point_bbox_targets = gt_bboxes.new_zeros((num_points, 4))
        
        valid_inds = torch.where(valid_mask == True)[0]
        positions_max_quality, gt_matched_idxs = matching_matrix.max(dim=1)
        pos_inds = torch.where(positions_max_quality>0)[0]
        pos_inds_all = valid_inds[pos_inds]
        point_labels[pos_inds_all] = gt_labels[gt_matched_idxs[pos_inds]]
        point_bbox_targets[pos_inds_all] = bbox_targets[pos_inds_all, gt_matched_idxs[pos_inds]]
        
        for gt_id in range(num_gts):
            pos_inds_i = pos_inds[torch.where(gt_matched_idxs[pos_inds] == gt_id)[0]]
            pos_inds_i_all = valid_inds[pos_inds_i]
            if len(pos_inds_i) > 0:
                prob_i = prob[pos_inds_i_all, gt_id]
                q_i = quality[pos_inds_i, gt_id]
                _, q_i_sorted_inds = q_i.sort(descending=True)
                max_q_i_ind = q_i_sorted_inds[0]
                st = cls_scores.new_zeros(len(q_i))
                sw = cls_scores.new_zeros(len(q_i)).fill_(self.soft_weight)
                if len(pos_inds_i) > 1:
                    num = len(pos_inds_i) 
                    t = self.ff(self.epoch)
                    max_prob_i = prob_i[q_i_sorted_inds[1:]].max()
                    for i, ind in enumerate(q_i_sorted_inds[1:]):
                        st[ind] = prob_i[ind] / max_prob_i * t
                
                st[max_q_i_ind] = 1
                sw[max_q_i_ind] = self.max_soft_weight
                soft_point_labels[pos_inds_i_all] = st
                soft_weights[pos_inds_i_all] = sw
        pos_gt_inds = gt_matched_idxs[pos_inds]
        return point_labels, point_bbox_targets, soft_point_labels, soft_weights, pos_gt_inds
    
    def centerness_target(self, pos_bbox_targets):
        """Compute centerness targets.

        Args:
            pos_bbox_targets (Tensor): BBox targets of positive bboxes in shape
                (num_pos, 4)

        Returns:
            Tensor: Centerness target.
        """
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        if len(left_right) == 0:
            centerness_targets = left_right[..., 0]
        else:
            centerness_targets = (
                left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                    top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)

