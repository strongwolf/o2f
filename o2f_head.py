import torch
import torch.nn as nn

from mmcv.cnn import  Scale,ConvModule
from mmcv.runner import force_fp32
from mmcv.ops import batched_nms
from mmdet.core import multi_apply, bbox_overlaps, reduce_mean, filter_scores_and_topk
from mmdet.models import HEADS, AnchorFreeHead
from mmdet.models.dense_heads.paa_head import levels_to_images 
from sfl import SoftFocalLoss
INF = 1e8

@HEADS.register_module()
class O2FHead(AnchorFreeHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 center_sampling=True,
                 center_sample_radius=1.5,
                 norm_on_bbox=True,
                 centerness_on_reg=True,
                 poto_alpha=0.8,
                 o2f_topk=7,
                 max_epoch=12,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0,
                     activated=True),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
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
        self.norm_on_bbox = norm_on_bbox
        self.centerness_on_reg = centerness_on_reg
        super().__init__(
            num_classes,
            in_channels,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            **kwargs)
        
        self.poto_alpha = poto_alpha
        self.o2f_topk = o2f_topk
        self.soft_weight = 0.4
        self.max_soft_weight = 1.
        self.qfl_loss = SoftFocalLoss(activated=True)
        self.epoch = 0
        self.max_epoch = max_epoch
        max_t = 0.6
        min_t = 0.2
        self.ff = lambda x: (min_t - max_t) / (self.max_epoch-1) * x + max_t 

    def _init_layers(self):
        """Initialize layers of the head."""
        self.cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))
        
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
        
        self.conv_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.conv_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.conv_centerness = nn.Conv2d(self.feat_channels//4, 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
    

    def forward(self, feats):
        cls_scores, bbox_preds, centernesses = multi_apply(self.forward_single, feats, self.scales,
                           self.strides)
        return cls_scores, bbox_preds, centernesses

    def forward_single(self, x, scale, stride):
        cls_feat = x
        reg_feat = x
        for conv in self.cls_convs:
            cls_feat = conv(cls_feat)
        cls_score = self.conv_cls(cls_feat)
        
        for conv in self.reg_convs:
            reg_feat = conv(reg_feat)
        bbox_pred = self.conv_reg(reg_feat)
        
        centerness_feat = reg_feat
        for centerness_conv in self.centerness_convs:
            centerness_feat = centerness_conv(centerness_feat)
        centerness = self.conv_centerness(centerness_feat)
        
        bbox_pred = scale(bbox_pred).float()
        if self.norm_on_bbox:
            bbox_pred = bbox_pred.clamp(min=0)
            bbox_pred *= stride
        else:
            bbox_pred = bbox_pred.exp()
        
        return cls_score, bbox_pred, centerness

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):

        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)
        with torch.no_grad():
            labels, bbox_targets, soft_labels, soft_weights = self.get_targets(all_level_points, gt_bboxes, gt_labels,
                                                cls_scores, bbox_preds, centernesses, img_metas)
        
        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1, 1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_cls_scores_o2o = flatten_cls_scores.sigmoid()*flatten_centerness.sigmoid()

        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_soft_labels = torch.cat(soft_labels)
        flatten_soft_weights = torch.cat(soft_weights)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos_o2o = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device) 
        num_pos_o2o = max(reduce_mean(num_pos_o2o), 1.0) 
        
        loss_cls = self.qfl_loss(flatten_cls_scores_o2o, (flatten_labels, flatten_soft_labels.float(),flatten_soft_weights), avg_factor=num_pos_o2o)
        
        pos_inds_for_reg = pos_inds
        pos_bbox_preds_o2o = flatten_bbox_preds[pos_inds_for_reg]
        pos_bbox_targets_o2o = flatten_bbox_targets[pos_inds_for_reg]
        pos_centerness_targets = self.centerness_target(pos_bbox_targets_o2o)
        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)
        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds_for_reg]
            pos_decoded_bbox_preds = self.bbox_coder.decode(
                pos_points, pos_bbox_preds_o2o)
            pos_decoded_target_preds = self.bbox_coder.decode(
                pos_points, pos_bbox_targets_o2o)
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=centerness_denorm)
        else:
            loss_bbox = flatten_bbox_preds.sum()

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox)

    def get_targets(self, points, gt_bboxes_list, gt_labels_list,
                        cls_scores, bbox_preds, centernesses, img_metas):
        num_levels = len(points)
        cls_scores = levels_to_images(cls_scores)
        bbox_preds = levels_to_images(bbox_preds)
        centernesses = levels_to_images(centernesses)
        num_points = [center.size(0) for center in points]
        concat_points = torch.cat(points)
        labels_list, bbox_targets_list, soft_labels_list, soft_weights = multi_apply(
            self._get_target_single_o2o,
            gt_bboxes_list,
            gt_labels_list,
            cls_scores,
            bbox_preds,
            centernesses,
            img_metas,
            points=concat_points,
            num_points_per_lvl=num_points)
        
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        soft_labels_list = [labels.split(num_points, 0) for labels in soft_labels_list]
        soft_weights = [weights.split(num_points, 0) for weights in soft_weights]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        concat_lvl_soft_labels = []
        concat_lvl_soft_weights = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            #if self.norm_on_bbox:
                #bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
            concat_lvl_soft_labels.append(
                torch.cat([labels[i] for labels in soft_labels_list]))
            concat_lvl_soft_weights.append(
                torch.cat([weights[i] for weights in soft_weights]))
        return concat_lvl_labels, concat_lvl_bbox_targets, concat_lvl_soft_labels, concat_lvl_soft_weights
        
    
    def _get_target_single_o2o(self, gt_bboxes, gt_labels, cls_scores, bbox_preds, centernesses, img_metas,
                        points, num_points_per_lvl):
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4))
        cls_scores = cls_scores.sigmoid()
        centernesses = centernesses.sigmoid()
        prob = (cls_scores * centernesses)[:, gt_labels]
        
        decode_bbox = self.bbox_coder.decode(points, bbox_preds)
        iou = bbox_overlaps(decode_bbox, gt_bboxes)
        quality = prob ** (1 - self.poto_alpha) * iou ** self.poto_alpha
        
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)
        lvl_inds = gt_labels.new_zeros(num_points)
        lvl_begin = 0
        for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
            lvl_end = lvl_begin + num_points_lvl
            lvl_inds[lvl_begin:lvl_end] = lvl_idx
            lvl_begin = lvl_end
                
        if self.center_sampling:
            radius = self.center_sample_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)

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
            _, pos_idx = torch.topk(quality[:, gt_idx], k=min(candidate_topk[gt_idx], self.o2f_topk), largest=True)
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
                    t = self.ff(self.epoch)
                    max_prob_i = prob_i[q_i_sorted_inds[1:]].max()
                    for i, ind in enumerate(q_i_sorted_inds[1:]):
                        st[ind] = prob_i[ind] / max_prob_i * t
                
                st[max_q_i_ind] = 1
                sw[max_q_i_ind] = self.max_soft_weight
                soft_point_labels[pos_inds_i_all] = st
                soft_weights[pos_inds_i_all] = sw
        
        return point_labels, point_bbox_targets, soft_point_labels, soft_weights
            
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
    
    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           score_factor_list,
                           mlvl_priors,
                           img_meta,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           **kwargs):

        cfg = self.test_cfg if cfg is None else cfg
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)
        with_nms = cfg.get('with_nms', False)

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_labels = []
        mlvl_score_factors = []
        for level_idx, (cls_score, bbox_pred, score_factor, priors) in \
                enumerate(zip(cls_score_list, bbox_pred_list,
                              score_factor_list, mlvl_priors)):

            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)

            score_factor = score_factor.permute(1, 2,
                                                0).reshape(-1).sigmoid()
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)

            scores = cls_score.sigmoid()
            results = filter_scores_and_topk(
                scores*score_factor[:,None], cfg.score_thr, nms_pre,
                dict(bbox_pred=bbox_pred, priors=priors))

            _, labels, keep_idxs, filtered_results = results
            scores = scores[keep_idxs, labels]
            bbox_pred = filtered_results['bbox_pred']
            priors = filtered_results['priors']

            score_factor = score_factor[keep_idxs]
            bboxes = self.bbox_coder.decode(
                priors, bbox_pred, max_shape=img_shape)

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)
            mlvl_score_factors.append(score_factor)

        return self._bbox_post_process(mlvl_scores, mlvl_labels, mlvl_bboxes,
                                       img_meta['scale_factor'], cfg, rescale,
                                       with_nms, mlvl_score_factors, **kwargs)
    
    def _bbox_post_process(self,
                           mlvl_scores,
                           mlvl_labels,
                           mlvl_bboxes,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           mlvl_score_factors=None,
                           **kwargs):
        assert len(mlvl_scores) == len(mlvl_bboxes) == len(mlvl_labels)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_labels = torch.cat(mlvl_labels)

        if mlvl_score_factors is not None:
            # TODOï¼š Add sqrt operation in order to be consistent with
            #  the paper.
            mlvl_score_factors = torch.cat(mlvl_score_factors)
            mlvl_scores = mlvl_scores * mlvl_score_factors
        
        if mlvl_bboxes.numel() == 0:
            det_bboxes = torch.cat([mlvl_bboxes, mlvl_scores[:, None]], -1)
            return det_bboxes, mlvl_labels

        if with_nms:
            det_bboxes, keep_idxs = batched_nms(mlvl_bboxes, mlvl_scores,
                                                mlvl_labels, cfg.nms)
            det_bboxes = det_bboxes[:cfg.max_per_img]
            det_labels = mlvl_labels[keep_idxs][:cfg.max_per_img]
            
        else:
            keep_idxs = mlvl_scores.argsort(descending=True)
            keep_idxs = keep_idxs[:cfg.max_per_img]
            det_bboxes = torch.cat([mlvl_bboxes, mlvl_scores[:,None]], -1)[keep_idxs]
            det_labels = mlvl_labels[keep_idxs]
        return det_bboxes, det_labels
    