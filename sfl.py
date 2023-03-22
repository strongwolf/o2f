import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.losses.utils import weighted_loss

@weighted_loss
def quality_focal_loss_with_prob(pred, target, beta=2.0):

    label, score, weight = target

    # negatives are supervised by 0 quality score
    pred_sigmoid = pred
    scale_factor = pred_sigmoid
    zerolabel = scale_factor.new_zeros(pred.shape)
    loss = F.binary_cross_entropy(
        pred, zerolabel, reduction='none') * scale_factor.pow(beta) * 0.75

    # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
    bg_class_ind = pred.size(1)
    pos = ((label >= 0) & (label < bg_class_ind)).nonzero().squeeze(1)
    pos_label = label[pos].long()

    loss[pos, pos_label] = F.binary_cross_entropy(
        pred[pos, pos_label], score[pos],
        reduction='none') * weight[pos]

    loss = loss.sum(-1)
    return loss


class SoftFocalLoss(nn.Module):
    def __init__(self,
                 use_sigmoid=True,
                 beta=2.0,
                 reduction='mean',
                 loss_weight=1.0,
                 activated=True):
        super(SoftFocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid in SFL supported now.'
        self.use_sigmoid = use_sigmoid
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.activated = activated

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            if self.activated:
                calculate_loss_func = quality_focal_loss_with_prob
            else:
                raise NotImplementedError
            loss_cls = self.loss_weight * calculate_loss_func(
                pred,
                target,
                weight,
                reduction=reduction,
                avg_factor=avg_factor,
                beta=self.beta,
                )
        else:
            raise NotImplementedError
        return loss_cls