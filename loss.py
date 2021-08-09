from utils import sum_tensor
import torch
import torch.nn as nn 
import torch.nn.functional as F

def dice_score(tp, fp, fn, smooth=1.):
    result = (2*tp + smooth) / (2*tp + fp + fn + smooth)
    return result

def get_tp_fp_fn_tn(pred, label, axis=None, square=False):
    if not axis:
        axis = tuple(range(2, len(pred.shape)))

    shape_pred = pred.shape
    shape_label = label.shape

    with torch.no_grad():
        if not len(shape_pred) == len(shape_label):
            label = label.view((shape_label[0], 1, *shape_label[1:]))
        
        if all([i == j for i, j in zip(shape_pred, shape_label)]):
            label_onehot = label
        else:
            label = label.long()
            label_onehot = torch.zeros(shape_pred, device=pred.device)
            label_onehot.scatter_(1, label, 1)
        
    tp = pred * label_onehot
    fp = pred * (1 - label_onehot)
    fn = (1 - pred) * label_onehot
    tn = (1 - pred) * (1 - label_onehot)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2
    
    if len(axis) > 0:
        tp = sum_tensor(tp, axis)
        fp = sum_tensor(fp, axis)
        fn = sum_tensor(fn, axis)
        tn = sum_tensor(tn, axis)

    return tp, fp, fn, tn


class DiceLoss(nn.Module):
    def __init__(self, apply_nonlinear=None, batch_dice=False, do_bg=True, smooth=1.):
        super().__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlinear = apply_nonlinear
        self.smooth = smooth

    def forward(self, pred, label):
        shape = pred.shape

        if self.batch_dice:
            axis = [0] + list(range(2, len(shape)))
        else:
            axis = list(range(2, len(shape)))

        if self.apply_nonlinear is not None:
            pred = self.apply_nonlinear(pred)

        tp, fp, fn, _ = get_tp_fp_fn_tn(pred, label, axis=axis)
        dice = dice_score(tp, fp, fn, self.smooth)

        if not self.do_bg:
            if self.batch_dice:
                dice = dice[1:]
            else:
                dice = dice[:, 1:]

        dice = dice.mean()
        return -dice


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    def forward(self, input, target):
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        
        return super().forward(input, target.long())


class DiceCELoss(nn.Module):
    def __init__(self, dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, log_dice=False):
        super().__init__()

        self.log_dice = log_dice
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dice = DiceLoss(apply_nonlinear=lambda x: F.softmax(x, 1), **dice_kwargs)

    def forward(self, pred, label):
        dice_loss = self.dice(pred, label) if self.weight_dice != 0 else 0
        if self.log_dice:
            dice_loss = -torch.log(-dice_loss)
        ce_loss = self.ce(pred, label[:, 0].long()) if self.weight_ce != 0 else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dice_loss
        return result


class MultipleOutputLoss(nn.Module):
    def __init__(self, loss, weight_factors=None):
        super().__init__()

        self.weight_factors = weight_factors
        self.loss = loss

    def forward(self, pred, label):
        assert isinstance(pred, (tuple, list)), "pred must be either tuple or list" 
        assert isinstance(label, (tuple, list)), "label must be either tuple or list"

        if self.weight_factors is None:
            weights = [1] * len(pred)
        else:
            weights = self.weight_factors

        _loss = weights[0] * self.loss(pred[0], label[0])
        for i in range(1, len(pred)):
            if weights[i] != 0:
                _loss += weights[0] * self.loss(pred[i], label[i])

        return _loss
