import torch
import torch.nn as nn 
import torch.nn.functional as F

class DiceScore(nn.Module):
    def __init__(self, threshold=0.5, smooth=1.0, axis=None):
        super(DiceScore, self).__init__()
        self.threshold = threshold
        self.smooth = smooth
        self.axis = axis

    def forward(self, preds, labels):
        if self.threshold:
            preds = (preds > self.threshold).float()

        if self.axis:
            preds = preds[:, self.axis]
            labels = labels[:, self.axis]
        else:
            preds = preds[:, 1:]
            labels = labels[:, 1:]

        preds = preds.reshape(-1)
        labels = labels.reshape(-1)

        intersection = (preds * labels).sum()
        dice = (2.*intersection + self.smooth)/(preds.sum() + labels.sum() + self.smooth)
    
        return dice 


class DiceCELoss(nn.Module):
    def __init__(self, smooth=1.0, axis=(1, 2), weight_dice=(1, 10)):
        super(DiceCELoss, self).__init__()

        self.dice = [DiceScore(threshold=None, smooth=smooth, axis=i) for i in axis]
        self.weight_dice = weight_dice

    def forward(self, preds, labels):
        arg_labels = labels.argmax(1).long()
        _ce = F.nll_loss(torch.log(preds), arg_labels)

        _dice = 0.0
        _coeff = 0.0
        for i, dice in enumerate(self.dice):
            _dice += self.weight_dice[i]*(1 - dice(preds, labels))
            _coeff += self.weight_dice[i]
            
        _total = _dice/_coeff + _ce
        return _total
