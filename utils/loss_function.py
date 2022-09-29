import torch.nn as nn


class BinaryCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCELoss()
    
    def forward(self, predict, target):
        return self.loss_fn(predict, target)


class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predict, target):
        smooth = 1e-7

        n = target.size(0)
        predict = predict.view(n, -1)
        target = target.view(n, -1)
        intersection = (predict * target).sum(dim=1)

        dice = (2. * intersection + smooth) / (predict.sum(dim=1) + target.sum(dim=1) + smooth)
        dice_loss = 1 - dice.sum() / n
        return dice_loss


class BinaryIoULoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predict, target):
        smooth = 1e-7

        n = target.size(0)
        predict = predict.view(n, -1)
        target = target.view(n, -1)
        intersection = (predict * target).sum(dim=1)

        iou = (intersection + smooth) / (predict.sum(dim=1) + target.sum(dim=1) - intersection + smooth)
        iou_loss = 1 - iou.sum() / n
        return iou_loss