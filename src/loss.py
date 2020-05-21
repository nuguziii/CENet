import torch
import torch.nn as nn
from torch.nn import functional as F

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.dice_loss = DiceLoss().cuda()
        self.ce_loss = CELoss().cuda()

    def forward(self, output, shape_output, contour_output, gt, contour_gt, shape_gt, class_weights):
        output_loss = self.dice_loss(gt, output, class_weights)
        shape_loss = self.dice_loss(shape_gt, shape_output, class_weights)
        contour_loss = self.ce_loss(contour_gt, contour_output, class_weights)
        return output_loss, shape_loss, contour_loss

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, target, output, weights=None):
        '''
        :param target: NxDxHxW LongTensor
        :param output: NxCxDxHxW Variable
        :param weights: C FloatTensor
        :param ignore_index: int index to ignore from loss
        :return:
        '''
        eps = 0.0001
        encoded_target = output.detach() * 0

        encoded_target.scatter_(1, target.unsqueeze(1), 1)

        if weights is None:
            weights = 1

        intersection = output * encoded_target
        numerator = 2 * torch.squeeze(intersection).sum(2).sum(2).sum(2)
        denominator = output + encoded_target
        denominator = torch.squeeze(denominator).sum(2).sum(2).sum(2) + eps
        loss_per_channel = weights * (1 - (numerator / denominator))

        return loss_per_channel.sum() / loss_per_channel.size(0)

class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()

    def forward(self, true, logits, weights, ignore=255):
        ce_loss = F.cross_entropy(
            logits.float(),
            true.long(),
            ignore_index=ignore,
            weight=weights,
        )
        return ce_loss