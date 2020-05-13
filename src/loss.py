import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.dice_loss = SoftDiceLoss().cuda()
        self.ce_loss = CELoss().cuda()

    def forward(self, output, shape_output, contour_output, gt, contour_gt, shape_gt, class_weights):
        output_loss = self.dice_loss(gt, output)
        shape_loss = self.dice_loss(shape_gt, shape_output)
        contour_loss = self.ce_loss(contour_gt, contour_output, class_weights)
        return output_loss, shape_loss, contour_loss

class SoftDiceLoss(nn.Module):
    def __init__(self, n_classes=2):
        super(SoftDiceLoss, self).__init__()
        self.one_hot_encoder = One_Hot(n_classes).forward
        self.n_classes = n_classes

    def forward(self, target, input):
        smooth = 0.01
        batch_size = input.size(0)
        input = input.view(batch_size, self.n_classes, -1)
        target = self.one_hot_encoder(target).contiguous().view(batch_size, self.n_classes, -1)
        inter = torch.sum(input * target, 2) + smooth
        union = torch.sum(input, 2) + torch.sum(target, 2) + smooth
        score = torch.sum(2.0 * inter / union)
        score = 1.0 - score / (float(batch_size) * float(self.n_classes))
        return score

class One_Hot(nn.Module):
    def __init__(self, depth):
        super(One_Hot, self).__init__()
        self.depth = depth
        self.ones = torch.sparse.torch.eye(depth).cuda()
    def forward(self, X_in):
        n_dim = X_in.dim()
        output_size = X_in.size() + torch.Size([self.depth])
        num_element = X_in.numel()
        X_in = X_in.data.long().view(num_element)
        out = Variable(self.ones.index_select(0, X_in)).view(output_size)
        return out.permute(0, -1, *range(1, n_dim)).squeeze(dim=2).float()
    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.depth)

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, target, output, weights=None, ignore_index=None):
        '''
        :param target: NxDxHxW LongTensor
        :param output: NxCxDxHxW Variable
        :param weights: C FloatTensor
        :param ignore_index: int index to ignore from loss
        :return:
        '''
        eps = 0.0001
        encoded_target = output.detach() * 0

        if ignore_index is not None:
            mask = target == ignore_index
            target = target.clone()
            target[mask] = 0
            encoded_target.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(encoded_target)
            encoded_target[mask] = 0
        else:
            encoded_target.scatter_(1, target.unsqueeze(1), 1)

        if weights is None:
            weights = torch.Tensor([0.4, 0.6]).type(torch.cuda.FloatTensor)

        intersection = output * encoded_target
        numerator = 2 * torch.squeeze(intersection).sum(1).sum(1).sum(1)
        denominator = output + encoded_target

        if ignore_index is not None:
            denominator[mask] = 0
        denominator = torch.squeeze(denominator).sum(1).sum(1).sum(1) + eps
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