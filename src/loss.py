import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Function

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.dice_loss = DiceLoss()
        self.ce_loss = CELoss()

    def forward(self, output, shape_output, contour_output, gt, contour_gt, shape_gt, class_weights):
        output_loss = self.dice_loss(gt, output)
        shape_loss = self.dice_loss(shape_gt, shape_output)
        contour_loss = self.ce_loss(contour_gt, contour_output, class_weights)
        return output_loss, shape_loss, contour_loss

class DiceLoss(Function):
    def __init__(self, *args, **kwargs):
        pass
    def forward(self, target, input, save=True):
        if save:
            self.save_for_backward(input, target)
        eps = 0.000001
        _, result_ = input.max(1)
        #result_ = torch.squeeze(result_)
        if input.is_cuda:
            result = torch.cuda.FloatTensor(result_.size())
            self.target_ = torch.cuda.FloatTensor(target.size())
        else:
            result = torch.FloatTensor(result_.size())
            self.target_ = torch.FloatTensor(target.size())
        result.copy_(result_)
        self.target_.copy_(target)
        target = self.target_

        intersect = torch.sum(torch.sum(result * target))
        # binary values so sum the same as sum of squares
        result_sum = torch.sum(result)
        target_sum = torch.sum(target)
        union = result_sum + target_sum + (2*eps)

        # the target volume can be empty - so we still want to
        # end up with a score of 1 if the result is 0/0
        IoU = intersect / union
        out = 2*IoU
        self.intersect, self.union = intersect, union
        return out

    def backward(self, grad_output):
        input, _ = self.saved_tensors
        intersect, union = self.intersect, self.union
        target = self.target_
        gt = torch.div(target, union)
        IoU2 = intersect/(union*union)
        pred = torch.mul(input[:, 1], IoU2)
        dDice = torch.add(torch.mul(gt, 2), torch.mul(pred, -4))
        grad_input = torch.cat((torch.mul(dDice, -grad_output),
                                torch.mul(dDice, grad_output)), 0)
        return grad_input , None

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