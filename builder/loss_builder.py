import torch
import torch.nn as nn
from utils.lovasz_losses import lovasz_softmax
import torch.nn.functional as F
import numpy as np

class Lovasz_loss(nn.Module):
    def __init__(self, ignore=None):
        super(Lovasz_loss, self).__init__()
        self.ignore = ignore

    def forward(self, probas, labels):
        return lovasz_softmax(probas, labels, ignore=self.ignore)

class LabelSmoothingLoss1(torch.nn.Module):
    def __init__(self, smoothing: float = 0.1, 
                 reduction="mean", weight=None):
        super(LabelSmoothingLoss1, self).__init__()
        self.smoothing   = smoothing
        self.reduction = reduction
        self.weight    = weight

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
         if self.reduction == 'sum' else loss

    def linear_combination(self, x, y):
        return self.smoothing * x + (1 - self.smoothing) * y

    def forward(self, preds, target):
        assert 0 <= self.smoothing < 1

        if self.weight is not None:
            self.weight = self.weight.to(preds.device)

        n = preds.size(-1)
        log_preds = F.log_softmax(preds, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1))
        nll = F.nll_loss(
            log_preds, target, reduction=self.reduction, weight=self.weight
        )
        return self.linear_combination(loss / n, nll)
    

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1, weight = None):
        """if smoothing == 0, it's one-hot method
           if 0 < smoothing < 1, it's smooth method
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.weight = weight
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        assert 0 <= self.smoothing < 1
        pred = pred.log_softmax(dim=self.dim)

        if self.weight is not None:
            pred = pred * self.weight.unsqueeze(0)   

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class SmoothCrossEntropy(torch.nn.Module):
    def __init__(self, label_smoothing=0.2, 
                 ignore_index=None, 
                 num_classes=20, 
                 weight=None, 
                 return_valid=False
                 ):
        super(SmoothCrossEntropy, self).__init__()
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        self.return_valid = return_valid
        # Reduce label values in the range of logit shape
        if ignore_index is not None:
            reducing_list = torch.range(0, num_classes).long().cuda(non_blocking=True)
            inserted_value = torch.zeros((1, )).long().cuda(non_blocking=True)
            self.reducing_list = torch.cat([
                reducing_list[:ignore_index], inserted_value,
                reducing_list[ignore_index:]
            ], 0)
        if weight is not None:
            self.weight = weight.float().cuda(
                non_blocking=True).squeeze()
        else:
            self.weight = None
            
    def forward(self, pred, gt):
        if len(pred.shape)>2:
            pred = pred.transpose(1, 2).reshape(-1, pred.shape[1])
        gt = gt.contiguous().view(-1)
        
        if self.ignore_index is not None: 
            valid_idx = gt != self.ignore_index
            pred = pred[valid_idx, :]
            gt = gt[valid_idx]        
            gt = torch.gather(self.reducing_list, 0, gt)
            
        if self.label_smoothing > 0:
            n_class = pred.size(1)
            one_hot = torch.zeros_like(pred).scatter(1, gt.view(-1, 1), 1)
            one_hot = one_hot * (1 - self.label_smoothing) + (1 - one_hot) * self.label_smoothing / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            if self.weight is not None:
                loss = -(one_hot * log_prb * self.weight).sum(dim=1).mean()
            else:
                loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(pred, gt, weight=self.weight)
        
        if self.return_valid:
            return loss, pred, gt
        else:
            return loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5, ignore_index=None):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        """
        logits: [N, C] (未经过softmax)
        target: [N] (long, 每个点的类别)
        """
        num_classes = logits.shape[1]
        pred = torch.softmax(logits, dim=1)
        target_onehot = torch.zeros_like(pred).scatter_(1, target.unsqueeze(1), 1)
        if self.ignore_index is not None:
            valid = (target != self.ignore_index)
            pred = pred[valid]
            target_onehot = target_onehot[valid]
        intersect = (pred * target_onehot).sum(dim=0)
        union = pred.sum(dim=0) + target_onehot.sum(dim=0)
        dice = (2 * intersect + self.smooth) / (union + self.smooth)
        loss = 1 - dice
        return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, ignore_index=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = logpt.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = pt.gather(1, target.unsqueeze(1)).squeeze(1)
        loss = -((1 - pt) ** self.gamma) * logpt
        if self.weight is not None:
            # 保证 weight 和 input 在同一 device
            weight = self.weight.to(input.device)
            loss = loss * weight[target]
        if self.ignore_index is not None:
            valid = (target != self.ignore_index)
            loss = loss[valid]
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class criterion(nn.Module):
    def __init__(self, config, device):
        super(criterion, self).__init__()
        self.config = config
        self.lambda_lovasz = self.config['train_params']['lambda_lovasz']
        self.lambda_cc = 1.0
        self.dice_weight = self.config['train_params']['dice_weight']
        if 'seg_labelweights' in config['dataset_params']:
            seg_num_per_class = config['dataset_params']['seg_labelweights']
            weight = seg_num_per_class / np.sum(seg_num_per_class).astype(float)
            seg_labelweights = 1 / (weight + 0.02)
            seg_labelweights[8] = 2000
            seg_labelweights[12] = 2000
            seg_labelweights = torch.from_numpy(seg_labelweights).float()
        else:
            seg_labelweights = None
        seg_labelweights = torch.from_numpy(seg_labelweights).float().to(device)
        self.ce_loss = nn.CrossEntropyLoss( 
            ignore_index=config['dataset_params']['ignore_label'],
            weight=seg_labelweights,
            label_smoothing=0.2
        ).to(device)


        self.lovasz_loss = Lovasz_loss(
            ignore=config['dataset_params']['ignore_label']
        )

        self.focal_loss = FocalLoss(
                gamma=config['train_params'].get('focal_gamma', 2.0),
                weight=seg_labelweights,
                ignore_index=config['dataset_params']['ignore_label']
            )
        
        self.dice_loss = DiceLoss(ignore_index=config['dataset_params']['ignore_label'])
        
        '''
        交叉熵更关注每个点的分类准确率。
        Dice Loss 更关注整体区域的重叠（IoU）。
        Lovasz Loss 直接优化 mIoU。
        
        普通交叉熵损失对所有样本一视同仁，容易被大量易分样本主导，导致模型对小类别或难分样本学习不足。
        Focal Loss 在交叉熵的基础上引入了一个调节因子 (1-pt)^gamma，其中 pt 是模型对真实类别的预测概率，gamma 是聚焦参数。
        对于易分样本（pt接近1），损失被大幅缩小，减少其对总损失的贡献。
        对于难分样本（pt接近0），损失被放大，模型会更关注这些样本。
        '''

    def forward(self, data_dict):

        loss_main_ce = self.ce_loss(data_dict['logits'], data_dict['labels'].long())
        loss_main_lovasz = self.lovasz_loss(torch.nn.functional.softmax(data_dict['logits'], dim=1), data_dict['labels'].long())
        loss_main_focal = self.focal_loss(data_dict['logits'], data_dict['labels'].long())
        loss_main_dice = self.dice_loss(data_dict['logits'], data_dict['labels'].long())
        # loss_main = loss_main_ce + loss_main_lovasz * self.lambda_lovasz 
        loss_main = loss_main_ce + loss_main_lovasz * self.lambda_lovasz +self.dice_weight*loss_main_dice+loss_main_focal

        return loss_main
    
    
    
    # data_dict['logits']：shape 通常为 [N, C]，N 是点的数量，C 是类别数。每一行是一个点属于每个类别的得分（未经过 softmax）。
    # data_dict['labels']：shape 通常为 [N]，每个元素是对应点的类别标签（整数）。
    # 损失函数会对每个点分别计算损失，然后求和或平均，作为整体的训练损失。这是点云分割任务的常规做法。