import torch
import torch.nn.functional as F
import pandas as pd
import math
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from einops import rearrange
import segmentation_models_pytorch as smp


def precision_at_k(output: torch.Tensor, target: torch.Tensor, top_k=(1,)):
    ''' Compute the accuracy over the k top predictions for the specified values of k'''
    with torch.no_grad():
        maxk = max(top_k)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in top_k:
            correct_k = correct[:k].contiguous(
            ).view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def clip_loss(x, y, temperature=0.07, device='cuda'):
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)

        sim = torch.einsum('i d, j d -> i j', x, y) * 1 / temperature

        labels = torch.arange(x.shape[0]).to(device)

        loss_t = F.cross_entropy(sim, labels) 
        loss_i = F.cross_entropy(sim.T, labels) 

        i2t_acc1, i2t_acc5 = precision_at_k(
            sim, labels, top_k=(1, 5))
        t2i_acc1, t2i_acc5 = precision_at_k(
            sim.T, labels, top_k=(1, 5))
        acc1 = (i2t_acc1 + t2i_acc1) / 2.
        acc5 = (i2t_acc5 + t2i_acc5) / 2.

        return (loss_t + loss_i), acc1, acc5

def bceDiceLoss(pred, target, train=True):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    smooth = 1e-5
    num = target.size(0)
    pred = pred.view(num, -1)
    target = target.view(num, -1)
    intersection = (pred * target)
    dice = (2. * intersection.sum(1) + smooth) / (pred.sum(1) + target.sum(1) + smooth)
    dice = 1 - dice.sum() / num
    if train:
        return dice + 0.2 * bce
    return dice
