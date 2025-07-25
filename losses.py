#!/usr/bin/env python3
# encoding: utf-8
# Modified from https://github.com/Wangyixinxin/ACN
import torch
from torch.nn import functional as F
import numpy as np
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image 
import cv2
import torch.nn as nn
from models.evd import ExtremeValueDistillation
from models.slkd import DistillKL_logit_stand, DistillKL
from scipy.ndimage import label, center_of_mass
from skimage.measure import label
from skimage.morphology import remove_small_objects
from typing import Sequence, List

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def get_current_consistency_weight(epoch, consistency = 10, consistency_rampup = 20.0):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return consistency * sigmoid_rampup(epoch, consistency_rampup)
  
def bce_loss(y_pred, y_label):
    y_truth_tensor = torch.FloatTensor(y_pred.size())
    y_truth_tensor.fill_(y_label)
    y_truth_tensor = y_truth_tensor.to(y_pred.get_device())
    return nn.BCEWithLogitsLoss()(y_pred, y_truth_tensor)

import numpy as np
from scipy.ndimage import distance_transform_edt as edt

def compute_sdf(gt):        # gt: [B,1,D,H,W] 0/1
    gt = gt.cpu().numpy()
    sdf = np.zeros_like(gt, dtype=np.float32)
    for b in range(gt.shape[0]):
        pos = gt[b, 0].astype(bool)
        if pos.any():
            neg = ~pos
            sdf[b, 0]  = -edt(pos)          # 前景负
            sdf[b, 0] +=  edt(neg)          # 背景正
    return torch.from_numpy(sdf).to(gt.device)

class BoundaryLoss(nn.Module):
    """p(x)*|φ(x)| 形的 3-D Boundary Loss"""
    def __init__(self):
        super().__init__()
    def forward(self, logits, gt):
        # logits:[B,C,D,H,W] raw logits or probs   gt:[B,C,D,H,W] 0/1
        if logits.shape[1] != gt.shape[1]:
            raise ValueError("pred/gt 通道数不一致")
        probs = torch.softmax(logits, dim=1) if logits.dtype==torch.float32 else logits
        loss = 0.0
        for c in range(probs.shape[1]):
            gt_c   = gt[:, c:c+1]               # [B,1,D,H,W]
            sdf_c  = compute_sdf(gt_c)          # [B,1,D,H,W]
            pc     = (probs[:, c:c+1] * sdf_c).abs().mean()
            loss  += pc
        return loss / probs.shape[1]


def dice_loss(input: torch.Tensor,
              target: torch.Tensor,
              reduction: str = 'mean') -> torch.Tensor:
    """
    Soft Dice loss, with support for per-sample output.
    
    Args:
      input:  Tensor of shape [B, ...] or [...]. Predictions in [0,1].
      target: Tensor of same shape. Ground-truth in {0,1}.
      reduction: 'none' | 'mean' | 'sum'.
        - 'none': returns a tensor of shape [B], per-sample loss.
        - 'mean': returns a scalar = loss.mean().
        - 'sum' : returns a scalar = loss.sum().
    
    Returns:
      loss: Tensor, either [B] or scalar.
    """
    eps = 1e-7

    # 如果有 batch 维度（dim>1），则逐样本计算
    if input.dim() > 1:
        B = input.size(0)
        losses = []
        for i in range(B):
            iflat = input[i].contiguous().view(-1)
            tflat = target[i].contiguous().view(-1)
            intersection = (iflat * tflat).sum()
            losses.append(1 - 2. * intersection /
                          ((iflat ** 2).sum() + (tflat ** 2).sum() + eps))
        losses = torch.stack(losses)  # [B]
        if reduction == 'none':
            return losses
        elif reduction == 'mean':
            return losses.mean()
        elif reduction == 'sum':
            return losses.sum()
        else:
            raise ValueError(f"Unsupported reduction: {reduction}")

    # 否则当成单样本处理
    else:
        iflat = input.reshape(-1)
        tflat = target.reshape(-1)
        intersection = (iflat * tflat).sum()
        loss = 1 - 2. * intersection / ((iflat ** 2).sum() + (tflat ** 2).sum() + eps)
        return loss

def weighted_dice_loss(input: torch.Tensor,
                          target: torch.Tensor,
                          alpha: float = 1.0,
                          eps: float   = 1e-7) -> torch.Tensor:
    B, H, W, D = input.shape
    # ————— 1. Dice & base_loss —————
    p = input.view(B, -1); g = target.view(B, -1)
    inter = (p*g).sum(1)
    sums  = p.sum(1) + g.sum(1)
    dice = 2*inter / (sums + eps)                  # (B,)
    base = 1 - dice                               # (B,)

    # ————— 2. 质心距离权重 —————
    # 生成坐标张量 (H*W*D, 3)
    coords = torch.stack(torch.meshgrid(
        torch.arange(H, device=input.device),
        torch.arange(W, device=input.device),
        torch.arange(D, device=input.device),
        indexing="ij"), dim=-1).view(-1,3).float()  # (H*W*D,3)
    # 计算批量质心
    p_probs = input.view(B, -1, 1)                # (B,HW*D,1)
    g_probs = target.view(B, -1, 1)
    c_pred = (coords[None] * p_probs).sum(1) / (p_probs.sum(1)+eps)  # (B,3)
    c_gt   = (coords[None] * g_probs).sum(1) / (g_probs.sum(1)+eps)  # (B,3)
    dist   = torch.norm(c_pred - c_gt, dim=1)     # (B,)
    d_max  = (H**2 + W**2 + D**2)**0.5
    w_dist = 1 + alpha * (dist / (d_max + eps))   # (B,)

    # ————— 3. 聚类数差（快速近似） —————
    def count_clusters(x: torch.Tensor,
                    threshold: float = 0.5,
                    min_size: int = 20) -> torch.Tensor:
        """
        对输入 x（shape: B×H×W×D）：
        1. 二值化 (x > threshold)
        2. 去除连通体积 < min_size
        3. 统计连通组件数
        返回 shape 为 (B,) 的整型张量。
        """
        B = x.shape[0]
        counts = []
        # 暂转到 CPU+NumPy 处理
        x_np = (x > threshold).cpu().numpy()
        for i in range(B):
            mask = x_np[i]  # H×W×D 二值掩码
            # 去除小连通域
            clean = remove_small_objects(mask, min_size=min_size, connectivity=1)
            # 标记连通组件
            labels = label(clean, connectivity=1)
            counts.append(int(labels.max()))
        return torch.tensor(counts, device=x.device, dtype=torch.float32)

    # 在主函数中替换原有的 count_peaks：
    n_pred = count_clusters(input, threshold=0.5, min_size=20)
    n_gt   = count_clusters(target, threshold=0.5, min_size=20)
    w_cnt  = 1 + 0.5 * torch.abs(n_pred - n_gt)   # (B,)
    # ————— 4. 单侧漏分场景权重 —————
    zero_pred = (p.sum(1) < eps) & (g.sum(1) > eps)
    zero_gt   = (g.sum(1) < eps) & (p.sum(1) > eps)
    w_leak = torch.where(zero_pred|zero_gt,
                         (1 - dice) + 1,
                         torch.ones_like(dice))        # (B,)

    # ————— 5. 综合 & clamp —————
    w = torch.stack([w_leak, w_dist, w_cnt], dim=1).max(1)[0]  # 取三者中最强惩罚
    w = torch.clamp(w, max=3.0, min=1)                              # w ≤ 3
    return (base * w).mean()

def gram_matrix(input):
    a, b, c, d, e = input.size()
    features = input.view(a * b, c * d * e)
    G = torch.mm(features, features.t())  # compute the gram product
    return G.div(a * b * c * d * e)

def get_style_loss(sf, sm):
    g_f = gram_matrix(sf)
    g_m = gram_matrix(sm)
    channels = sf.size(1)
    size     = sf.size(2)*sf.size(3) 
    sloss = torch.sum(torch.square(g_f-g_m)) / (4.0 * (channels ** 2) * (size ** 2))
    return sloss*0.0001

def mix_matrix(f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
    """
    跨层混合：Gram(f1, f2)  = f1(B,C,N) · f2(B,N,C) / N
    """
    B, C, H, W = f1.shape
    N   = H * W
    f1  = f1.flatten(2)               # (B,C,N)
    f2  = f2.flatten(2)               # (B,C,N)
    G   = torch.bmm(f1, f2.transpose(1, 2)) / N   # (B,C,C)
    return G

def get_GS_loss(Gs_f, Gs_m):
    Ms_f = []
    Ms_m = []
    for i in range(len(Gs_f)):
        Ms_f.append(mix_matrix(Gs_f[i], Gs_f[i-1]))
        Ms_m.append(mix_matrix(Gs_m[i], Gs_m[i-1]))
    channels = Gs_f[0].size(1)
    size     = Gs_f[0].size(2)*Gs_f[0].size(3)

    sloss = 0.0
    for M_f, M_m in zip(Ms_f, Ms_m):
        sloss = sloss + torch.sum(torch.square(M_f-M_m)) / (4.0 * (channels ** 2) * (size ** 2))
    
    return sloss*10e1

def coral_loss(t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """
    输入:
        t, s : (B,C,H,W)  —— Teacher 和 Student 同层特征
    返回:
        标量张量，可反传
    """
    B, C, H, W = t.shape
    N = H * W

    t_flat = t.flatten(2)            # (B,C,N)
    s_flat = s.flatten(2)

    # 通道均值
    mu_t, mu_s = t_flat.mean(-1), s_flat.mean(-1)

    # 通道协方差
    ct = (t_flat @ t_flat.transpose(1, 2)) / (N - 1)   # (B,C,C)
    cs = (s_flat @ s_flat.transpose(1, 2)) / (N - 1)

    loss_mu = F.mse_loss(mu_t, mu_s, reduction="mean")
    loss_ct = F.mse_loss(ct , cs , reduction="mean")
    return loss_mu + loss_ct


# ------------------------------------------------------------
# 2.  总蒸馏损失 = 仅 CORAL
# ------------------------------------------------------------
def get_distill_loss_coral(
    teacher_feats: Sequence[torch.Tensor],   # List[(B,C,H,W)]
    student_feats: Sequence[torch.Tensor],   # List[(B,C,H,W)]
    alpha: float = 0.1                       # λ_coral
) -> torch.Tensor:
    """
    遍历所有对应层求 CORAL‑Loss，取平均后乘 α
    """
    assert len(teacher_feats) == len(student_feats), "层数需一致"
    device = teacher_feats[0].device

    loss = torch.zeros(1, device=device)
    for t_feat, s_feat in zip(teacher_feats, student_feats):
        loss = loss + coral_loss(t_feat, s_feat)

    loss = (loss / len(teacher_feats)) * alpha
    return loss

def unet_Co_loss(config, batch_pred_full, content_full, cls_probs_full, batch_y, batch_pred_missing, content_missing, cls_probs_missing, sf, sm, epoch, cls):
    loss_dict = {}
    # loss_dict['wt_dc_loss']  = dice_loss(batch_pred_full[:, 0], batch_y[:, 0])  # whole tumor
    # loss_dict['tc_dc_loss'] = dice_loss(batch_pred_full[:, 1], batch_y[:, 1])  # tumore core
    # loss_dict['et_dc_loss']  = dice_loss(batch_pred_full[:, 2], batch_y[:, 2])  # enhance tumor
    
    # loss_dict['wt_miss_dc_loss']  = dice_loss(batch_pred_missing[:, 0], batch_y[:, 0])  # whole tumor
    # loss_dict['tc_miss_dc_loss'] = dice_loss(batch_pred_missing[:, 1], batch_y[:, 1])  # tumore core
    # loss_dict['et_miss_dc_loss']  = dice_loss(batch_pred_missing[:, 2], batch_y[:, 2])  # enhance tumor

    # ——— 计算权重，不做 mean ———
    # w_full, w_miss: [B,3]
    w_full = torch.abs(cls_probs_full    - cls.float()) + 1.0
    w_miss = torch.abs(cls_probs_missing - cls.float()) + 1.0

    # ——— 逐通道计算加权 Dice loss ———
    # 我们让 dice_loss 返回每个样本的损失：reduction='none'
    # 然后乘以对应的 w[:,i]，再对 batch 维度求 mean

    # 全模态 WT
    dice_full_wt = dice_loss(batch_pred_full[:,0], batch_y[:,0], reduction='none')  # [B]
    loss_dict['wt_dc_loss'] = (w_full[:,0] * dice_full_wt).mean()

    # 全模态 TC
    dice_full_tc = dice_loss(batch_pred_full[:,1], batch_y[:,1], reduction='none')
    loss_dict['tc_dc_loss'] = (w_full[:,1] * dice_full_tc).mean()

    # 全模态 ET
    dice_full_et = dice_loss(batch_pred_full[:,2], batch_y[:,2], reduction='none')
    loss_dict['et_dc_loss'] = (w_full[:,2] * dice_full_et).mean()

    # loss_dict['bg_dc_loss'] = dice_loss(batch_pred_full[:,3], batch_y[:,3], reduction='none')

    # 缺模态 WT
    dice_miss_wt = dice_loss(batch_pred_missing[:,0], batch_y[:,0], reduction='none')
    loss_dict['wt_miss_dc_loss'] = (w_miss[:,0] * dice_miss_wt).mean()

    # 缺模态 TC
    dice_miss_tc = dice_loss(batch_pred_missing[:,1], batch_y[:,1], reduction='none')
    loss_dict['tc_miss_dc_loss'] = (w_miss[:,1] * dice_miss_tc).mean()

    # 缺模态 ET
    dice_miss_et = dice_loss(batch_pred_missing[:,2], batch_y[:,2], reduction='none')
    loss_dict['et_miss_dc_loss'] = (w_miss[:,2] * dice_miss_et).mean()

    # loss_dict['bg_miss_loss'] = dice_loss(batch_pred_full[:,3], batch_y[:,3], reduction='none')

    ## Dice loss predictions
    loss_dict['loss_dc'] = loss_dict['wt_dc_loss'] + loss_dict['tc_dc_loss'] + loss_dict['et_dc_loss']
    loss_dict['loss_miss_dc'] = loss_dict['wt_miss_dc_loss'] + loss_dict['tc_miss_dc_loss'] + loss_dict['et_miss_dc_loss']
    
    ## Consistency loss
    loss_dict['wt_mse_loss']  = F.mse_loss(batch_pred_full[:, 0], batch_pred_missing[:, 0], reduction='mean') 
    loss_dict['tc_mse_loss'] = F.mse_loss(batch_pred_full[:, 1], batch_pred_missing[:, 1], reduction='mean') 
    loss_dict['et_mse_loss']  = F.mse_loss(batch_pred_full[:, 2], batch_pred_missing[:, 2], reduction='mean') 
    # loss_dict['bg_mse_loss']  = F.mse_loss(batch_pred_full[:, 3], batch_pred_missing[:, 3], reduction='mean') 
    loss_dict['consistency_loss'] = loss_dict['wt_mse_loss'] + loss_dict['tc_mse_loss'] + loss_dict['et_mse_loss']
    
    ## Content loss
    loss_dict['content_loss'] = F.mse_loss(content_full, content_missing, reduction='mean')
    
    ## Style loss
    sloss = get_style_loss(sf, sm)
    
    
    ## Weights for each loss the lamba values
    weight_content = float(config['weight_content'])
    weight_missing = float(config['weight_mispath'])
    weight_full    = 1 - float(config['weight_mispath'])
    
    weight_consistency = get_current_consistency_weight(epoch)

    bce = nn.BCELoss()
    loss_cls_full    = bce(cls_probs_full,    cls.float())
    loss_cls_missing = bce(cls_probs_missing, cls.float())

    loss_dict['loss_Co'] = weight_full * loss_dict['loss_dc'] + weight_missing * loss_dict['loss_miss_dc'] + \
                            weight_consistency * loss_dict['consistency_loss'] + weight_content * loss_dict['content_loss']+sloss+loss_cls_full+loss_cls_missing
    
    return loss_dict

def unet_Co_loss_wo_LPGM(config, batch_pred_full, content_full, cls_probs_full, batch_y, batch_pred_missing, content_missing, cls_probs_missing, sf, sm, epoch, cls):
    loss_dict = {}
    loss_dict['wt_dc_loss']  = dice_loss(batch_pred_full[:, 0], batch_y[:, 0])  # whole tumor
    loss_dict['tc_dc_loss'] = dice_loss(batch_pred_full[:, 1], batch_y[:, 1])  # tumore core
    loss_dict['et_dc_loss']  = dice_loss(batch_pred_full[:, 2], batch_y[:, 2])  # enhance tumor
    
    loss_dict['wt_miss_dc_loss']  = dice_loss(batch_pred_missing[:, 0], batch_y[:, 0])  # whole tumor
    loss_dict['tc_miss_dc_loss'] = dice_loss(batch_pred_missing[:, 1], batch_y[:, 1])  # tumore core
    loss_dict['et_miss_dc_loss']  = dice_loss(batch_pred_missing[:, 2], batch_y[:, 2])  # enhance tumor

    ## Dice loss predictions
    loss_dict['loss_dc'] = loss_dict['wt_dc_loss'] + loss_dict['tc_dc_loss'] + loss_dict['et_dc_loss']
    loss_dict['loss_miss_dc'] = loss_dict['wt_miss_dc_loss'] + loss_dict['tc_miss_dc_loss'] + loss_dict['et_miss_dc_loss']
    
    ## Consistency loss
    loss_dict['wt_mse_loss']  = F.mse_loss(batch_pred_full[:, 0], batch_pred_missing[:, 0], reduction='mean') 
    loss_dict['tc_mse_loss'] = F.mse_loss(batch_pred_full[:, 1], batch_pred_missing[:, 1], reduction='mean') 
    loss_dict['et_mse_loss']  = F.mse_loss(batch_pred_full[:, 2], batch_pred_missing[:, 2], reduction='mean') 
    loss_dict['consistency_loss'] = loss_dict['wt_mse_loss'] + loss_dict['tc_mse_loss'] + loss_dict['et_mse_loss']
    
    ## Content loss
    loss_dict['content_loss'] = F.mse_loss(content_full, content_missing, reduction='mean')
    
    ## Style loss
    sloss = get_style_loss(sf, sm)
    
    
    ## Weights for each loss the lamba values
    weight_content = float(config['weight_content'])
    weight_missing = float(config['weight_mispath'])
    weight_full    = 1 - float(config['weight_mispath'])
    
    weight_consistency = get_current_consistency_weight(epoch)
    loss_dict['loss_Co'] = weight_full * loss_dict['loss_dc'] + weight_missing * loss_dict['loss_miss_dc'] + \
                            weight_consistency * loss_dict['consistency_loss'] + weight_content * loss_dict['content_loss']+sloss
    
    return loss_dict

def unet_Co_loss_divide(config, batch_pred_full, content_full, batch_y, batch_pred_missing, content_missing, sf, sm, epoch):
    loss_dict = {}
    loss_dict['net_dc_loss']  = dice_loss(batch_pred_full[:, 0], batch_y[:, 0])  # whole tumor
    loss_dict['snfh_dc_loss'] = dice_loss(batch_pred_full[:, 1], batch_y[:, 1])  # tumore core
    loss_dict['et_dc_loss']  = dice_loss(batch_pred_full[:, 2], batch_y[:, 2])  # enhance tumor
    
    loss_dict['net_miss_dc_loss']  = dice_loss(batch_pred_missing[:, 0], batch_y[:, 0])  # whole tumor
    loss_dict['snfh_miss_dc_loss'] = dice_loss(batch_pred_missing[:, 1], batch_y[:, 1])  # tumore core
    loss_dict['et_miss_dc_loss']  = dice_loss(batch_pred_missing[:, 2], batch_y[:, 2])  # enhance tumor

    ## Dice loss predictions
    loss_dict['loss_dc'] = loss_dict['net_dc_loss'] + loss_dict['snfh_dc_loss'] + loss_dict['et_dc_loss']
    loss_dict['loss_miss_dc'] = loss_dict['net_miss_dc_loss'] + loss_dict['snfh_miss_dc_loss'] + loss_dict['et_miss_dc_loss']
    
    ## Consistency loss
    loss_dict['net_mse_loss']  = F.mse_loss(batch_pred_full[:, 0], batch_pred_missing[:, 0], reduction='mean') 
    loss_dict['snfh_mse_loss'] = F.mse_loss(batch_pred_full[:, 1], batch_pred_missing[:, 1], reduction='mean')  
    loss_dict['et_mse_loss']  = F.mse_loss(batch_pred_full[:, 2], batch_pred_missing[:, 2], reduction='mean') 
    loss_dict['consistency_loss'] = loss_dict['net_mse_loss'] + loss_dict['snfh_mse_loss'] + loss_dict['et_mse_loss']
    
    ## Content loss
    loss_dict['content_loss'] = F.mse_loss(content_full, content_missing, reduction='mean')
    
    ## Style loss
    sloss = get_style_loss(sf, sm)
    
    ## Weights for each loss the lamba values
    weight_content = float(config['weight_content'])
    weight_missing = float(config['weight_mispath'])
    weight_full    = 1 - float(config['weight_mispath'])
    
    weight_consistency = get_current_consistency_weight(epoch)
    loss_dict['loss_Co'] = weight_full * loss_dict['loss_dc'] + weight_missing * loss_dict['loss_miss_dc'] + \
                            weight_consistency * loss_dict['consistency_loss'] + weight_content * loss_dict['content_loss']+sloss
    
    return loss_dict

def unet_Co_loss_smunet(config, batch_pred_full, content_full, batch_y, batch_pred_missing, content_missing, epoch):
    loss_dict = {}
    loss_dict['net_dc_loss']  = dice_loss(batch_pred_full[:, 0], batch_y[:, 0])  # whole tumor
    loss_dict['snfh_dc_loss'] = dice_loss(batch_pred_full[:, 1], batch_y[:, 1])  # tumore core
    loss_dict['et_dc_loss']  = dice_loss(batch_pred_full[:, 2], batch_y[:, 2])  # enhance tumor
    loss_dict['bg_dc_loss']  = dice_loss(batch_pred_full[:, 3], batch_y[:, 3])  # enhance tumor
    
    loss_dict['net_miss_dc_loss']  = dice_loss(batch_pred_missing[:, 0], batch_y[:, 0])  # whole tumor
    loss_dict['snfh_miss_dc_loss'] = dice_loss(batch_pred_missing[:, 1], batch_y[:, 1])  # tumore core
    loss_dict['et_miss_dc_loss']  = dice_loss(batch_pred_missing[:, 2], batch_y[:, 2])  # enhance tumor
    loss_dict['bg_miss_dc_loss']  = dice_loss(batch_pred_full[:, 3], batch_y[:, 3])  # enhance tumor

    ## Dice loss predictions
    loss_dict['loss_dc'] = loss_dict['net_dc_loss'] + loss_dict['snfh_dc_loss'] + loss_dict['et_dc_loss'] + loss_dict['bg_dc_loss']
    loss_dict['loss_miss_dc'] = loss_dict['net_miss_dc_loss'] + loss_dict['snfh_miss_dc_loss'] + loss_dict['et_miss_dc_loss'] + loss_dict['bg_miss_dc_loss']
    
    ## Consistency loss
    loss_dict['net_mse_loss']  = F.mse_loss(batch_pred_full[:, 0], batch_pred_missing[:, 0], reduction='mean') 
    loss_dict['snfh_mse_loss'] = F.mse_loss(batch_pred_full[:, 1], batch_pred_missing[:, 1], reduction='mean') 
    loss_dict['et_mse_loss']  = F.mse_loss(batch_pred_full[:, 2], batch_pred_missing[:, 2], reduction='mean') 
    loss_dict['bg_mse_loss']  = dice_loss(batch_pred_full[:, 3], batch_y[:, 3])  # enhance tumor
    loss_dict['consistency_loss'] = loss_dict['net_mse_loss'] + loss_dict['snfh_mse_loss'] + loss_dict['et_mse_loss'] + loss_dict['bg_mse_loss']
    
    ## Content loss
    loss_dict['content_loss'] = F.mse_loss(content_full, content_missing, reduction='mean')
    
    ## Weights for each loss the lamba values
    weight_content = float(config['weight_content'])
    weight_missing = float(config['weight_mispath'])
    weight_full    = 1 - float(config['weight_mispath'])
    
    weight_consistency = get_current_consistency_weight(epoch)
    loss_dict['loss_Co'] = weight_full * loss_dict['loss_dc'] + weight_missing * loss_dict['loss_miss_dc'] + \
                            weight_consistency * loss_dict['consistency_loss'] + weight_content * loss_dict['content_loss']
    
    return loss_dict

def get_losses(config):
    losses = {}
    losses['co_loss'] = unet_Co_loss
    losses['unetr_loss'] = Unetr_Loss()
    losses['slkd_loss'] = DistillKL_logit_stand()
    losses['gsm_loss'] = get_GS_loss
    losses['coral_loss'] = get_distill_loss_coral
    losses['re_l2_loss'] = nn.MSELoss()  # L2损失
    return losses

def get_losses_woLPGM(config):
    losses = {}
    losses['co_loss'] = unet_Co_loss_wo_LPGM
    losses['unetr_loss'] = Unetr_Loss()
    losses['slkd_loss'] = DistillKL_logit_stand()
    losses['gsm_loss'] = get_GS_loss
    losses['coral_loss'] = get_distill_loss_coral
    losses['re_l2_loss'] = nn.MSELoss()  # L2损失
    return losses

def get_losses_divide(config):
    losses = {}
    losses['co_loss'] = unet_Co_loss_divide
    losses['unetr_loss'] = Unetr_Loss()
    losses['slkd_loss'] = DistillKL_logit_stand()
    losses['gsm_loss'] = ExtremeValueDistillation(
                                    use_max=True,
                                    use_mean=True,
                                    alpha=10e7,
                                    with_proj=True,
                                    temperature=2.0,
                                    mask_ratio=0.25,         # 25% 随机遮挡
                                    mask_mode="spatial"      # 也可以 "channel"
                                )
    losses['co_smunet_loss'] = unet_Co_loss_smunet
    return losses


class Dice(torch.nn.Module):
    def __init__(self, smooth=1.0):
        super(Dice, self).__init__()
        self.smooth = smooth

    def forward(self, prediction, target):
        prediction = torch.Tensor(prediction)
        target = torch.Tensor(target)
        iflat = prediction.reshape(-1)
        tflat = target.reshape(-1)
        intersection = (iflat * tflat).sum()

        return ((2.0 * intersection + self.smooth) / (iflat.sum() + tflat.sum() + self.smooth)).numpy()

class IoU(nn.Module):
    def __init__(self, smooth=1.0):
        super(IoU, self).__init__()
        self.smooth = smooth

    def forward(self, prediction, target):
        prediction = torch.Tensor(prediction)
        target = torch.Tensor(target)
        # Flatten tensors
        iflat = prediction.reshape(-1)
        tflat = target.reshape(-1)
        # Calculate intersection and union
        intersection = (iflat * tflat).sum()
        union = iflat.sum() + tflat.sum() - intersection
        # Compute IoU
        return ((intersection + self.smooth) / (union + self.smooth)).numpy()

class Sensitivity(nn.Module):
    def __init__(self, smooth=1.0):
        super(Sensitivity, self).__init__()
        self.smooth = smooth

    def forward(self, prediction, target):
        prediction = torch.Tensor(prediction)
        target = torch.Tensor(target)
        # Flatten tensors
        iflat = prediction.reshape(-1)
        tflat = target.reshape(-1)
        # True positives and false negatives
        tp = (iflat * tflat).sum()
        fn = ((1 - iflat) * tflat).sum()
        # Compute sensitivity (recall)
        return ((tp + self.smooth) / (tp + fn + self.smooth)).numpy()
    
class Unetr_Loss(nn.Module):
    def __init__(self, weight_factor=[1.0,1.0,1.0,1.0]):
        super(Unetr_Loss, self).__init__()
        self.weight_factor = weight_factor  # 可以调整每个特征层的权重

    def forward(self, teacher_fs, student_fs):
        """
        计算教师和学生网络特征列表之间的L2损失

        参数:
        - teacher_fs: 教师网络的特征列表 (list of tensors)
        - student_fs: 学生网络的特征列表 (list of tensors)

        返回:
        - L2损失的总和
        """
        total_loss = 0.0
        for i, (t, s) in enumerate(zip(teacher_fs, student_fs)):
            # 确保teacher_fs和student_fs的形状相同
            assert t.shape == s.shape, f"Feature map shapes must match: {t.shape} vs {s.shape}"

            # 计算L2损失，使用F.mse_loss作为L2范数的实现
            loss = F.mse_loss(s, t, reduction='mean')
            total_loss = total_loss + loss * self.weight_factor[i]

        return total_loss

if __name__ == '__main__':
    # 输入logit的大小为 (1, 4, 160, 192, 128)
    input_shape = (1, 4, 160, 192, 128)
    
    y_1 = torch.randn(input_shape).cuda() 
    y_2 = torch.randn(input_shape).cuda()  
    y_3 = torch.randn(input_shape).cuda() 

    

    s_1 = torch.randn(input_shape).cuda() 
    s_2 = torch.randn(input_shape).cuda()  
    s_3 = torch.randn(input_shape).cuda() 
    
    
