import numpy as np
from medpy.metric import hd95 as hd95_medpy
from torch import Tensor
import torch
import torch.nn as nn
import medpy.metric.binary as medpy_metrics

def dice(output, target, eps=1e-5):
    """计算3D数据的Dice系数，适配[B, D, H, W]维度"""
    sum_dims = tuple(range(1, output.ndim))  # 动态获取非批次维度
    inter = torch.sum(output * target, dim=sum_dims) + eps
    union = torch.sum(output, dim=sum_dims) + torch.sum(target, dim=sum_dims) + eps * 2
    dice_score = 2 * inter / union
    return torch.mean(dice_score)


def cal_dice(output, target):
    """
    计算三类结构的Dice系数（根据新标签定义）：
    - ET (4)
    - TC (1, 4)
    - WT (1, 2, 4)
    """
    # output = torch.argmax(output, dim=1)  # 转换为类别索引 [B, D, H, W]
    target = target.long()
    
    # 生成各类别掩码
    et_out = (output == 3).long()
    et_tar = (target == 3).long()
    tc_out = ((output == 1) | (output == 3)).long()
    tc_tar = ((target == 1) | (target == 3)).long()
    wt_out = ((output == 1) | (output == 2) | (output == 3)).long()
    wt_tar = ((target == 1) | (target == 2) | (target == 3)).long()
    
    # 计算Dice系数（处理全零情况）
    dice1 = dice(et_out, et_tar) #if torch.any(et_tar) or torch.any(et_out) else 1.0
    dice2 = dice(tc_out, tc_tar) #if torch.any(tc_tar) or torch.any(tc_out) else 1.0
    dice3 = dice(wt_out, wt_tar) #if torch.any(wt_tar) or torch.any(wt_out) else 1.0
    
    return dice1, dice2, dice3


def cal_hd95(output: Tensor, target: Tensor, spacing=None):
    """计算95%豪斯多夫距离（适配新标签定义）"""
    # output = torch.argmax(output, dim=1)  # 转换为类别索引 [B, D, H, W]
    target = target.long()
    
    # 生成各类别掩码
    et_out = (output == 3).float()
    et_tar = (target == 3).float()
    tc_out = ((output == 1) | (output == 3)).float()
    tc_tar = ((target == 1) | (target == 3)).float()
    wt_out = ((output == 1) | (output == 2) | (output == 3)).float()
    wt_tar = ((target == 1) | (target == 2) | (target == 3)).float()
    
    # 计算HD95
    hd95_ec = compute_hd95(et_out, et_tar, spacing)
    hd95_co = compute_hd95(tc_out, tc_tar, spacing)
    hd95_wt = compute_hd95(wt_out, wt_tar, spacing)
    
    return hd95_ec, hd95_co, hd95_wt


def compute_hd95(pred, gt, spacing=None):
    """计算单个样本的95%豪斯多夫距离"""
    if pred.ndim == 4: pred = pred.squeeze(0)  # 移除批次维度
    if gt.ndim == 4: gt = gt.squeeze(0)
    
    pred_np = pred.cpu().numpy().astype(bool)
    gt_np = gt.cpu().numpy().astype(bool)
    
    try:
        if spacing and len(spacing) == 3:
            hd = hd95_medpy(pred_np, gt_np, voxelspacing=spacing)
        else:
            hd = hd95_medpy(pred_np, gt_np)
    except:
        hd = 373.1287 if (np.any(gt_np) or np.any(pred_np)) else 0.0  # 异常值处理
    return hd


from medpy.metric.binary import jc
def IoU(output, target):
    """计算交并比（适配新标签定义）"""
    # output = torch.argmax(output, dim=1)  # 转换为类别索引 [B, D, H, W]
    target = target.long()
    
    # 生成各类别掩码并转换为NumPy
    et_out = (output == 3).cpu().numpy()
    et_tar = (target == 3).cpu().numpy()
    tc_out = ((output == 1) | (output == 3)).cpu().numpy()
    tc_tar = ((target == 1) | (target == 3)).cpu().numpy()
    wt_out = ((output == 1) | (output == 2) | (output == 3)).cpu().numpy()
    wt_tar = ((target == 1) | (target == 2) | (target == 3)).cpu().numpy()
    
    # 计算IoU
    jc_ec = jc(et_out, et_tar)
    jc_co = jc(tc_out, tc_tar)
    jc_wt = jc(wt_out, wt_tar)
    return jc_ec, jc_co, jc_wt


def sensitivity(output, target, smooth=1e-5):
    """计算敏感度（召回率）"""
    if torch.is_tensor(output): output = output.data.cpu().numpy()
    if torch.is_tensor(target): target = target.data.cpu().numpy()
    intersection = (output * target).sum()
    return (intersection + smooth) / (target.sum() + smooth)


def cal_sensitivity(output, target):
    """计算三类结构的敏感度（适配新标签定义）"""
    # output = torch.argmax(output, dim=1)  # 转换为类别索引 [B, D, H, W]
    target = target.long()
    
    # 生成各类别掩码
    et_out = (output == 3).float()
    et_tar = (target == 3).float()
    tc_out = ((output == 1) | (output == 3)).float()
    tc_tar = ((target == 1) | (target == 3)).float()
    wt_out = ((output == 1) | (output == 2) | (output == 3)).float()
    wt_tar = ((target == 1) | (target == 2) | (target == 3)).float()
    
    # 计算敏感度
    se_ec = sensitivity(et_out, et_tar)
    se_co = sensitivity(tc_out, tc_tar)
    se_wt = sensitivity(wt_out, wt_tar)
    return se_ec, se_co, se_wt


def main():
    """示例演示"""
    torch.manual_seed(49)
    np.random.seed(49)
    
    # 模拟3D医学图像数据
    batch_size, depth, height, width = 1, 146, 169, 139
    output = torch.randn(batch_size, 5, depth, height, width)  # [B, C, D, H, W]
    label = torch.randint(0, 5, (batch_size, depth, height, width))  # [B, D, H, W]
    
    # 计算指标
    dice1, dice2, dice3 = cal_dice(output, label)
    hd95_ec, hd95_co, hd95_wt = cal_hd95(output, label)
    iou1, iou2, iou3 = IoU(output, label)
    sen1, sen2, sen3 = cal_sensitivity(output, label)
    
    # 打印结果
    print(f"Dice (ET/TC/WT): {dice1:.4f}/{dice2:.4f}/{dice3:.4f}")
    print(f"HD95 (ET/TC/WT): {hd95_ec:.4f}/{hd95_co:.4f}/{hd95_wt:.4f}")
    print(f"IoU (ET/TC/WT): {iou1:.4f}/{iou2:.4f}/{iou3:.4f}")
    print(f"Sensitivity (ET/TC/WT): {sen1:.4f}/{sen2:.4f}/{sen3:.4f}")


if __name__ == "__main__":
    main()