#!/usr/bin/env python3
# encoding: utf-8
import os
import random
import torch
import warnings
from losses import Dice, Sensitivity, IoU
from binary import hd95
import logging
import numpy as np
from scipy.ndimage import label

criteria = Dice() 
sensitivity = Sensitivity()
iou = IoU()

def init_env(gpu_id='0', seed=42):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    warnings.filterwarnings('ignore')

def get_mask(seg_volume, filter=False):
    seg_volume = seg_volume.detach().cpu().numpy()
    seg_volume = np.squeeze(seg_volume)
    wt_pred = seg_volume[0]
    tc_pred = seg_volume[1]
    et_pred = seg_volume[2]
    wt_mask = np.zeros_like(wt_pred)
    tc_mask = np.zeros_like(tc_pred)
    et_mask = np.zeros_like(et_pred)
    wt_mask[wt_pred > 0.5] = 1
    tc_mask[tc_pred > 0.5] = 1
    et_mask[et_pred > 0.5] = 1
    wt_mask = wt_mask.astype("uint8")
    tc_mask = tc_mask.astype("uint8")
    et_mask = et_mask.astype("uint8")
    if filter:
        wt_mask = connectivity_filter_onehot(wt_mask)
        tc_mask = connectivity_filter_onehot(tc_mask)
        et_mask = connectivity_filter_onehot(et_mask)
    masks = [wt_mask, tc_mask, et_mask]
    return masks

def get_mask_divide(seg_volume, filter=False):
    seg_volume = seg_volume.detach().cpu().numpy()
    seg_volume = np.squeeze(seg_volume)
    net_pred = seg_volume[0]
    snfh_pred = seg_volume[1]
    et_pred = seg_volume[2]
    wt_mask = np.zeros_like(net_pred)
    tc_mask = np.zeros_like(snfh_pred)
    et_mask = np.zeros_like(et_pred)
    wt_mask = ((net_pred > 0.5) | (snfh_pred > 0.5) | (et_pred > 0.5)).astype("uint8")
    tc_mask = ((net_pred > 0.5) | (et_pred > 0.5)).astype("uint8")
    et_mask[et_pred > 0.5] = 1
    wt_mask = wt_mask.astype("uint8")
    tc_mask = tc_mask.astype("uint8")
    et_mask = et_mask.astype("uint8")
    if filter:
        wt_mask = connectivity_filter_onehot(wt_mask)
        tc_mask = connectivity_filter_onehot(tc_mask)
        et_mask = connectivity_filter_onehot(et_mask)
    masks = [wt_mask, tc_mask, et_mask]
    return masks

def get_mask_divide_argmax(seg_volume):
    """
    Args
    ----
    seg_volume: torch.Tensor  shape = (4, D, H, W)
        channel 0: net (necrotic / non‑enhancing core)
        channel 1: snfh (edema)
        channel 2: et  (enhancing tumor)
        channel 3: bg  (background)

    Returns
    -------
    list[np.ndarray]  [wt_mask, tc_mask, et_mask]  每个 uint8, shape = (D, H, W)
    """
    # → numpy, 去梯度 & squeeze 以防批量维
    seg_volume = seg_volume.detach().cpu().numpy()
    seg_volume = np.squeeze(seg_volume)          # (4, D, H, W)

    # 1. argmax 得到类别标签 [0,1,2,3]，shape = (D, H, W)
    pred_lbl = np.argmax(seg_volume, axis=0)

    # 2. 依据标签生成掩码
    wt_mask = (pred_lbl != 3).astype("uint8")             # 非背景即 whole‑tumor
    tc_mask = np.isin(pred_lbl, [0, 2]).astype("uint8")   # net 或 et → tumor‑core
    et_mask = (pred_lbl == 2).astype("uint8")             # 仅 et

    return [wt_mask, tc_mask, et_mask]


def eval_metrics(gt, pred):
    score_wt = criteria(np.where(gt[0]==1, 1, 0), np.where(pred[0]==1, 1, 0))
    score_ct = criteria(np.where(gt[1]==1, 1, 0), np.where(pred[1]==1, 1, 0))
    score_et = criteria(np.where(gt[2]==1, 1, 0), np.where(pred[2]==1, 1, 0))
    
    return score_wt, score_et, score_ct

def eval_sensitivity(gt, pred):
    score_wt = sensitivity(np.where(gt[0]==1, 1, 0), np.where(pred[0]==1, 1, 0))
    score_ct = sensitivity(np.where(gt[1]==1, 1, 0), np.where(pred[1]==1, 1, 0))
    score_et = sensitivity(np.where(gt[2]==1, 1, 0), np.where(pred[2]==1, 1, 0))
    
    return score_wt, score_et, score_ct

def eval_iou(gt, pred):
    score_wt = iou(np.where(gt[0]==1, 1, 0), np.where(pred[0]==1, 1, 0))
    score_ct = iou(np.where(gt[1]==1, 1, 0), np.where(pred[1]==1, 1, 0))
    score_et = iou(np.where(gt[2]==1, 1, 0), np.where(pred[2]==1, 1, 0))
    
    return score_wt, score_et, score_ct

def measure_dice_score(batch_pred, batch_y, divide=False, argmax=False):
    if divide:
        if argmax:
            pred = get_mask_divide_argmax(batch_pred)
            gt = get_mask_divide_argmax(batch_y)
        else:
            pred = get_mask_divide(batch_pred)
            gt = get_mask_divide(batch_y)
    else:
        pred = get_mask(batch_pred)
        gt = get_mask(batch_y)

    score_wt, score_et, score_ct = eval_metrics(gt, pred)

    return score_wt, score_et, score_ct


def measure_hd95(batch_pred, batch_y, divide=False, argmax=False):

    #对于 whole tumor
    # mask_gt = ((gt == 0)|(gt==1)|(gt==2)).astype(int)
    # mask_pred = ((pred == 0)|(pred==1)|(pred==2)).astype(int)
    if divide:
        if argmax:
            pred = get_mask_divide_argmax(batch_pred)
            gt = get_mask_divide_argmax(batch_y)
        else:
            pred = get_mask_divide(batch_pred)
            gt = get_mask_divide(batch_y)
    else:
        pred = get_mask(batch_pred)
        gt = get_mask(batch_y)

    hd95_whole = hd95(pred[0], gt[0])
    #对于tumor core
    hd95_core = hd95(pred[1], gt[1])
    #对于enhancing tumor
    hd95_enh = hd95(pred[2], gt[2])

    return hd95_whole, hd95_enh, hd95_core

def measure_sensitivity_score(batch_pred, batch_y, divide=False, connectivity_filter=False):
    if divide:
        pred = get_mask_divide(batch_pred, connectivity_filter)
        gt = get_mask_divide(batch_y, connectivity_filter)
    else:
        pred = get_mask(batch_pred, connectivity_filter)
        gt = get_mask(batch_y, connectivity_filter)
    score_wt, score_et, score_ct = eval_sensitivity(gt, pred)
    # score = (score_wt + score_et + score_ct) / 3.0
    
    return score_wt, score_et, score_ct

def measure_iou_score(batch_pred, batch_y, divide=False, connectivity_filter=False):
    if divide:
        pred = get_mask_divide(batch_pred, connectivity_filter)
        gt = get_mask_divide(batch_y, connectivity_filter)
    else:
        pred = get_mask(batch_pred, connectivity_filter)
        gt = get_mask(batch_y, connectivity_filter)
    score_wt, score_et, score_ct = eval_iou(gt, pred)
    
    return score_wt, score_et, score_ct

def measure_scores(batch_pred, batch_y, divide=False, connectivity_filter=False):
    if divide:
        pred = get_mask_divide(batch_pred, connectivity_filter)
        gt = get_mask_divide(batch_y, connectivity_filter)
    else:
        pred = get_mask(batch_pred, connectivity_filter)
        gt = get_mask(batch_y, connectivity_filter)

    dice_wt, dice_et, dice_ct = eval_metrics(gt, pred)
    sens_wt, sens_et, sens_ct = eval_sensitivity(gt, pred)
    iou_wt, iou_et, iou_ct = eval_iou(gt, pred)
    hd95_wt= hd95(pred[0], gt[0])
    hd95_tc = hd95(pred[0], gt[0])
    hd95_et = hd95(pred[0], gt[0])

    return dice_wt, dice_et, dice_ct, hd95_wt, hd95_tc, hd95_et, sens_wt, sens_et, sens_ct, iou_wt, iou_et, iou_ct

# def compute_BraTS_HD95(ref, pred):
#     """
#     计算 HD95 指标，内部进行二值化处理。
#     :param ref: 真实标签（NumPy 数组）
#     :param pred: 预测值（NumPy 数组）
#     :return: HD95 距离
#     """
#     # 二值化处理
#     ref = (ref > 0.5) # 真实标签二值化，阈值 0.5
#     pred = (pred > 0.5)# 预测值二值化，阈值 0.5
#
#     ref=ref.cpu().numpy()
#     pred=pred.cpu().numpy()
#
#     ref = np.squeeze(ref)  # 去掉不必要的维度，确保是 (D, H, W)
#     pred = np.squeeze(pred)
#
#     num_ref = np.sum(ref)
#     num_pred = np.sum(pred)
#
#     # 如果真实或预测为空，按约定返回值
#     if num_ref == 0 and num_pred == 0:
#         return 0.0
#     elif num_ref == 0 or num_pred == 0:
#         return 373.13
#
#     # 计算 HD95
#     return hd95(pred, ref, (1, 1, 1))  # 假设 spacing=(1, 1, 1)

def evaluate_sample(batch_pred_full, batch_pred_missing, batch_y):

    pred_nii_full = get_mask(batch_pred_full)
    pred_nii_miss = get_mask(batch_pred_missing)
    gt_nii = get_mask(batch_y)
    
    metric_full  = eval_metrics(gt_nii, pred_nii_full)
    metric_miss  = eval_metrics(gt_nii, pred_nii_miss)
    hd95_full = measure_hd95(gt_nii, pred_nii_full)
    hd95_miss = measure_hd95(gt_nii, pred_nii_miss)
    return metric_full, metric_miss , hd95_full, hd95_miss

def load_old_model(model_full, model_missing, d_style, optimizer, saved_model_path):
    print(f"Constructing model from saved file: {saved_model_path}")
    checkpoint = torch.load(saved_model_path)
    model_full.load_state_dict(checkpoint["model_full"])
    model_missing.load_state_dict(checkpoint["model_missing"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    d_style.load_state_dict(checkpoint["d_style"])
    epoch = checkpoint["epochs"]
    if checkpoint["best_dice"]:
        best_dice = checkpoint["best_dice"]
    elif checkpoint["dice"]:
        best_dice = checkpoint["dice"]
    else:
        best_dice = 0.0

    return model_full, model_missing, d_style, optimizer, epoch, best_dice
        
def to_numpy(tensor):
    if isinstance(tensor, (int, float)):
        return tensor
    else:
        return tensor.data.cpu().numpy()

def get_logging(log_dir):
    # 创建日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # 设置全局日志级别

    # 创建文件处理器，输出到文件
    file_handler = logging.FileHandler(log_dir)
    file_handler.setLevel(logging.INFO)  # 设置文件日志的级别（例如，INFO）

    # 创建控制台处理器，输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)  # 设置控制台日志的级别（例如，DEBUG）

    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 将处理器添加到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

#!/usr/bin/env python3
"""
onehot_cc_filter.py

针对已经做了 one-hot（二值化）输出的分割张量 (B, C, H, W, D)，
对每个通道独立执行 nnU-Net 风格的“保留最大连通组件”后处理。

依赖:
    numpy
    scipy.ndimage.label
"""
import numpy as np
from scipy.ndimage import label

def remove_all_but_largest_component(
    binary_mask: np.ndarray
) -> np.ndarray:
    struct = np.ones((3,) * binary_mask.ndim, dtype=bool)
    labeled, num_features = label(binary_mask, structure=struct)
    if num_features == 0:
        return binary_mask.copy()
    counts = np.bincount(labeled.ravel())
    largest_label = counts[1:].argmax() + 1
    return (labeled == largest_label)

def connectivity_filter_onehot(
    onehot_seg: np.ndarray
) -> np.ndarray:
    """
    对 one-hot（二值化）输出 (B, H, W, D) 做最大连通域过滤。

    参数:
        onehot_seg: np.ndarray, dtype {0,1}, shape = (B, H, W, D)

    返回:
        cleaned: np.ndarray, 同 shape、同 dtype，
                 每个样本仅保留体素数最大的连通组件，其它置 0。
    """
    if onehot_seg.ndim < 2:
        raise ValueError("输入张量维度过低，至少应为 (B, H, ...)")
    B, *spatial = onehot_seg.shape
    cleaned = np.zeros_like(onehot_seg)
    for b in range(B):
        mask = onehot_seg[b].astype(bool)
        if not mask.any():
            continue
        keep = remove_all_but_largest_component(mask)
        cleaned[b] = keep.astype(onehot_seg.dtype)
    return cleaned


if __name__ == "__main__":
    # ========= 用法示例 =========
    # 假设有一个 batch 大小为 2、3 类、体素维度 64×64×64 的 one-hot 输出
    B, H, W, D = 2,192, 160, 128
    onehot_pred = np.random.randint(0, 2, size=(B, H, W, D), dtype=np.uint8)

    # 对每个通道保留其最大连通组件
    cleaned_onehot = connectivity_filter_onehot(onehot_pred)

    print("原 one-hot 形状:", onehot_pred.shape)
    print("处理后 one-hot 形状:", cleaned_onehot.shape)
    # 每通道只有最大连通块保留，其它小斑块均被置 0
