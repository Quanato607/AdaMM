#!/usr/bin/env python3
# encoding: utf-8
# Code modified from https://github.com/Wangyixinxin/ACN
import glob
import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import random

class Brats2018(Dataset):

    def __init__(self, patients_dir, crop_size, modes, train=True, normalization = True, dataset='brats'):
        self.patients_dir = patients_dir
        self.modes = modes
        self.train = train
        self.crop_size = crop_size
        self.normalization = normalization
        self.dataset = dataset

    def __len__(self):
        return len(self.patients_dir)

    def __getitem__(self, index):
        patient_dir = self.patients_dir[index]
        volumes = []
        modes = list(self.modes) + ['seg']
        p = "-" if self.dataset == 'brats' else '_'
        for mode in modes:
            patient_id = os.path.split(patient_dir)[-1]
            volume_path = os.path.join(patient_dir, patient_id + p + mode + '.nii.gz')
            volume = nib.load(volume_path).get_fdata()
            if not mode == "seg" and self.normalization:
                volume = self.normlize(volume)  # [0, 1.0]
            volumes.append(volume)                  # [h, w, d]
        seg_volume = volumes[-1]
        volumes = volumes[:-1]
        volume, seg_volume = self.aug_sample(volumes, seg_volume)
        # ed_volume = (seg_volume == 2) # peritumoral edema ED
        # net_volume = (seg_volume == 1) # enhancing tumor core NET
        # et_volume = (seg_volume == 4) # enhancing tumor ET
        # bg_volume = (seg_volume == 0)
        
        # seg_volume = [ed_volume, net_volume, et_volume, bg_volume]
        # seg_volume = np.concatenate(seg_volume, axis=0).astype("float32")

        # return (torch.tensor(volume.copy(), dtype=torch.float),
        #         torch.tensor(seg_volume.copy(), dtype=torch.float))
        if self.dataset == 'fets':
            wt_volume = ((seg_volume ==1)|(seg_volume==2)|(seg_volume==4)).astype('uint8')# peritumoral edema ED
            tc_volume = ((seg_volume == 1)|(seg_volume==4)).astype('uint8') # enhancing tumor core NET
            et_volume = ((seg_volume == 4)).astype('uint8') # enhancing tumor ET
            bg_volume = ((seg_volume == 0)).astype('uint8')
        elif self.dataset == 'brats':
            wt_volume = ((seg_volume ==1)|(seg_volume==2)|(seg_volume==3)).astype('uint8')# peritumoral edema ED
            tc_volume = ((seg_volume == 1)|(seg_volume==3)).astype('uint8') # enhancing tumor core NET
            et_volume = ((seg_volume == 3)).astype('uint8') # enhancing tumor ET
            bg_volume = ((seg_volume == 0)).astype('uint8')
        
        seg_volume = [wt_volume, tc_volume, et_volume, bg_volume]
        seg_volume = np.concatenate(seg_volume, axis=0).astype("float32")

        # ---------- cls：各类是否存在 ----------
        cls = torch.tensor(
            [wt_volume.any(), tc_volume.any(), et_volume.any()],
            dtype=torch.float32
        )  # shape == (3,)

        # ---------- 拼接掩码通道 ----------
        mask_volume = np.concatenate(
            [wt_volume, tc_volume, et_volume, bg_volume],
            axis=0
        ).astype(np.float32)  # shape (4, H, W, D)

        # ---------- 转成 Tensor 并返回 ----------
        return (
            torch.tensor(volume.copy(),  dtype=torch.float32),
            torch.tensor(mask_volume.copy(), dtype=torch.float32),
            cls
        )

    def aug_sample(self, volumes, mask):
        """
            Args:
                volumes: list of array, [h, w, d]
                mask: array [h, w, d], segmentation volume
            Ret: x, y: [channel, h, w, d]

        """
        x = np.stack(volumes, axis=0)       # [N, H, W, D]
        y = np.expand_dims(mask, axis=0)    # [channel, h, w, d]

        if self.train:
            # crop volume
            x, y = self.random_crop(x, y)
            if random.random() < 0.5:
                x = np.flip(x, axis=1)
                y = np.flip(y, axis=1)
            if random.random() < 0.5:
                x = np.flip(x, axis=2)
                y = np.flip(y, axis=2)
            if random.random() < 0.5:
                x = np.flip(x, axis=3)
                y = np.flip(y, axis=3)
        else:
            x, y = self.center_crop(x, y)

        return x, y

    def random_crop(self, x, y):
        """
        Args:
            x: 4d array, [channel, h, w, d]
        """
        crop_size = self.crop_size
        height, width, depth = x.shape[-3:]
        sx = random.randint(0, height - crop_size[0] - 1)
        sy = random.randint(0, width - crop_size[1] - 1)
        sz = random.randint(0, depth - crop_size[2] - 1)
        crop_volume = x[:, sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]
        crop_seg = y[:, sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]

        return crop_volume, crop_seg

    def center_crop(self, x, y):
        crop_size = self.crop_size
        height, width, depth = x.shape[-3:]
        sx = (height - crop_size[0] - 1) // 2
        sy = (width - crop_size[1] - 1) // 2
        sz = (depth - crop_size[2] - 1) // 2
        crop_volume = x[:, sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]
        crop_seg = y[:, sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]

        return crop_volume, crop_seg

    def normlize(self, x):
        return (x - x.min()) / (x.max() - x.min())
    
    
    def normlize_brain(self, x, epsilon=1e-8):
        average        = x[np.nonzero(x)].mean()
        std            = x[np.nonzero(x)].std() + epsilon
        mask           = x>0
        sub_mean       = np.where(mask, x-average, x)
        x_normalized   = np.where(mask, sub_mean/std, x)
        return x_normalized
    
class Brats2018_divide(Dataset):

    def __init__(self, patients_dir, crop_size, modes, train=True, normalization = True, dataset='brats'):
        self.patients_dir = patients_dir
        self.modes = modes
        self.train = train
        self.crop_size = crop_size
        self.normalization = normalization
        self.dataset = dataset

    def __len__(self):
        return len(self.patients_dir)

    def __getitem__(self, index):
        patient_dir = self.patients_dir[index]
        volumes = []
        modes = list(self.modes) + ['seg']
        p = "-" if self.dataset == 'brats' else '_'
        for mode in modes:
            patient_id = os.path.split(patient_dir)[-1]
            volume_path = os.path.join(patient_dir, patient_id + p + mode + '.nii.gz')
            volume = nib.load(volume_path).get_fdata()
            if not mode == "seg" and self.normalization:
                volume = self.normlize(volume)  # [0, 1.0]
            volumes.append(volume)                  # [h, w, d]
        seg_volume = volumes[-1]
        volumes = volumes[:-1]
        volume, seg_volume = self.aug_sample(volumes, seg_volume)
        # ed_volume = (seg_volume == 2) # peritumoral edema ED
        # net_volume = (seg_volume == 1) # enhancing tumor core NET
        # et_volume = (seg_volume == 4) # enhancing tumor ET
        # bg_volume = (seg_volume == 0)
        
        # seg_volume = [ed_volume, net_volume, et_volume, bg_volume]
        # seg_volume = np.concatenate(seg_volume, axis=0).astype("float32")

        # return (torch.tensor(volume.copy(), dtype=torch.float),
        #         torch.tensor(seg_volume.copy(), dtype=torch.float))
        if self.dataset == 'fets':
            net_volume = ((seg_volume ==1)).astype('uint8')# peritumoral edema ED
            snfh_volume = ((seg_volume == 2)).astype('uint8') # enhancing tumor core NET
            et_volume = ((seg_volume == 4)).astype('uint8') # enhancing tumor ET
            bg_volume = ((seg_volume == 0)).astype('uint8')
        elif self.dataset == 'brats':
            net_volume = ((seg_volume ==1)).astype('uint8')# peritumoral edema ED
            snfh_volume = ((seg_volume == 2)).astype('uint8') # enhancing tumor core NET
            et_volume = ((seg_volume == 3)).astype('uint8') # enhancing tumor ET
            bg_volume = ((seg_volume == 0)|(seg_volume == 4)).astype('uint8')

        cls = torch.tensor(
            [net_volume.any(), snfh_volume.any(), et_volume.any()],
            dtype=torch.float32
        )  # shape == (3,)
    
        seg_volume = [net_volume, snfh_volume, et_volume, bg_volume]
        seg_volume = np.concatenate(seg_volume, axis=0).astype("float32")

        return (torch.tensor(volume.copy(), dtype=torch.float),
                torch.tensor(seg_volume.copy(), dtype=torch.float),
                cls)

    def aug_sample(self, volumes, mask):
        """
            Args:
                volumes: list of array, [h, w, d]
                mask: array [h, w, d], segmentation volume
            Ret: x, y: [channel, h, w, d]

        """
        x = np.stack(volumes, axis=0)       # [N, H, W, D]
        y = np.expand_dims(mask, axis=0)    # [channel, h, w, d]

        if self.train:
            # crop volume
            x, y = self.random_crop(x, y)
            if random.random() < 0.5:
                x = np.flip(x, axis=1)
                y = np.flip(y, axis=1)
            if random.random() < 0.5:
                x = np.flip(x, axis=2)
                y = np.flip(y, axis=2)
            if random.random() < 0.5:
                x = np.flip(x, axis=3)
                y = np.flip(y, axis=3)
        else:
            x, y = self.center_crop(x, y)

        return x, y

    def random_crop(self, x, y):
        """
        Args:
            x: 4d array, [channel, h, w, d]
        """
        crop_size = self.crop_size
        height, width, depth = x.shape[-3:]
        sx = random.randint(0, height - crop_size[0] - 1)
        sy = random.randint(0, width - crop_size[1] - 1)
        sz = random.randint(0, depth - crop_size[2] - 1)
        crop_volume = x[:, sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]
        crop_seg = y[:, sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]

        return crop_volume, crop_seg

    def center_crop(self, x, y):
        crop_size = self.crop_size
        height, width, depth = x.shape[-3:]
        sx = (height - crop_size[0] - 1) // 2
        sy = (width - crop_size[1] - 1) // 2
        sz = (depth - crop_size[2] - 1) // 2
        crop_volume = x[:, sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]
        crop_seg = y[:, sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]

        return crop_volume, crop_seg

    def normlize(self, x):
        return (x - x.min()) / (x.max() - x.min())
    
    
    def normlize_brain(self, x, epsilon=1e-8):
        average        = x[np.nonzero(x)].mean()
        std            = x[np.nonzero(x)].std() + epsilon
        mask           = x>0
        sub_mean       = np.where(mask, x-average, x)
        x_normalized   = np.where(mask, sub_mean/std, x)
        return x_normalized

def split_dataset(data_root, test_p):
    patients_dir = glob.glob(os.path.join(data_root, "*GG", "Brats18*"))
    patients_dir.sort()
    N = int(len(patients_dir)*test_p)
    train_patients_list =  patients_dir[N:]
    val_patients_list   =  patients_dir[:N]

    return train_patients_list, val_patients_list
    
def make_data_loaders(config):
    if config['dataset'] == 'fets':
        train_list = glob.glob(os.path.join(config['path_to_data'], 'train', 'FeTS*'))
        val_list = glob.glob(os.path.join(config['path_to_data'], 'test', 'FeTS*'))
    elif config['dataset'] == 'brats':
        train_list = glob.glob(os.path.join(config['path_to_data'], 'train', 'BraTS*'))
        val_list = glob.glob(os.path.join(config['path_to_data'], 'test', 'BraTS*'))
    crop_size = np.zeros((3))
    crop_size[0] = config['inputshape'][0]
    crop_size[1] = config['inputshape'][1]
    crop_size[2] = config['inputshape'][2]
    crop_size    = crop_size.astype(np.uint16)
    crop_size    = (160, 192, 128)
    train_ds = Brats2018(train_list, crop_size=crop_size, modes=config['modalities'], train=True, dataset=config['dataset'])
    val_ds = Brats2018(val_list, crop_size=crop_size, modes=config['modalities'], train=False, dataset=config['dataset'])
    loaders = {}
    loaders['train'] = DataLoader(train_ds, batch_size=int(config['batch_size_tr']),
                                  num_workers=4,
                                  pin_memory=True,
                                  shuffle=True)
    loaders['eval'] = DataLoader(val_ds, batch_size=int(config['batch_size_va']),
                                  num_workers=4,
                                  pin_memory=True,
                                  shuffle=False)
    return loaders

def make_data_loaders_divide(config):
    if config['dataset'] == 'fets':
        train_list = glob.glob(os.path.join(config['path_to_data'], 'train', 'FeTS*'))
        val_list = glob.glob(os.path.join(config['path_to_data'], 'test', 'FeTS*'))
    elif config['dataset'] == 'brats':
        train_list = glob.glob(os.path.join(config['path_to_data'], 'train', 'BraTS*'))
        val_list = glob.glob(os.path.join(config['path_to_data'], 'test', 'BraTS*'))
    crop_size = np.zeros((3))
    crop_size[0] = config['inputshape'][0]
    crop_size[1] = config['inputshape'][1]
    crop_size[2] = config['inputshape'][2]
    crop_size    = crop_size.astype(np.uint16)
    crop_size    = (160, 192, 128)
    train_ds = Brats2018_divide(train_list, crop_size=crop_size, modes=config['modalities'], train=True, dataset=config['dataset'])
    val_ds = Brats2018_divide(val_list, crop_size=crop_size, modes=config['modalities'], train=False, dataset=config['dataset'])
    loaders = {}
    loaders['train'] = DataLoader(train_ds, batch_size=int(config['batch_size_tr']),
                                  num_workers=4,
                                  pin_memory=True,
                                  shuffle=True)
    loaders['eval'] = DataLoader(val_ds, batch_size=int(config['batch_size_va']),
                                  num_workers=4,
                                  pin_memory=True,
                                  shuffle=False)
    return loaders

