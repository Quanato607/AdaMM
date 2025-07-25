#!/usr/bin/env python3
# encoding: utf-8
from .unet import Unet_module, UNet3D
from .smunetr import SMUNetr
from .AdaMMKD import AdaMMKD
from .AdaMMKD_old import AdaMMKD_old
from .AdaMMKD_wo_GARM import AdaMMKD_wo_GARM
from .AdaMMKD_wo_LPGM import AdaMMKD_wo_LPGM

def build_model(inp_shape = (160, 192, 128), inp_dim1=4, inp_dim2 = 1):
    model_full    = UNet3D(inp_shape,
                      in_channels=inp_dim1,
                      out_channels=4,
                      init_channels=16,
                      p=0.2)

    model_missing = UNet3D(inp_shape,
                      in_channels=inp_dim2,
                      out_channels=4,
                      init_channels=16,
                      p=0.2)
    return model_full, model_missing

def build_smunetr(inp_shape = (160, 192, 128), inp_dim1=4, inp_dim2 = 1):
    model_full    = SMUNetr(inp_shape,
                      in_channels=inp_dim1,
                      out_channels=4,
                      init_channels=16,
                      p=0.2, unetr=True)

    model_missing = SMUNetr(inp_shape,
                      in_channels=inp_dim2,
                      out_channels=4,
                      init_channels=16,
                      p=0.2, unetr=True)
    return model_full, model_missing

def build_AdaMMKD_old(inp_shape = (160, 192, 128), inp_dim1=4, inp_dim2 = 1):
    model_full    = AdaMMKD_old(inp_shape,
                      in_channels=inp_dim1,
                      out_channels=4,
                      init_channels=16,
                      p=0.2)

    model_missing = AdaMMKD_old(inp_shape,
                      in_channels=inp_dim2,
                      out_channels=4,
                      init_channels=16,
                      p=0.2)
    return model_full, model_missing

def build_AdaMMKD(inp_shape = (160, 192, 128), inp_dim1=4, inp_dim2 = 1):
    model_full    = AdaMMKD(inp_shape,
                      in_channels=inp_dim1,
                      out_channels=4,
                      init_channels=16,
                      p=0.2)

    model_missing = AdaMMKD(inp_shape,
                      in_channels=inp_dim2,
                      out_channels=4,
                      init_channels=16,
                      p=0.2)
    return model_full, model_missing

def build_AdaMMKD_wo_GARM(inp_shape = (160, 192, 128), inp_dim1=4, inp_dim2 = 1):
    model_full    = AdaMMKD_wo_GARM(inp_shape,
                      in_channels=inp_dim1,
                      out_channels=4,
                      init_channels=16,
                      p=0.2)

    model_missing = AdaMMKD_wo_GARM(inp_shape,
                      in_channels=inp_dim2,
                      out_channels=4,
                      init_channels=16,
                      p=0.2)
    return model_full, model_missing

def build_AdaMMKD_wo_LPGM(inp_shape = (160, 192, 128), inp_dim1=4, inp_dim2 = 1):
    model_full    = AdaMMKD_wo_LPGM(inp_shape,
                      in_channels=inp_dim1,
                      out_channels=4,
                      init_channels=16,
                      p=0.2)

    model_missing = AdaMMKD_wo_LPGM(inp_shape,
                      in_channels=inp_dim2,
                      out_channels=4,
                      init_channels=16,
                      p=0.2)
    return model_full, model_missing