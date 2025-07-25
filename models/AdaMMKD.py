import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product
import random
from .AdaGEM import AdaptiveGraphEnhanceModule

class Adapter(nn.Module):
    def __init__(self, channels, bottleneck=64, p=0.1):
        super().__init__()
        self.down = nn.Conv3d(channels, bottleneck, kernel_size=1)
        self.act  = nn.GELU()
        self.up   = nn.Conv3d(bottleneck, channels, kernel_size=1)
        self.drop = nn.Dropout(p)
    def forward(self, x):
        # x: [B, C, D, H, W]
        z = self.down(x)
        z = self.act(z)
        z = self.up(z)
        z = self.drop(z)
        return x + z


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_groups=8):
        super(BasicBlock, self).__init__()
        self.gn1 = nn.GroupNorm(n_groups, in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.gn2 = nn.GroupNorm(n_groups, in_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))

    def forward(self, x):
        residul = x
        x = self.relu1(self.gn1(x))
        x = self.conv1(x)

        x = self.relu2(self.gn2(x))
        x = self.conv2(x)
        x = x + residul

        return x

class AdaMMKD(nn.Module):
    def __init__(self, input_shape, in_channels=4, out_channels=3, init_channels=32, p=0.2):
        super(AdaMMKD, self).__init__()
        self.input_shape   = input_shape
        self.in_channels   = in_channels
        self.out_channels  = out_channels
        self.init_channels = init_channels

        # 构建编码器和解码器
        self.make_encoder()
        self.make_decoder()
        self.dropout = nn.Dropout(p=p)

        # ===== 为每种非全模态 mask 构建专属 encoder/decoder Adapter 列表 =====
        self.enc_adapters = nn.ModuleDict()
        self.dec_adapters = nn.ModuleDict()
        self.convc_ = nn.ModuleDict()
        self.convco_ = nn.ModuleDict()
        for combo in product([0,1], repeat=self.in_channels):
            if all(combo):  # 全模态不需要 Adapter
                continue
            key = ''.join(str(bit) for bit in combo)
            # 三个编码器阶段 Adapter·1
            self.enc_adapters[key] = nn.ModuleList([
                Adapter(self.init_channels * 2, bottleneck=64, p=p),  # after ds1
                Adapter(self.init_channels * 4, bottleneck=64, p=p),  # after ds2
                Adapter(self.init_channels * 8, bottleneck=64, p=p),   # after ds3
                Adapter(self.init_channels * 8, bottleneck=64, p=p)   # after ds3
            ])
            # 三个解码器阶段 Adapter
            self.dec_adapters[key] = nn.ModuleList([
                Adapter(self.init_channels * 4, bottleneck=64, p=p),  # after up4convb
                Adapter(self.init_channels * 2, bottleneck=64, p=p),  # after up3convb
                Adapter(self.init_channels,     bottleneck=64, p=p)   # after up2
            ])
            self.convc_[key]    = nn.Conv3d(self.init_channels*20, self.init_channels*8, 1)
            self.convco_[key]    = nn.Conv3d(self.init_channels*16, self.init_channels*8, 1)

            self.cls_conv1 = nn.Conv3d(self.init_channels, self.init_channels, kernel_size=3, padding=1)
            self.cls_bn1   = nn.BatchNorm3d(self.init_channels)
            self.cls_conv2 = nn.Conv3d(self.init_channels, 3, kernel_size=1)
            self.cls_pool  = nn.AdaptiveAvgPool3d(1)
            self.cls_softmax = nn.Softmax(dim=1)

    def make_encoder(self):
        C = self.init_channels
        self.conv1a = nn.Conv3d(self.in_channels, C, 3, padding=1)
        self.conv1b = BasicBlock(C, C)
        self.ds1    = nn.Conv3d(C, C*2, 3, stride=2, padding=1)
        self.conv2a = BasicBlock(C*2, C*2)
        self.conv2b = BasicBlock(C*2, C*2)
        self.ds2    = nn.Conv3d(C*2, C*4, 3, stride=2, padding=1)
        self.conv3a = BasicBlock(C*4, C*4)
        self.conv3b = BasicBlock(C*4, C*4)
        self.ds3    = nn.Conv3d(C*4, C*8, 3, stride=2, padding=1)
        self.conv4a = BasicBlock(C*8, C*8)
        self.conv4b = BasicBlock(C*8, C*8)
        self.conv4c = BasicBlock(C*8, C*8)
        self.conv4d = BasicBlock(C*8, C*8)

    def make_decoder(self):
        C = self.init_channels
        self.up4conva = nn.Conv3d(C*8, C*4, 1)
        self.up4      = nn.Upsample(scale_factor=2)
        self.up4convb = BasicBlock(C*4, C*4)
        self.up3conva = nn.Conv3d(C*4, C*2, 1)
        self.up3      = nn.Upsample(scale_factor=2)
        self.up3convb = BasicBlock(C*2, C*2)
        self.up2conva = nn.Conv3d(C*2, C, 1)
        self.up2      = nn.Upsample(scale_factor=2)
        self.up2convb = BasicBlock(C, C)
        self.pool     = nn.MaxPool3d(2)
        self.convc    = nn.Conv3d(C*20, C*8, 1)
        self.convco   = nn.Conv3d(C*16, C*8, 1)
        self.up1conv  = nn.Conv3d(C, self.out_channels, 1)
        self.adagem = AdaptiveGraphEnhanceModule(BatchNorm=nn.BatchNorm3d, dim=self.init_channels * 8, out_dim=self.init_channels * 8, num_clusters=8, dropout=0.1)

    def forward(self, x, mask=[1,1,1,1]):
        # 1. 生成 mask key
        key = ''.join(str(int(b)) for b in mask)

        # 编码器阶段
        c1 = self.conv1a(x); c1 = self.conv1b(c1)
        c1d = self.ds1(c1)
        if key != '1111':
            c1d_ = self.enc_adapters[key][0](c1d) # [2, 64, 96, 64, 80]
            c2 = self.conv2a(c1d_)
        else:
            c2 = self.conv2a(c1d)

        c2 = self.conv2b(c2)
        c2d = self.ds2(c2)
        if key != '1111':
            c2d_ = self.enc_adapters[key][1](c2d) # 128, 48, 32, 40
            c2d_p_ = self.pool(c2d_)
            c2d_p = self.pool(c2d)
            c3 = self.conv3a(c2d_)
        else:
            c2d_p = self.pool(c2d)
            c3 = self.conv3a(c2d)
        
        c3 = self.conv3b(c3)
        c3d = self.ds3(c3)
        if key != '1111':
            c3d_ = self.enc_adapters[key][2](c3d) # [2, 256, 24, 16, 20]
            c4 = self.conv4a(c3d_)
        else:
            c4 = self.conv4a(c3d)
        c4 = self.conv4b(c4)
        c4 = self.conv4c(c4); c4d = self.conv4d(c4)
        if key != '1111':
            c4d_ = self.enc_adapters[key][3](c4) # [2, 256, 24, 16, 20]

        # 构造 style/content
        style   = self.convc(torch.cat([c2d_p, c3d, c4d], dim=1))
        global_Gs = torch.cat([c3d, style, c4d], dim=1)
        content = c4d
        c4d     = self.convco(torch.cat([style, content], dim=1))
        c4d     = self.dropout(c4d)
        if key != '1111':
            style_   = self.convc_[key](torch.cat([c2d_p_, c3d_, c4d_], dim=1))
            content_ = c4d_
            c4d_     = self.convco_[key](torch.cat([style_, content_], dim=1))
            c4d_     = self.dropout(c4d_)
            c4d      = self.adagem(c4d_, c4d)

        # 解码器阶段
        u4 = self.up4conva(c4d); u4 = self.up4(u4); u4 = u4 + c3
        u4 = self.up4convb(u4)
        if key != '1111':
            u4 = self.dec_adapters[key][0](u4)

        u3 = self.up3conva(u4); u3 = self.up3(u3); u3 = u3 + c2
        u3 = self.up3convb(u3)
        if key != '1111':
            u3 = self.dec_adapters[key][1](u3)

        u2 = self.up2conva(u3); u2 = self.up2(u2)
        if key != '1111':
            u2 = self.dec_adapters[key][2](u2)
        u2 = u2 + c1

        # —— 新增：u2 分类分支 —— #
        cls_feat = F.relu(self.cls_bn1(self.cls_conv1(u2)))  # [B, C, D, H, W]
        cls_logits = self.cls_conv2(cls_feat)               # [B, 3, D, H, W]
        cls_logits = self.cls_pool(cls_logits)              # [B, 3, 1, 1, 1]
        cls_logits = cls_logits.view(cls_logits.size(0), 3) # [B, 3]
        cls_probs  = torch.sigmoid(cls_logits)           # [B, 3] 归一化概率

        u2 = self.up2convb(u2)

        logit = self.up1conv(u2)
        uout  = torch.sigmoid(logit)

        return uout, style, content, global_Gs, logit, cls_probs


class Unet_module(nn.Module):

    def __init__(self, input_shape, in_channels=4, out_channels=3, init_channels=32, p=0.2):
        super(Unet_module, self).__init__()
        self.unet = AdaMMKD(input_shape, in_channels, out_channels, init_channels, p)
 
    def forward(self, x):
        uout, style, content = self.unet(x)
        return uout, style, content

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params   

if __name__ == "__main__":
    masks_test = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
         [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
         [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
         [True, True, True, True]]
    input1 = torch.randn(2, 4, 192, 128, 160)  # (N=2, C=4, D=128, H=128, W=128)
    model = AdaMMKD(input_shape=(192, 128, 160))
    total, trainable = count_parameters(model)
    print(f"模型总参数量：{total:,}  ({total/1e6:.2f}M)")
    print(f"可训练参数量：{trainable:,}  ({trainable/1e6:.2f}M)")
    uout, style, content, logit = model(input1, masks_test[random.randint(0, 14)])
    print(uout.shape)