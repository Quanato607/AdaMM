from torch import nn
import torch

class Discriminator3D(nn.Module):
    def __init__(self, in_channels=4, base_filters=16):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(in_channels, base_filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv3d(base_filters, base_filters*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(base_filters*2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv3d(base_filters*2, base_filters*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(base_filters*4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 最后一层输出单通道评分 map（未 sigmoid）
        self.layer4 = nn.Conv3d(base_filters*4, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)
    
def get_style_discriminator(num_classes, ndf=64):
    return nn.Sequential(
        nn.Conv3d(num_classes, ndf, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv3d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv3d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv3d(ndf * 4, 1, kernel_size=(2,3,2), stride=1, padding=0)
    )

def get_reconst_discriminator():
    return Discriminator3D()