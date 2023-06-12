import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
    
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# Conditional Unet defined below
class CondINorm(nn.Module):
    def __init__(self, in_channels, z_channels, eps=1e-5):
        super(CondINorm, self).__init__()
        self.eps = eps
        self.shift_conv = nn.Sequential(
            nn.Conv2d(z_channels, in_channels, kernel_size=1, padding=0, bias=True),
            nn.ReLU(True)
        )
        self.scale_conv = nn.Sequential(
            nn.Conv2d(z_channels, in_channels, kernel_size=1, padding=0, bias=True),
            nn.ReLU(True)
        )

    def forward(self, x, z):
        shift = self.shift_conv.forward(z)
        scale = self.scale_conv.forward(z)
        size = x.size()
        x_reshaped = x.view(size[0], size[1], size[2]*size[3])
        mean = x_reshaped.mean(2, keepdim=True)
        var = x_reshaped.var(2, keepdim=True)
        std =  torch.rsqrt(var + self.eps)
        norm_features = ((x_reshaped - mean) * std).view(*size)
        output = norm_features * scale + shift
        return output
    
class CondDoubleConv(nn.Module):
    """(convolution => [CIN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, z_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.norm1 = CondINorm(mid_channels, z_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = CondINorm(out_channels, z_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, z):
        x = self.conv1(x)
        x = self.norm1(x, z)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.norm2(x, z)
        x = self.relu2(x)
        return x

class CondUp(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, z_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = CondDoubleConv(in_channels, out_channels, z_channels, in_channels // 2)

    def forward(self, x1, x2, z):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x, z)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class CUNet(nn.Module):
    def __init__(self, n_channels, n_classes, z_channels, base_factor=32):
        super(CUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.z_channels = z_channels
        self.base_factor = base_factor

        self.inc = DoubleConv(n_channels, base_factor)
        self.down1 = Down(base_factor, 2 * base_factor)
        self.down2 = Down(2 * base_factor, 4 * base_factor)
        self.down3 = Down(4 * base_factor, 8 * base_factor)
        factor = 2
        self.down4 = Down(8 * base_factor, 16 * base_factor // factor)
        self.adain1 = CondINorm(16 * base_factor // factor, z_channels)
        self.up1 = Up(16 * base_factor, 8 * base_factor // factor)
        self.adain2 = CondINorm(8 * base_factor // factor, z_channels)
        self.up2 = Up(8 * base_factor, 4 * base_factor // factor)
        self.adain3 = CondINorm(4 * base_factor // factor, z_channels)
        self.up3 = Up(4 * base_factor, 2 * base_factor // factor)
        self.adain4 = CondINorm(2 * base_factor // factor, z_channels)
        self.up4 = Up(2 * base_factor, base_factor)
        self.outc = OutConv(base_factor, n_classes)

    def forward(self, x, z):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.adain1(x5, z)
        x = self.up1(x, x4)
        x = self.adain2(x, z)
        x = self.up2(x, x3)
        x = self.adain3(x, z)
        x = self.up3(x, x2)
        x = self.adain4(x, z)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits