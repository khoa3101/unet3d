import torch
import torch.nn as nn

class ConvDouble(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvDouble, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channel)
        self.conv2 = nn.Conv3d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.res = nn.Identity()

    def forward(self, x):
        x = self.relu(self.norm1(self.conv1(x)))
        residual = self.res(x)
        x = self.relu(self.norm2(self.conv2(x)))
        return x + residual


class DownSample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownSample, self).__init__()

        self.convdouble = ConvDouble(in_channel=in_channel, out_channel=out_channel)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x1 = self.convdouble(x)
        x = self.pool(x1)
        return x, x1


class UpSample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UpSample, self).__init__()
        
        self.upconv = nn.ConvTranspose3d(in_channels=in_channel, out_channels=in_channel//2, kernel_size=2, stride=2)
        self.convdouble = ConvDouble(in_channel=in_channel, out_channel=out_channel)

    def forward(self, x1, x2):
        x1 = self.upconv(x1)
        diff_z = x2.size(-3) - x1.size(-3)
        diff_x = x2.size(-2) - x1.size(-2)
        diff_y = x2.size(-1) - x1.size(-1)
        if x1.size() != x2.size():
            pad = F.pad(x1, [
                diff_z // 2, diff_z - diff_z // 2,
                diff_x // 2, diff_x - diff_x // 2,
                diff_y // 2, diff_y - diff_y // 2,
            ])
        else:
            pad = x1
        x = torch.cat((x2, pad), 1)
        x = self.convdouble(x)
        return x


class UNet3D(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet3D, self).__init__()

        self.down1 = DownSample(n_channels, 32)
        self.down2 = DownSample(32, 64)
        self.down3 = DownSample(64, 128)
        # self.down4 = DownSample(128, 256)

        self.convdouble = ConvDouble(128, 256)
        self.drop = nn.Dropout()

        self.up1 = UpSample(256, 128)
        self.up2 = UpSample(128, 64)
        self.up3 = UpSample(64, 32)
        # self.up4 = UpSample(64, 32)

        self.conv_last = nn.Conv3d(32, n_classes, kernel_size=3, padding=1)
        self.normalize = nn.Softmax(1)
    
    def forward(self, x):
        x, x1 = self.down1(x)
        x, x2 = self.down2(x)
        x, x3 = self.down3(x)
        # x, x4 = self.down4(x)

        x = self.convdouble(x)
        x = self.drop(x)

        x = self.up1(x, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        # x = self.up4(x, x1)
        x = self.conv_last(x)
        x = self.normalize(x)
        return x

    def total_params(self):
        total = sum(p.numel() for p in self.parameters())
        return format(total, ',')

    def total_trainable_params(self):
        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return format(total_trainable, ',')