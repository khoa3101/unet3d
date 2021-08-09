import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNormReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=True):
        super().__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.act = nn.ReLU(inplace=True) if activation else None
        

class GlobalAverage(nn.Module):
    def forward(self, x):
        return x.mean(axis=(-3, -2, -1))


class SEBlock(nn.Module):
    def __init__(self, channels, ratio=0.5):
        super().__init__()

        self.channels = channels
        excitation = int(ratio * channels)
        self.se = nn.Sequential(
            GlobalAverage(),
            nn.Linear(channels, excitation),
            nn.ReLU(inplace=True),
            nn.Linear(excitation, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        c_se = self.se(x)
        c_se = c_se.reshape(-1, self.channels, 1, 1, 1)
        return c_se * x


class ResBlock(nn.Module):
    def __init__(self, channels, bottleneck_ratio=0.5, se_ratio=0.5, kernel_size=3, padding=1):
        super().__init__()

        bottleneck = int(channels * bottleneck_ratio)

        self.res = nn.Sequential(
            ConvNormReLU(channels, bottleneck, kernel_size=kernel_size, padding=padding),
            ConvNormReLU(bottleneck, channels, kernel_size=kernel_size, padding=padding, activation=False),
            SEBlock(channels, se_ratio)
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        skip = x
        x = self.res(x)
        return self.act(x + skip)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck_ratio=0.5, blocks=1):
        super().__init__()
        self.down = nn.Sequential(
            ConvNormReLU(in_channels, out_channels, stride=2),
            *[ResBlock(out_channels, bottleneck_ratio=bottleneck_ratio) for _ in range(blocks)]
        )
    
    def forward(self, x):
        x = self.down(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, bottleneck_ratio=0.5, blocks=1):
        super().__init__()

        self.up = nn.Sequential(
            # nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3),
            nn.Upsample(scale_factor=2, mode='trilinear'),
            ConvNormReLU(in_channels, out_channels),
        )
        self.combined_res = nn.Sequential(
            ConvNormReLU(out_channels+skip_channels, out_channels),
            *[ResBlock(out_channels, bottleneck_ratio=bottleneck_ratio) for _ in range(blocks)]
        )

    def forward(self, x, skip):
        up = self.up(x)
        diff_z = up.size(-3) - skip.size(-3)
        diff_x = up.size(-2) - skip.size(-2)
        diff_y = up.size(-1) - skip.size(-1)
        if skip.size() != up.size():
            pad = up[:, :, diff_z:, diff_x:, diff_y:]
        else:
            pad = up
        combines = torch.cat([pad, skip], dim=1)
        res = self.combined_res(combines)
        return res


class Unet3D(nn.Module):
    def __init__(self, n_channels, n_classes, bottleneck_ratio=0.5, encoders=(16, 32, 64, 128, 256), blocks=(2, 2, 3, 3)):
        super().__init__()

        decoders = encoders[-2::-1]
        skips = encoders[:-1][::-1]
        self.low_level = ConvNormReLU(in_channels=n_channels, out_channels=encoders[0], kernel_size=7, padding=3)

        self.downs = nn.ModuleList([
            DownBlock(i, o, blocks=b) for (i, o, b) in zip(encoders[:-1], encoders[1:], blocks)
        ])
        self.ups = nn.ModuleList([
            UpBlock(i, o, s) for (i, o, s) in zip(encoders[::-1], decoders, skips)
        ])
        self.head = nn.Sequential(
            nn.Conv3d(decoders[-1], 1 if n_classes == 2 else n_classes, kernel_size=3, padding=1),
            # nn.Sigmoid() if n_classes == 2 else nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.low_level(x)

        downs = [x]
        for d in self.downs:
            x = d(x)
            downs.append(x)

        skips = downs[-2::-1]
        for s, u in zip(skips, self.ups):
            x = u(x, s)
        
        x = self.head(x)
        return x

    def total_params(self):
        total = sum(p.numel() for p in self.parameters())
        return format(total, ',')

    def total_trainable_params(self):
        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return format(total_trainable, ',')
