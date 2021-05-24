import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from rotation import rot, idx

rot = torch.Tensor(rot)
if torch.cuda.is_available():
    rot = rot.cuda()


class ConvZ3P24(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 bias=True, stride=1, padding=0):
        super().__init__()
        w = torch.empty(out_channels, in_channels, kernel_size, kernel_size, kernel_size)
        self.weight = nn.Parameter(w)
        nn.init.kaiming_uniform_(self.weight, a=(5 ** 0.5))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
        self.stride = stride
        self.padding = padding

    def _rotated(self, w):
        ws = []
        for r in rot:
            grid = F.affine_grid(torch.cat([r for _ in range(w.size(0))]), w.size(), align_corners=False)
            _ws = F.grid_sample(w, grid)
            ws.append(_ws)
        return torch.cat(ws, 0)

    def forward(self, x):
        w = self._rotated(self.weight)
        y = F.conv3d(x, w, stride=self.stride, padding=self.padding)
        y = y.view(y.size(0), 24, -1, y.size(2), y.size(3), y.size(4))
        if self.bias is not None:
            y = y + self.bias.view(1, 1, -1, 1, 1, 1)
        return y


# class ConvP24(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size,
#                  bias=True, stride=1, padding=0, transpose=False):
#         super().__init__()
#         if transpose:
#             w = torch.empty(in_channels, out_channels, 24, kernel_size, kernel_size, kernel_size)
#         else:
#             w = torch.empty(out_channels, in_channels, 24, kernel_size, kernel_size, kernel_size)
#         self.weight = nn.Parameter(w)
#         nn.init.kaiming_uniform_(self.weight, a=(5 ** 0.5))
#         if bias:
#             self.bias = nn.Parameter(torch.zeros(out_channels))
#         else:
#             self.bias = None
#         self.stride = stride
#         self.padding = padding
#         self.transpose = transpose

#     def _rotated(self, w):
#         ws = []
#         w = w.view(w.size(0), -1, w.size(3), w.size(4), w.size(5))
#         for r in rot:
#             grid = F.affine_grid(torch.cat([r for _ in range(w.size(0))]), w.size(), align_corners=False)
#             _ws = F.grid_sample(w, grid)
#             print(_ws.size())
#             ws.append(_ws)   

#         return torch.cat(ws, 0)

#     def forward(self, x):
#         x = x.view(x.size(0), -1, x.size(3), x.size(4), x.size(5))
#         w = self._rotated(self.weight)
#         if self.transpose:
#             y = F.conv_transpose3d(x, w, stride=self.stride, padding=self.padding)
#         else:
#             y = F.conv3d(x, w, stride=self.stride, padding=self.padding)
#         y = y.view(y.size(0), 24, -1, y.size(2), y.size(3), y.size(4))
        
#         if self.bias is not None:
#             y = y + self.bias.view(1, 1, -1, 1, 1, 1)
#         return y


class MaxRotationPoolP24(nn.Module):
    def forward(self, x):
        return x.max(1).values


class MaxSpatialPoolP24(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.inner = nn.MaxPool3d(kernel_size, stride, padding)
    
    def forward(self, x):
        y = x.view(x.size(0), -1, x.size(3), x.size(4), x.size(5))
        y = self.inner(y)
        y = y.view(x.size(0), 24, -1, y.size(2), y.size(3), y.size(4))
        return y


class AvgRootPoolP24(nn.Module):
    def forward(self, x):
        return x.mean(1)

if __name__ == '__main__':
    conv = ConvZ3P24(1, 3, 3)
    _conv = nn.Conv3d(1, 3, 3)
    x = torch.rand(1, 1, 5, 5, 5)
    y = conv(x)
    avg = y.mean((1, -3, -2, -1))
    # print(y)
    grid = F.affine_grid(rot[1], x.size(), align_corners=False)
    x_rot = F.grid_sample(x, grid, align_corners=False)
    y_rot = conv(x_rot)
    avg_rot = y_rot.mean((1, -3, -2, -1))
    # print(y_rot)
    y_rot = y_rot[:, idx[1]]
    y_rot = y_rot.view(y_rot.size(0), -1, y_rot.size(3), y_rot.size(4), y_rot.size(5))
    grid_re = F.affine_grid(rot[3], y_rot.size(), align_corners=False)
    y_rot_re = F.grid_sample(y_rot, grid_re, align_corners=False)
    y_rot_re = y_rot_re.view(y_rot.size(0), 24, -1, y_rot.size(2), y_rot.size(3), y_rot.size(4))

    print((avg - avg_rot).abs().max().item())
    print((y_rot_re - y).abs().max().item()) 