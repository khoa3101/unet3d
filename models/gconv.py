import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

idx = np.load('models/idx.npy')
rot = np.load('models/rot.npy')
rot = torch.from_numpy(rot)
n_rot = len(rot)
if torch.cuda.is_available():
    rot = rot.cuda()

class GConvZ3S4(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 bias=True, stride=1, padding=0):
        super().__init__()
        w = torch.empty(out_channels, in_channels, kernel_size, kernel_size, kernel_size)
        self.weight = nn.Parameter(w)
        nn.init.kaiming_uniform_(self.weight, a=(5 ** 0.5))
        self.grids = self._compute_grid()
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
        self.stride = stride
        self.padding = padding

    def _compute_grid(self):
        w_shape = self.weight.shape

        # Cout x G x 3 x 4
        r = torch.stack([rot] * w_shape[0])
        # GCout x 3 x 4
        r = r.transpose(0, 1).flatten(end_dim=1)
        
        grids = F.affine_grid(r, (n_rot * w_shape[0], *w_shape[1:]), align_corners=False)
        # grids = [F.affine_grid(torch.cat([r for _ in range(self.weight.size(0))]), self.weight.size(), align_corners=False) for r in rot]
        return grids

    def _rotated(self):
        # GCout x Cin x K^3
        ws = torch.cat([self.weight] * n_rot)
        
        ws = F.grid_sample(ws, self.grids, align_corners=False)
        # ws = [F.grid_sample(self.weight, self.grids[i]) for i in range(n_rot)]
        # for r in rot:
        #     grid = F.affine_grid(torch.cat([r for _ in range(w.size(0))]), w.size(), align_corners=False)
        #     _ws = F.grid_sample(w, grid)
        #     ws.append(_ws)
        return ws #torch.cat(ws, 0)

    def forward(self, x):
        w = self._rotated()
        y = F.conv3d(x, w, stride=self.stride, padding=self.padding)
        y = y.reshape(y.size(0), n_rot, -1, y.size(2), y.size(3), y.size(4)) 
        if self.bias is not None:
            y = y + self.bias.reshape(1, 1, -1, 1, 1, 1)
        return y


class GConvS4(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 bias=True, stride=1, padding=0, transpose=False):
        super().__init__()
        if transpose:
            w = torch.empty(in_channels, n_rot, out_channels, kernel_size, kernel_size, kernel_size)
        else:
            w = torch.empty(out_channels, n_rot, in_channels, kernel_size, kernel_size, kernel_size)
        self.weight = nn.Parameter(w)
        nn.init.kaiming_uniform_(self.weight, a=(5 ** 0.5))
        self.grids = self._compute_grid()
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
        self.stride = stride
        self.padding = padding
        self.transpose = transpose

    def _compute_grid(self):
        w_shape = self.weight.shape

        # Cout x G x 3 x 4
        r = torch.stack([rot] * w_shape[0])
        # GCout x 3 x 4
        r = r.transpose(0, 1).flatten(end_dim=1)

        grids = F.affine_grid(r, (n_rot * w_shape[0], w_shape[1] * w_shape[2], *w_shape[3:]), align_corners=False)
        # grids = []
        # for i, r in enumerate(rot):
        #     _ws = self.weight[:, idx[i]] #torch.clone(self.weight[:, idx[i]])
        #     _ws = _ws.reshape(self.weight.size(0), -1, self.weight.size(3), self.weight.size(4), self.weight.size(5))
        #     grid = F.affine_grid(torch.cat([r for _ in range(_ws.size(0))]), _ws.size(), align_corners=False)
        #     grids.append(grid)
        return grids

    def _rotated(self):
        # GCout x G x Cin x K^3
        ws = torch.cat([self.weight[:, i] for i in idx])
        # GCout x GCin x K^3
        ws = ws.flatten(start_dim=1, end_dim=2)

        ws = F.grid_sample(ws, self.grids, align_corners=False)
        # ws = []
        # for i, r in enumerate(rot):
        #     _ws = self.weight[:, idx[i]] #torch.clone(w[:, idx[i]])
        #     _ws = _ws.reshape(self.weight.size(0), -1, self.weight.size(3), self.weight.size(4), self.weight.size(5))
        #     # grid = F.affine_grid(torch.cat([r for _ in range(_ws.size(0))]), _ws.size(), align_corners=False)
        #     _ws = F.grid_sample(_ws, self.grids[i])
        #     ws.append(_ws)   
        return ws #torch.cat(ws, 0)

    def forward(self, x):
        x = x.reshape(x.size(0), -1, x.size(3), x.size(4), x.size(5))
        w = self._rotated()
        if self.transpose:
            y = F.conv_transpose3d(x, w, stride=self.stride, padding=self.padding)
        else:
            y = F.conv3d(x, w, stride=self.stride, padding=self.padding)
        y = y.reshape(y.size(0), n_rot, -1, y.size(2), y.size(3), y.size(4))
        
        if self.bias is not None:
            y = y + self.bias.reshape(1, 1, -1, 1, 1, 1)
        return y


class GMaxRotationPoolS4(nn.Module):
    def forward(self, x):
        return x.max(1).values


class GMaxSpatialPoolS4(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.inner = nn.MaxPool3d(kernel_size, stride, padding)
    
    def forward(self, x):
        y = x.reshape(x.size(0), -1, x.size(3), x.size(4), x.size(5))
        y = self.inner(y)
        y = y.reshape(x.size(0), n_rot, -1, y.size(2), y.size(3), y.size(4))
        return y


class GAvgRootPoolS4(nn.Module):
    def forward(self, x):
        return x.mean(1)


class GNorm(nn.InstanceNorm3d):
    def forward(self, x):
        x = x.transpose(1, 2)
        x = x.reshape(x.size(0), -1, x.size(3), x.size(4), x.size(5))
        x = super().forward(x).reshape(x.size(0), -1, n_rot, x.size(2), x.size(3), x.size(4))
        return x.transpose(1, 2)


if __name__ == '__main__':
    conv = nn.Sequential(
        GConvZ3S4(1, 3, 3),
        GConvS4(3, 5, 3),
        GConvS4(5, 10, 3),
        GConvS4(10, 15, 3)
    )
    _conv = nn.Conv3d(1, 3, 3)
    x = torch.rand(1, 1, 32, 64, 128)
    y = conv(x)
    avg = y.mean((1, -3, -2, -1))
    # print(y)
    grid = F.affine_grid(rot[1], torch.Size((x.size(0), x.size(1), x.size(3), x.size(2), x.size(4))), align_corners=False)
    x_rot = F.grid_sample(x, grid, align_corners=False)
    y_rot = conv(x_rot)
    avg_rot = y_rot.mean((1, -3, -2, -1))
    y_rot = y_rot.reshape(y_rot.size(0), -1, y_rot.size(3), y_rot.size(4), y_rot.size(5))
    grid_re = F.affine_grid(rot[3], torch.Size((y_rot.size(0), y_rot.size(1), y_rot.size(3), y_rot.size(2), y_rot.size(4))), align_corners=False)
    y_rot_re = F.grid_sample(y_rot, grid_re, align_corners=False)
    y_rot_re = y_rot_re.reshape(y_rot.size(0), n_rot, -1, y_rot_re.size(2), y_rot_re.size(3), y_rot_re.size(4))
    y_rot_re = y_rot_re[:, idx[3]]
    print((avg - avg_rot).abs().max().item())
    print((y_rot_re - y).abs().max().item())