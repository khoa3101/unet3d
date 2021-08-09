import numpy as np

def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs)**exponent

def sum_tensor(x, axis, keepdim=False):
    axis = np.unique(axis).astype(int)
    if keepdim:
        for ax in axis:
            x = x.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axis, reverse=True):
            x = x.sum(int(ax))
    return x

def mean_tensor(x, axis, keepdim=False):
    axis = np.unique(axis).astype(int)
    if keepdim:
        for ax in axis:
            x = x.mean(int(ax), keepdim=True)
    else:
        for ax in sorted(axis, reverse=True):
            x = x.mean(int(ax))
    return x
    