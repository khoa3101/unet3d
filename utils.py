from scipy.ndimage.filters import gaussian_filter
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
    
def compute_steps_for_sliding_window(patch_size, vol_size, step_size):
    assert [i >= j for i, j in zip(vol_size, patch_size)], 'volume image must be larger or as large as the patch size'
    assert 0 < step_size <= 1, 'step size must be in range (0, 1]'

    # our step width is patch_size*step_size at most, but can be narrower. For example if we have volume size of
    # 110, patch size of 64 and step_size of 0.5, then we want to make 3 steps starting at coordinate 0, 23, 46
    target_step_sizes_in_voxels = [i * step_size for i in patch_size]

    num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(vol_size, target_step_sizes_in_voxels, patch_size)]

    steps = []
    for i in range(len(patch_size)):
        # the highest step value for this dimension is
        max_step_value = vol_size[i] - patch_size[i]
        if num_steps[i] > 1:
            actual_step_size = max_step_value / (num_steps[i] - 1)
        else:
            actual_step_size = 99999999999  # does not matter because there is only one step at 0

        steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[i])]

        steps.append(steps_here)

    return steps

def get_gaussian(patch_size, sigma_scale=1./8):
    tmp = np.zeros(patch_size)
    center_coords = [i // 2 for i in patch_size]
    sigmas = [i * sigma_scale for i in patch_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
    gaussian_importance_map = gaussian_importance_map.astype(np.float32)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map