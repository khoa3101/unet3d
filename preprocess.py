from collections import OrderedDict
from batchgenerators.augmentations.utils import resize_segmentation
from scipy.ndimage.interpolation import map_coordinates
from skimage.transform import resize
from multiprocessing.pool import Pool
from pathlib import Path
from tqdm import tqdm
import SimpleITK as sitk
import numpy as np
import argparse
import pickle

def args_parser():
    parser = argparse.ArgumentParser(description='Preprocess data')
    parser.add_argument('-p', '--path', required=True, type=str, help='Path to data folder')
    parser.add_argument('-t', '--threshold', default=3.0, type=float, help='Threshold to separate z axis')
    parser.add_argument('-n', '--n_classes', default=3, type=int, help='Number of classes in label')
    args = parser.parse_args()
    return args

def check_separate_z(spacing, threshold=3.0):
    separate_z = (np.max(spacing) / np.min(spacing)) > threshold
    return separate_z

def get_lowres_axis(spacing):
    axis = np.where((np.max(spacing) / spacing) == 1)[0]
    return axis

def resample_instance(instance, is_seg, new_shape, axis=None, order=3, do_separate_z=False, cval=0, order_z=0):
    if is_seg:
        resize_function = resize_segmentation
        kwargs = OrderedDict()
    else:
        resize_function = resize
        kwargs = {'mode': 'edge', 'anti_aliasing': False}
        
    dtype_instance = instance.dtype
    instance = instance.astype(float)
    shape = np.array(instance[0].shape)
    new_shape = np.array(new_shape)

    if np.any(shape != new_shape):
        if do_separate_z:
            axis = axis[0]
            if axis == 0:
                new_shape_2d = new_shape[1:]
            elif axis == 1:
                new_shape_2d = new_shape[[0, 2]]
            else:
                new_shape_2d = new_shape[:-1]

            reshaped_final_instance = []
            for c in range(instance.shape[0]):
                reshaped_instance = []
                for slice_id in range(shape[axis]):
                    if axis == 0:
                        reshaped_instance.append(resize_function(instance[c, slice_id], new_shape_2d, order, cval=cval, **kwargs))
                    elif axis == 1:
                        reshaped_instance.append(resize_function(instance[c, :, slice_id], new_shape_2d, order, cval=cval, **kwargs))
                    else:
                        reshaped_instance.append(resize_function(instance[c, :, :, slice_id], new_shape_2d, order, cval=cval, **kwargs))
                reshaped_instance = np.stack(reshaped_instance, axis)
                if shape[axis] != new_shape[axis]:

                    # The following few lines are blatantly copied and modified from sklearn's resize()
                    rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
                    orig_rows, orig_cols, orig_dim = reshaped_instance.shape

                    row_scale = float(orig_rows) / rows
                    col_scale = float(orig_cols) / cols
                    dim_scale = float(orig_dim) / dim

                    map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
                    map_rows = row_scale * (map_rows + 0.5) - 0.5
                    map_cols = col_scale * (map_cols + 0.5) - 0.5
                    map_dims = dim_scale * (map_dims + 0.5) - 0.5

                    coord_map = np.array([map_rows, map_cols, map_dims])
                    if not is_seg or order_z == 0:
                        reshaped_final_instance.append(map_coordinates(reshaped_instance, coord_map, order=order_z, cval=cval, mode='nearest'))
                    else:
                        unique_segs = np.unique(reshaped_instance)
                        reshaped = np.zeros(new_shape, dtype=dtype_instance)

                        for cl in unique_segs:
                            reshaped_multihot = np.round(
                                map_coordinates((reshaped_instance == cl).astype(float), 
                                coord_map, order=order_z, cval=cval, mode='nearest')
                            )
                            reshaped[reshaped_multihot > 0.5] = cl
                        reshaped_final_instance.append(reshaped)
                else:
                    reshaped_final_instance.append(reshaped_instance)
            reshaped_final_instance = np.stack(reshaped_final_instance)
        else:
            reshaped = []
            for c in range(instance.shape[0]):
                reshaped.append(resize_function(instance[c], new_shape, order, cval=cval, **kwargs))
            reshaped_final_instance = np.stack(reshaped)
        return reshaped_final_instance.astype(dtype_instance)

    return instance

def resample(data, seg, og_spacing, target_spacing, order_data=3, order_seg=0, 
             cval_data=0, cval_seg=-1, order_z_data=0, order_z_seg=0, threshold=3.0):
    shape = np.array(data[0].shape)
    new_shape = (shape * np.array(og_spacing) / np.array(target_spacing)).astype(int)

    if check_separate_z(og_spacing, threshold=threshold):
        do_separate_z = True
        lowres_axis = get_lowres_axis(og_spacing)
    elif check_separate_z(target_spacing, threshold=threshold):
        do_separate_z = True
        lowres_axis = get_lowres_axis(target_spacing)
    else:
        do_separate_z = False
        lowres_axis = None

    if lowres_axis is not None and len(lowres_axis) != 1:
        # case (0.24, 1.25, 1.25)
        do_separate_z = False

    if data is not None:
        data_reshaped = resample_instance(
            data, False, new_shape, axis=lowres_axis, 
            order=order_data, do_separate_z=do_separate_z, cval=cval_data, order_z=order_z_data
        )
    else:
        data_reshaped = None

    if seg is not None:
        seg_reshaped = resample_instance(
            seg, True, new_shape, axis=lowres_axis,
            order=order_seg, do_separate_z=do_separate_z, cval=cval_seg, order_z=order_z_seg
        )
    else:
        seg_reshaped = None

    return data_reshaped, seg_reshaped

def pre(inp):
    path, (stats, n_classes, threshold) = inp
    data = sitk.ReadImage(str(path / 'imaging.nii.gz'))
    seg = sitk.ReadImage(str(path / 'segmentation.nii.gz')) if (path / 'segmentation.nii.gz').exists() else None
    preprocessed_folder = path.parents[1] / 'preprocessed'
    preprocessed_folder.mkdir(exist_ok=True)
    save_path = preprocessed_folder / path.name
    save_path.mkdir(exist_ok=True)

    data_reshaped, seg_reshaped = resample(
        sitk.GetArrayFromImage(data).transpose()[None], 
        sitk.GetArrayFromImage(seg).transpose()[None] if seg else None, 
        data.GetSpacing(), stats['target_spacing'], 
        order_data=3, order_seg=1, threshold=threshold
    )
    clipped = np.clip(data_reshaped, stats['lower_bound'], stats['upper_bound'])
    data_reshaped = (clipped - stats['mean']) / stats['std']

    # We need to find out where the foreground is and sample some random location
    # let's do 10.000 samples per class
    num_samples = 10000
    min_percent_coverage = 0.01
    rndst = np.random.RandomState(3101)
    class_locs = {}
    for c in range(1, n_classes):
        all_locs = np.argwhere(seg_reshaped[0] == c) if seg_reshaped is not None else []
        if len(all_locs) == 0:
            class_locs[c] = []
            continue
        
        target_num_samples = min(num_samples, len(all_locs))
        target_num_samples = max(target_num_samples, int(np.ceil(min_percent_coverage * len(all_locs))))

        selected = all_locs[rndst.choice(len(all_locs), target_num_samples, replace=False)]
        class_locs[c] = selected
    
    result = {
        'ID': path.name,
        'size': data.GetSize(),
        'new_size': data_reshaped.shape[1:],
        'spacing': data.GetSpacing(),
        'origin': data.GetOrigin(),
        'direction': data.GetDirection(),
        'class_locations': class_locs
    }
    with open(save_path / ('property.pkl'), 'wb') as f:
        pickle.dump(result, f)

    if not seg:
        seg_reshaped = np.zeros_like(data_reshaped, dtype=np.uint8)

    data = np.vstack((data_reshaped, seg_reshaped))
    np.savez_compressed(save_path / 'data.npz', data=data.astype(np.float32))
    
    return result

def get_voxels_foreground(path):
    data = sitk.ReadImage(str(path / 'imaging.nii.gz'))
    spacing = data.GetSpacing()
    size = data.GetSize()
    data = sitk.GetArrayFromImage(data)
    seg = sitk.GetArrayFromImage(sitk.ReadImage(str(path / 'segmentation.nii.gz'))) if (path / 'segmentation.nii.gz').exists() else None
    if seg is None:
        return None, spacing, size

    mask = data[seg > 0]
    voxels= list(mask[::10])
    return voxels, spacing, size

def compute_stats(voxels, spacings, sizes, args):
    mean = np.mean(voxels)
    std = np.std(voxels)
    lower_bound = np.percentile(voxels, 0.5)
    upper_bound = np.percentile(voxels, 99.5)
    
    # per default we use the 50th percentile=median for the target spacing. Higher spacing results in smaller data
    # and thus faster and easier training. Smaller spacing results in larger data and thus longer and harder training
    #     
    # For some datasets the median is not a good choice. Those are the datasets where the spacing is very anisotropic
    # (for example ACDC with (10, 1.5, 1.5)). These datasets still have examples with a spacing of 5 or 6 mm in the low
    # resolution axis. Choosing the median here will result in bad interpolation artifacts that can substantially
    # impact performance (due to the low number of slices).
    
    target_spacing = np.percentile(np.vstack(spacings), 50, 0)
    target_size = np.percentile(np.vstack(sizes), 50, 0)
    # we need to identify datasets for which a different target spacing could be beneficial. These datasets have
    # the following properties:
    # - one axis which much lower resolution than the others
    # - the lowres axis has much less voxels than the others
    worst_spacing_axis = np.argmax(target_spacing)
    other_axis = [i for i in range(len(target_spacing)) if i != worst_spacing_axis]
    other_spacing = [target_spacing[i] for i in other_axis]
    other_size = [target_size[i] for i in other_axis]
    
    has_aniso_spacing = target_spacing[worst_spacing_axis] > (args.threshold * max(other_spacing))
    has_aniso_voxels = (target_size[worst_spacing_axis] * args.threshold) < min(other_size)
    
    if has_aniso_spacing and has_aniso_voxels:
        spacing_of_that_axis = np.vstack(spacing)[:, worst_spacing_axis]
        target_spacing_of_that_axis = np.percentile(spacing_of_that_axis, 10)
        # don't let the spacing of that axis get higher than the other axes
        if target_spacing_of_that_axis < max(other_spacing):
            target_spacing_of_that_axis = max(max(other_spacing), target_spacing_of_that_axis) + 1e-5
        target_spacing[worst_spacing_axis] = target_spacing_of_that_axis
        
    return mean, std, lower_bound, upper_bound, target_spacing


if __name__ == '__main__':
    args = args_parser()

    pool = Pool(8)

    # Compute stats
    print('Computing stats')
    paths = sorted(Path('%s/data' % args.path).glob('case_*'))
    voxels = []
    spacings = []
    sizes = []
    for case, spacing, size in tqdm(pool.imap_unordered(get_voxels_foreground, paths), total=len(paths)):
        if case:
            voxels += case
        spacings.append(spacing)
        sizes.append(size)

    mean, std, lower_bound, upper_bound, target_spacing = compute_stats(voxels, spacings, sizes, args)
    stats = {
        'mean': mean,
        'std': std,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'target_spacing': target_spacing
    }    
    with open('%s/stats.pkl' % args.path, 'wb') as f:
        pickle.dump(stats, f)
    print('\nPreprocessing')

    # Preprocess data
    properties = []
    for _ in tqdm(pool.imap_unordered(pre, zip(paths, [(stats, args.n_classes, args.threshold)]*len(paths))), total=len(paths)):
        pass

    pool.close()
    pool.join()