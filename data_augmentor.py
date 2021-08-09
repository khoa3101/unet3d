import numpy as np
from batchgenerators.dataloading import MultiThreadedAugmenter
from batchgenerators.transforms import DataChannelSelectionTransform, SegChannelSelectionTransform, SpatialTransform, \
    GammaTransform, MirrorTransform, Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, BrightnessTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor
from batchgenerators.transforms import AbstractTransform
from batchgenerators.augmentations.utils import resize_segmentation, rotate_coords_3d, rotate_coords_2d

try:
    from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
except ImportError as ie:
    NonDetMultiThreadedAugmenter = None


default_3D_augmentation_params = {
    'selected_data_channels': None,
    'selected_seg_channels': None,

    'do_elastic': True,
    'elastic_deform_alpha': (0., 900.),
    'elastic_deform_sigma': (9., 13.),
    'p_eldef': 0.2,

    'do_scaling': True,
    'scale_range': (0.85, 1.25),
    'independent_scale_factor_for_each_axis': False,
    'p_independent_scale_per_axis': 1,
    'p_scale': 0.2,

    'do_rotation': True,
    'rotation_x': (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
    'rotation_y': (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
    'rotation_z': (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
    'rotation_p_per_axis': 1,
    'p_rot': 0.2,

    'random_crop': False,
    'random_crop_dist_to_border': None,

    'do_gamma': True,
    'gamma_retain_stats': True,
    'gamma_range': (0.7, 1.5),
    'p_gamma': 0.3,

    'do_mirror': True,
    'mirror_axes': (0, 1, 2),

    'border_mode_data': 'constant',

    'do_additive_brightness': False,
    'additive_brightness_p_per_sample': 0.15,
    'additive_brightness_p_per_channel': 0.5,
    'additive_brightness_mu': 0.0,
    'additive_brightness_sigma': 0.1,

    'num_threads': 12,
    'num_cached_per_thread': 1,
}


def get_patch_size(final_patch_size, rot_x, rot_y, rot_z, scale_range):
    if isinstance(rot_x, (tuple, list)):
        rot_x = max(np.abs(rot_x))
    if isinstance(rot_y, (tuple, list)):
        rot_y = max(np.abs(rot_y))
    if isinstance(rot_z, (tuple, list)):
        rot_z = max(np.abs(rot_z))
    rot_x = min(90 / 360 * 2. * np.pi, rot_x)
    rot_y = min(90 / 360 * 2. * np.pi, rot_y)
    rot_z = min(90 / 360 * 2. * np.pi, rot_z)
    coords = np.array(final_patch_size)
    final_shape = np.copy(coords)
    if len(coords) == 3:
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, rot_x, 0, 0)), final_shape)), 0)
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, rot_y, 0)), final_shape)), 0)
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, 0, rot_z)), final_shape)), 0)
    elif len(coords) == 2:
        final_shape = np.max(np.vstack((np.abs(rotate_coords_2d(coords, rot_x)), final_shape)), 0)
    final_shape /= min(scale_range)
    return final_shape.astype(int)

def get_default_augmentation(dataloader_train, dataloader_val, patch_size, params=default_3D_augmentation_params,
                             pin_memory=True, seeds_train=None, seeds_val=None):
    assert params.get('mirror') is None, 'old version of params, use new keyword do_mirror'
    tr_transforms = []

    if params.get('selected_data_channels') is not None:
        tr_transforms.append(DataChannelSelectionTransform(params.get('selected_data_channels')))
    if params.get('selected_seg_channels') is not None:
        tr_transforms.append(SegChannelSelectionTransform(params.get('selected_seg_channels')))

    tr_transforms.append(SpatialTransform(
        patch_size, patch_center_dist_from_border=None, do_elastic_deform=params.get('do_elastic'),
        alpha=params.get('elastic_deform_alpha'), sigma=params.get('elastic_deform_sigma'),
        do_rotation=params.get('do_rotation'), angle_x=params.get('rotation_x'), angle_y=params.get('rotation_y'),
        angle_z=params.get('rotation_z'), do_scale=params.get('do_scaling'), scale=params.get('scale_range'),
        border_mode_data=params.get('border_mode_data'), border_cval_data=0, order_data=3, border_mode_seg='constant',
        border_cval_seg=0, order_seg=1, random_crop=params.get('random_crop'), p_el_per_sample=params.get('p_eldef'),
        p_scale_per_sample=params.get('p_scale'), p_rot_per_sample=params.get('p_rot'),
        independent_scale_for_each_axis=params.get('independent_scale_factor_for_each_axis')
    ))

    if params.get('do_gamma'):
        tr_transforms.append(
            GammaTransform(params.get('gamma_range'), False, True, retain_stats=params.get('gamma_retain_stats'),
                           p_per_sample=params['p_gamma']))

    if params.get('do_mirror'):
        tr_transforms.append(MirrorTransform(params.get('mirror_axes')))

    tr_transforms.append(NumpyToTensor(['data', 'seg'], 'float'))

    tr_transforms = Compose(tr_transforms)

    batchgenerator_train = MultiThreadedAugmenter(dataloader_train, tr_transforms, params.get('num_threads'),
                                                  params.get('num_cached_per_thread'), seeds=seeds_train,
                                                  pin_memory=pin_memory)

    val_transforms = []
    if params.get('selected_vol_channels') is not None:
        val_transforms.append(DataChannelSelectionTransform(params.get('selected_data_channels')))
    if params.get('selected_label_channels') is not None:
        val_transforms.append(SegChannelSelectionTransform(params.get('selected_seg_channels')))

    val_transforms.append(NumpyToTensor(['data', 'seg'], 'float'))
    val_transforms = Compose(val_transforms)

    batchgenerator_val = MultiThreadedAugmenter(dataloader_val, val_transforms, max(params.get('num_threads') // 2, 1),
                                                params.get('num_cached_per_thread'), seeds=seeds_val,
                                                pin_memory=pin_memory)
    return batchgenerator_train, batchgenerator_val

def get_moreDA_augmentation(dataloader_train, dataloader_val, patch_size, 
                            params=default_3D_augmentation_params,
                            border_val_seg=-1,
                            seeds_train=None, seeds_val=None, order_seg=1, order_data=3, deep_supervision_scales=None,
                            pin_memory=True, use_nondetMultiThreadedAugmenter: bool = False):
    assert params.get('mirror') is None, 'old version of params, use new keyword do_mirror'

    tr_transforms = []

    if params.get('selected_data_channels') is not None:
        tr_transforms.append(DataChannelSelectionTransform(params.get('selected_data_channels')))
    if params.get('selected_seg_channels') is not None:
        tr_transforms.append(SegChannelSelectionTransform(params.get('selected_seg_channels')))

    tr_transforms.append(SpatialTransform(
        patch_size, patch_center_dist_from_border=None,
        do_elastic_deform=params.get('do_elastic'), alpha=params.get('elastic_deform_alpha'),
        sigma=params.get('elastic_deform_sigma'),
        do_rotation=params.get('do_rotation'), angle_x=params.get('rotation_x'), angle_y=params.get('rotation_y'),
        angle_z=params.get('rotation_z'), p_rot_per_axis=params.get('rotation_p_per_axis'),
        do_scale=params.get('do_scaling'), scale=params.get('scale_range'),
        border_mode_data=params.get('border_mode_data'), border_cval_data=0, order_data=order_data,
        border_mode_seg='constant', border_cval_seg=border_val_seg,
        order_seg=order_seg, random_crop=params.get('random_crop'), p_el_per_sample=params.get('p_eldef'),
        p_scale_per_sample=params.get('p_scale'), p_rot_per_sample=params.get('p_rot'),
        independent_scale_for_each_axis=params.get('independent_scale_factor_for_each_axis')
    ))

    # we need to put the color augmentations after the dummy 2d part (if applicable). Otherwise the overloaded color
    # channel gets in the way
    tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
    tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                                               p_per_channel=0.5))
    tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))

    if params.get('do_additive_brightness'):
        tr_transforms.append(BrightnessTransform(params.get('additive_brightness_mu'),
                                                 params.get('additive_brightness_sigma'),
                                                 True, p_per_sample=params.get('additive_brightness_p_per_sample'),
                                                 p_per_channel=params.get('additive_brightness_p_per_channel')))

    tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
    tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                        p_per_channel=0.5,
                                                        order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                                        ignore_axes=None))
    tr_transforms.append(
        GammaTransform(params.get('gamma_range'), True, True, retain_stats=params.get('gamma_retain_stats'),
                       p_per_sample=0.1))  # inverted gamma

    if params.get('do_gamma'):
        tr_transforms.append(
            GammaTransform(params.get('gamma_range'), False, True, retain_stats=params.get('gamma_retain_stats'),
                           p_per_sample=params['p_gamma']))

    if params.get('do_mirror') or params.get('mirror'):
        tr_transforms.append(MirrorTransform(params.get('mirror_axes')))

    if deep_supervision_scales is not None:
        tr_transforms.append(DownsampleSegForDSTransform(deep_supervision_scales, 0, 0, input_key='seg',
                                                              output_key='seg'))

    tr_transforms.append(NumpyToTensor(['data', 'seg'], 'float'))
    tr_transforms = Compose(tr_transforms)

    if use_nondetMultiThreadedAugmenter:
        if NonDetMultiThreadedAugmenter is None:
            raise RuntimeError('NonDetMultiThreadedAugmenter is not yet available')
        batchgenerator_train = NonDetMultiThreadedAugmenter(dataloader_train, tr_transforms, params.get('num_threads'),
                                                            params.get('num_cached_per_thread'), seeds=seeds_train,
                                                            pin_memory=pin_memory)
    else:
        batchgenerator_train = MultiThreadedAugmenter(dataloader_train, tr_transforms, params.get('num_threads'),
                                                      params.get('num_cached_per_thread'),
                                                      seeds=seeds_train, pin_memory=pin_memory)

    val_transforms = []
    val_transforms.append(RemoveLabelTransform(-1, 0))
    if params.get('selected_data_channels') is not None:
        val_transforms.append(DataChannelSelectionTransform(params.get('selected_data_channels')))
    if params.get('selected_seg_channels') is not None:
        val_transforms.append(SegChannelSelectionTransform(params.get('selected_seg_channels')))

    if deep_supervision_scales is not None:
        val_transforms.append(DownsampleSegForDSTransform(deep_supervision_scales, 0, 0, input_key='seg',
                                                               output_key='seg'))

    val_transforms.append(NumpyToTensor(['data', 'seg'], 'float'))
    val_transforms = Compose(val_transforms)

    if use_nondetMultiThreadedAugmenter:
        if NonDetMultiThreadedAugmenter is None:
            raise RuntimeError('NonDetMultiThreadedAugmenter is not yet available')
        batchgenerator_val = NonDetMultiThreadedAugmenter(dataloader_val, val_transforms,
                                                          max(params.get('num_threads') // 2, 1),
                                                          params.get('num_cached_per_thread'),
                                                          seeds=seeds_val, pin_memory=pin_memory)
    else:
        batchgenerator_val = MultiThreadedAugmenter(dataloader_val, val_transforms,
                                                    max(params.get('num_threads') // 2, 1),
                                                    params.get('num_cached_per_thread'),
                                                    seeds=seeds_val, pin_memory=pin_memory)

    return batchgenerator_train, batchgenerator_val


class DownsampleSegForDSTransform(AbstractTransform):
    '''
    data_dict['output_key'] will be a list of segmentations scaled according to ds_scales
    '''
    def __init__(self, ds_scales=(1, 0.5, 0.25), order=0, cval=0, input_key='seg', output_key='seg', axes=None):
        self.axes = axes
        self.output_key = output_key
        self.input_key = input_key
        self.cval = cval
        self.order = order
        self.ds_scales = ds_scales

    def __call__(self, **data_dict):
        seg = data_dict[self.input_key]

        if self.axes is None:
            self.axes = list(range(2, len(seg.shape)))
        output = []
        for s in self.ds_scales:
            if all([i == 1 for i in s]):
                output.append(seg)
            else:
                new_shape = np.array(seg.shape).astype(float)
                for i, a in enumerate(self.axes):
                    new_shape[a] *= s[i]
                new_shape = np.round(new_shape).astype(int)
                out_seg = np.zeros(new_shape, dtype=seg.dtype)
                for b in range(seg.shape[0]):
                    for c in range(seg.shape[1]):
                        out_seg[b, c] = resize_segmentation(seg[b, c], new_shape[2:], self.order, self.cval)
                output.append(out_seg)

        data_dict[self.output_key] = output
        return data_dict