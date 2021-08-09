from pathlib import Path
from batchgenerators.dataloading import SlimDataLoaderBase, Dataset
import numpy as np
import torch
import pickle


class Dataset3D(Dataset):
    def __init__(self, path, mode='train'):
        super().__init__()

        self.paths = sorted(Path(path).glob('%s/case_*' % mode))
        self.mode = mode

    def __getitem__(self, index): 
        with open(f'{self.paths[index]}/property.pkl', 'rb') as f:
            property = pickle.load(f)
        data = np.load(f'{self.paths[index]}/data.npz')['data']
        return {'data': data, 'property': property}

    def __len__(self):
        return len(self.paths)


class DataLoader3D(SlimDataLoaderBase):
    def __init__(self, dataset, patch_size, final_patch_size, batch_size, oversample_foreground_percent=0.0, pad_mode='edge', pad_kwargs_data=None):
        self.dataset = dataset
        self.patch_size = patch_size
        self.final_patch_size = final_patch_size
        self.need_to_pad = (np.array(patch_size) - np.array(final_patch_size)).astype(int)
        self.batch_size = batch_size
        self.oversample_foreground_percent = oversample_foreground_percent
        self.pad_mode = pad_mode
        if not pad_kwargs_data:
            pad_kwargs_data = {}
        self.pad_kwargs_data = pad_kwargs_data
        self.data_shape, self.seg_shape = self.get_shapes()

    def get_shapes(self):
        case_shape = self.dataset[0]['data'].shape
        data_shape = (self.batch_size, case_shape[0] - 1, *self.patch_size)
        seg_shape = (self.batch_size, 1, *self.patch_size)
        return data_shape, seg_shape

    def do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))

    def generate_train_batch(self):
        selected_cases = np.random.choice(len(self.dataset), self.batch_size)
        data = np.zeros(self.data_shape, dtype=np.float32)
        seg = np.zeros(self.seg_shape, dtype=np.float32)
        properties = []
        for batch_idx, i in enumerate(selected_cases):
            case = self.dataset[i]
            properties.append(case['property'])

            # Get bbox location to crop
            shape = case['data'].shape[1:]
            need_to_pad = self.need_to_pad
            for d in range(3):
                if need_to_pad[d] + shape[d] < self.patch_size[d]:
                    need_to_pad[d] = self.patch_size[d] - shape[d]
            lb_z = - need_to_pad[0]//2
            ub_z = shape[0] + need_to_pad[0]//2 + need_to_pad[0]%2 - self.patch_size[0]
            lb_x = - need_to_pad[1]//2
            ub_x = shape[1] + need_to_pad[1]//2 + need_to_pad[1]%2 - self.patch_size[1]
            lb_y = - need_to_pad[2]//2
            ub_y = shape[2] + need_to_pad[2]//2 + need_to_pad[2]%2 - self.patch_size[2]

            # In trivial batch, just random literally. Otherwise, crop around a random voxel in foreground classes
            force_fg = True if self.do_oversample(batch_idx) else False
            if not force_fg:
                bbox_lb_z = np.random.randint(lb_z, ub_z + 1)
                bbox_lb_x = np.random.randint(lb_x, ub_x + 1)
                bbox_lb_y = np.random.randint(lb_y, ub_y + 1)
            else:
                # Following lines to ensure the selected class exists in the case
                fg_classes = np.array([
                    idx for idx in case['property']['class_locations'].keys() if len(case['property']['class_locations'][idx]) > 0
                ])
                fg_classes = fg_classes[fg_classes > 0]

                if len(fg_classes) > 0:
                    selected_class = np.random.choice(fg_classes)
                    voxel_locations = case['property']['class_locations'][selected_class]
                else:
                    voxel_locations = None

                if voxel_locations is not None:
                    selected_voxel = voxel_locations[np.random.choice(len(voxel_locations))]
                    # Selected voxel is in the center of the crop dataume
                    bbox_lb_z = max(lb_z, selected_voxel[0] - self.patch_size[0]//2)
                    bbox_lb_x = max(lb_x, selected_voxel[1] - self.patch_size[1]//2)
                    bbox_lb_y = max(lb_y, selected_voxel[2] - self.patch_size[2]//2)
                else:
                    bbox_lb_z = np.random.randint(lb_z, ub_z + 1)
                    bbox_lb_x = np.random.randint(lb_x, ub_x + 1)
                    bbox_lb_y = np.random.randint(lb_y, ub_y + 1)

            bbox_ub_z = bbox_lb_z + self.patch_size[0]
            bbox_ub_x = bbox_lb_x + self.patch_size[1]
            bbox_ub_y = bbox_lb_y + self.patch_size[2]

            valid_bbox_lb_z = max(0, bbox_lb_z)
            valid_bbox_ub_z = min(shape[0], bbox_ub_z)
            valid_bbox_lb_x = max(0, bbox_lb_x)
            valid_bbox_ub_x = min(shape[1], bbox_ub_x)
            valid_bbox_lb_y = max(0, bbox_lb_y)
            valid_bbox_ub_y = min(shape[2], bbox_ub_y)

            case_data = np.copy(case['data'][
                :,
                valid_bbox_lb_z:valid_bbox_ub_z,
                valid_bbox_lb_x:valid_bbox_ub_x,
                valid_bbox_lb_y:valid_bbox_ub_y
            ])

            data[batch_idx] = np.pad(
                case_data[:-1], 
                (
                    (0,0),
                    (-min(0, bbox_lb_z), max(bbox_ub_z - shape[0], 0)),
                    (-min(0, bbox_lb_x), max(bbox_ub_x - shape[1], 0)),
                    (-min(0, bbox_lb_y), max(bbox_ub_y - shape[2], 0)),
                ),
                self.pad_mode,
                **self.pad_kwargs_data
            )
            seg[batch_idx, 0] = np.pad(
                case_data[-1:],
                (
                    (0,0),
                    (-min(0, bbox_lb_z), max(bbox_ub_z - shape[0], 0)),
                    (-min(0, bbox_lb_x), max(bbox_ub_x - shape[1], 0)),
                    (-min(0, bbox_lb_y), max(bbox_ub_y - shape[2], 0)),
                ),
                'constant',
                **{'constant_values': 0}
            )
        
        return {'data': data, 'seg': seg, 'properties': properties}