from torch.utils.data import Dataset
from pathlib import Path
import torchio as tio
import numpy as np
import torch

class KiTS2019(Dataset):
    def __init__(self, path, mode='train', n_channel=1):
        super(KiTS2019, self).__init__()

        self.paths = sorted(Path(path).glob('preprocessed_3d/%s/case_*' % mode))
        self.mode = mode

        train_transform = tio.Compose([
            tio.RandomMotion(p=0.2),
            tio.RandomBiasField(p=0.3),
            tio.RandomNoise(p=0.5),
            tio.RandomFlip(axes=(2,)),
            tio.OneOf({
                tio.RandomAffine(scales=0.002): 0.8,
                tio.RandomElasticDeformation(): 0.2,
            }, p=0.3),
            tio.RescaleIntensity(out_min_max=(-1, 1)),
            tio.OneHot(3),
        ])
        val_transform = tio.Compose([
            tio.RescaleIntensity(out_min_max=(-1, 1)),
            tio.OneHot(3),
        ])
        test_transform = tio.Compose([
            tio.RescaleIntensity(out_min_max=(-1, 1)),
        ])

        self.transform = {
            'train': train_transform,
            'val': val_transform,
            'test': test_transform
        }

    def __getitem__(self, index):
        vol = tio.ScalarImage(self.paths[index] / 'imaging.nii.gz')
        vol = vol[tio.DATA]
        vol = np.clip(vol, -80, 300)
        vol = self.transform[self.mode](vol)
        label = tio.LabelMap(self.paths[index] / 'segmentation.nii.gz')
        label = label[tio.DATA]
        label = np.clip(label, -80, 300)
        label = self.transform[self.mode](label)
        return vol, label

    def __len__(self):
        return len(self.paths)