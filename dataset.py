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
            tio.ZNormalization(masking_method=tio.ZNormalization.mean),
            # tio.RandomBiasField(p=0.3),
            # tio.RandomNoise(p=0.5),
            # tio.RandomMotion(p=0.2),
            # tio.RandomFlip(axes=(2,)),
            tio.OneOf({
                tio.RandomAffine(scales=0.002): 0.8,
                tio.RandomElasticDeformation(): 0.2,
            }, p=0.3),
            tio.OneHot(3),
        ])
        val_transform = tio.Compose([
            tio.ZNormalization(masking_method=tio.ZNormalization.mean),
            tio.OneHot(3),
        ])
        test_transform = tio.Compose([
            tio.ZNormalization(masking_method=tio.ZNormalization.mean)
        ])

        self.transform = {
            'train': train_transform,
            'val': val_transform,
            'test': test_transform
        }

    def __getitem__(self, index):
        vol = tio.ScalarImage(self.paths[index] / 'imaging.nii.gz')
        vol = self.transform[self.mode](vol)
        label = tio.LabelMap(self.paths[index] / 'segmentation.nii.gz')
        label = self.transform[self.mode](label)
        return vol[tio.DATA], label[tio.DATA]

    def __len__(self):
        return len(self.paths)