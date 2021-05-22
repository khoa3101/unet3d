from torch.utils.data import Dataset
from pathlib import Path
import torchio as tio
import numpy as np
import torch

class KiTS2019(Dataset):
    def __init__(self, path, mode='train', n_channel=1):
        super(KiTS2019, self).__init__()

        self.paths = sorted(Path(path).glob('%s/case_*' % mode))
        self.mode = mode

        train_transform = tio.Compose([
            tio.RandomFlip(axes=(2,)),
            tio.OneOf({
                tio.RandomAffine(translation=0.002): 0.8,
                tio.RandomElasticDeformation(): 0.2,
            }),
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
        label = tio.LabelMap(self.paths[index] / 'segmentation.nii.gz')
        subject = tio.Subject(
            vol=tio.ScalarImage(tensor=vol),
            label=label
        )
        subject = self.transform[self.mode](subject)
        vol = subject['vol'][tio.DATA]
        label = subject['label'][tio.DATA]
        return vol.float(), label.float()

    def __len__(self):
        return len(self.paths)
