from ast import parse
from dataset import Dataset3D
from loss import dice_score
from models.model import Unet3D
from models.model_gconv import GUnet3D
from utils import get_gaussian, compute_steps_for_sliding_window
from preprocess import check_separate_z, get_lowres_axis, resample_instance
from batchgenerators.augmentations.utils import pad_nd_image
from tqdm import trange
from time import time
from pathlib import Path
import os
import argparse
import pickle
import torch
import torch.nn.functional as F
import numpy as np
import SimpleITK as sitk

def args_parser():
    parser = argparse.ArgumentParser(description='Unet3D Tester')
    parser.add_argument('-p', '--path', required=True, type=str, help='Path to data folder')
    parser.add_argument('-s', '--stats', required=True, type=str, help='Path to stats of dataset')
    parser.add_argument('-o', '--output', default='.', type=str, help='Output folder to save testing result')
    parser.add_argument('-g', '--gpu', default=0, help='Index of GPU to train on, None for CPU training (not recommended)')
    parser.add_argument('-ckpt', '--checkpoint', required=True, type=str, help='Path to checkpoint for testing')
    parser.add_argument('-gt', '--ground_truth', default=None, type=str, help='Path to ground truth folder for computing score')
    parser.add_argument('-m', '--mode', default='test', choices=['val', 'test'], help='Mode of testing phase (val or test)')
    args = parser.parse_args()
    return args

def create_folders(path='.'):
    if not os.path.isdir(path):
        os.mkdir(path)
    if not os.path.isdir(os.path.join(path, 'predictions')):
        os.mkdir(os.path.join(path, 'predictions'))

class Tester(object):
    def __init__(self, args, do_tta=True, step_size=0.5):
        self.args = args
        create_folders(self.args.output)
        with open(self.args.stats, 'rb') as f:
            self.stats = pickle.load(f)
        self.do_tta = do_tta
        self.step_size = step_size
        self.device = torch.device(f'cuda:{self.args.gpu}' if self.args.gpu else 'cpu')

        self.patch_size = np.array((128, 128, 128))
        self.n_classes = 3
        self.n_channels = 1
        self.group = False

        self.initialize()

    def initialize(self):
        if self.group:
            self.model = GUnet3D(n_channels=self.n_channels, n_classes=self.n_classes)
        else:
            self.model = Unet3D(n_channels=self.n_channels, n_classes=self.n_classes)
        self.model.to(self.device)

        checkpoint = torch.load(self.args.checkpoint)
        self.model.load_state_dict(checkpoint['state_dict'])

        self.dataset = Dataset3D(self.args.path, mode=self.args.mode)

    def run(self):
        time_start = time()
        if self.device == 'cpu':
            self.logger.log('WARNING!! You are attempting to run training on a CPU. This can be VERY slow!')

        if not self.device == 'cpu':
            torch.cuda.empty_cache()

        print('\nComputing Gaussian..')
        gaussian_importance_map = get_gaussian(self.patch_size, sigma_scale=1./8)
        gaussian_importance_map = torch.from_numpy(gaussian_importance_map)
        gaussian_importance_map = gaussian_importance_map.to(self.device)
        # half precision for the predictions should be good enough. If the outputs here are half,
        # the gaussian_importance_map should be as well
        gaussian_importance_map = gaussian_importance_map.half()

        # make sure we did not round anything to 0
        gaussian_importance_map[gaussian_importance_map == 0] = gaussian_importance_map[gaussian_importance_map != 0].min()
        self.weighted_map = gaussian_importance_map

        print('\nPredicting..')
        test_bar = trange(len(self.dataset))
        for i in test_bar:
            case, property = self.dataset[i]['data'], self.dataset[i]['property']
            predicted_segmentations, _ = self.run_iteration(case)
            self.save_pred(predicted_segmentations, property)

        if self.args.ground_truth is not None:
            print('\nComputing result..')
            test_bar = trange(len(self.dataset))
            dice = np.zeros(self.n_classes-1)
            for i in test_bar:
                case_ID = self.dataset[i]['property']['ID']
                path_gt = Path(self.args.ground_truth) / case_ID / 'segmentation.nii.gz'
                path_pred = Path(self.args.output) / ('prediction_' + case_ID + '.nii.gz')
                dice += self.comput_dice(str(path_pred), str(path_gt))
                test_bar.set_description('Dice: %.4f, Avg: %.4f' % (dice / (i+1), (dice / (i+1)).mean()))

        time_end = time()
        print('Done in %.2f s\n' % (time_end - time_start))

    def run_iteration(self, case):
        vol = case[:-1]
        label = case[-1:]

        vol, slicer = pad_nd_image(vol, self.patch_size, 'constanst', {'constant_values': 0}, True, None)
        shape = vol.shape

        vol = torch.from_numpy(vol).to(self.device)
        label = torch.from_numpy(label).to(self.device)
        
        steps = compute_steps_for_sliding_window(self.patch_size, shape[1:], self.step_size)
        num_tiles = len(steps[0]) * len(steps[1]) * len(steps[2])

        if num_tiles > 1:
            tmp_map = self.weighted_map
        else:
            tmp_map = None

        if num_tiles > 1:
            weight_pred = tmp_map
        else:
            weight_pred = torch.ones(vol.shape[1:], device=self.device)

        aggregated_preds = torch.zeros(
            [self.n_classes] + list(vol.shape[1:]),
            dtype=torch.half, device=self.device
        )
        aggregated_weight_preds = torch.zeros(
            [self.n_classes] + list(vol.shape[1:]),
            dtype=torch.half, device=self.device
        )

        for z in steps[0]:
            lb_z = z
            ub_z = z + self.patch_size[0]
            for x in steps[1]:
                lb_x = x
                ub_x = x + self.patch_size[1]
                for y in steps[2]:
                    lb_y = y
                    ub_y = y + self.patch_size[2]

                    pred_patch = self.predict_and_mirror(vol[None, :, lb_z:ub_z, lb_x:ub_x, lb_y:ub_y], self.do_tta)[0]

                    pred_patch = pred_patch.half()

                    aggregated_preds[:, lb_z:ub_z, lb_x:ub_x, lb_y:ub_y] += pred_patch
                    aggregated_weight_preds[:, lb_z:ub_z, lb_x:ub_x, lb_y:ub_y] += weight_pred

        # we reverse the padding here (remember that we padded the input to be at least as large as the patch size
        slicer = tuple(
            [
                slice(0, aggregated_preds.shape[i]) for i in
                range(len(aggregated_preds.shape) - (len(slicer) - 1))
            ] + slicer[1:]
        )
        aggregated_preds = aggregated_preds[slicer]
        aggregated_weight_preds = aggregated_weight_preds[slicer]

        class_probabilities = aggregated_preds / aggregated_weight_preds
        predicted_segmentations = class_probabilities.argmax(0)

        predicted_segmentations = predicted_segmentations.detach().to(torch.device('cpu')).numpy()
        class_probabilities = class_probabilities.detach().to(torch.device('cpu')).numpy()

        return predicted_segmentations, class_probabilities

    def predict_and_mirror(self, vol, do_tta, mirror_axes=(0, 1, 2)):
        vol = vol.to(self.device)
        pred_torch = torch.zeros([1, self.n_classes] + list(vol.shape[2:]), dtype=torch.float, device=self.device)

        non_linear = lambda x: F.softmax(x, 1)

        if do_tta:
            mirror_idx = 8
            num_results = 2 ** len(mirror_axes)
        else:
            mirror_idx = 1
            num_results = 1

        for m in range(mirror_idx):
            if m == 0:
                pred = non_linear(self.model(vol))
                pred_torch += 1/num_results * pred
            
            if m == 1 and (0 in mirror_axes):
                pred = non_linear(self.model(torch.flip(vol, (2,))))
                pred_torch += 1/num_results * torch.flip(pred, (2,))

            if m == 2 and (1 in mirror_axes):
                pred = non_linear(self.model(torch.flip(vol, (3,))))
                pred_torch += 1/num_results * torch.flip(pred, (3,))

            if m == 3 and (2 in mirror_axes):
                pred = non_linear(self.model(torch.flip(vol, (4,))))
                pred_torch += 1/num_results * torch.flip(pred, (4,))
            
            if m == 4 and (0 in mirror_axes) and (1 in mirror_axes):
                pred = non_linear(self.model(torch.flip(vol, (2, 3))))
                pred_torch += 1/num_results * torch.flip(pred, (2, 3))

            if m == 5 and (1 in mirror_axes) and (2 in mirror_axes):
                pred = non_linear(self.model(torch.flip(vol, (3, 4))))
                pred_torch += 1/num_results * torch.flip(pred, (3, 4))

            if m == 6 and (0 in mirror_axes) and (2 in mirror_axes):
                pred = non_linear(self.model(torch.flip(vol, (2, 4))))
                pred_torch += 1/num_results * torch.flip(pred, (2, 4))

            if m == 7 and (0 in mirror_axes) and (1 in mirror_axes) and (2 in mirror_axes):
                pred = non_linear(self.model(torch.flip(vol, (2, 3, 4))))
                pred_torch += 1/num_results * torch.flip(pred, (2, 3, 4))

        return pred_torch

    def save_pred(self, pred, property, order=1, order_z=0, cval=0):
        if np.any([i != j for i, j in zip(np.array(property['new_shape']), np.array(property['shape']))]):
            if check_separate_z(property['spacing']):
                do_separate_z = True
                lowres_axis = get_lowres_axis(property['spacing'])
            elif check_separate_z(self.stats['target_spacing']):
                do_separate_z = True
                lowres_axis = get_lowres_axis(self.stats['target_spacing'])
            else:
                do_separate_z = False
                lowres_axis = None

            if lowres_axis is not None and len(lowres_axis) != 1:
                # case (0.24, 1.25, 1.25)
                do_separate_z = False

            pred_old_spacing = resample_instance(
                pred, False, property['shape'], axis=lowres_axis,
                order=order, do_separate_z=do_separate_z, cval=cval, order_z=order_z
            )
        else:
            pred_old_spacing = pred
                    
        pred_itk = sitk.GetImageFromArray(pred_old_spacing.transpose().astype(np.uint8))
        pred_itk.SetSpacing(property['spacing'])
        pred_itk.SetOrigin(property['origin'])
        pred_itk.SetDirection(property['direction'])
        sitk.WriteImage(pred_itk, os.path.join(self.output, 'prediction_' + property['ID'] + '.nii.gz'))

    def compute_score(self, path_pred, path_gt):
        pred = sitk.ReadImage(path_pred)
        pred = sitk.GetArrayFromImage(pred)
        gt = sitk.ReadImage(path_gt)
        gt = sitk.GetArrayFromImage(pred)

        tp = np.zeros(self.n_classes - 1)
        fp = np.zeros(self.n_classes - 1)
        fn = np.zeros(self.n_classes - 1)

        for c in range(1, self.n_classes):
            tp[c - 1] = ((pred == c).astype(np.float32) * (gt == c).astype(np.float32)).sum()
            fp[c - 1] = ((pred == c).astype(np.float32) * (gt != c).astype(np.float32)).sum()
            fn[c - 1] = ((pred != c).astype(np.float32) * (gt == c).astype(np.float32)).sum()
        
        return dice_score(tp, fp, fn, smooth=1e-5)


if __name__ == '__main__':
    args = args_parser()

    tester = Tester(args)
    tester.run()