from dataset import Dataset3D, DataLoader3D
from data_augmentor import default_3D_augmentation_params, get_patch_size, get_default_augmentation, get_moreDA_augmentation
from loss import DiceCELoss, MultipleOutputLoss, dice_score
from models.model import Unet3D
from models.model_gconv import GUnet3D
from utils import poly_lr, sum_tensor
from logger import Logger
from _warnings import warn
from torch.optim.lr_scheduler import _LRScheduler
from torch.cuda.amp import GradScaler, autocast
from tqdm import trange
from time import time
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

def args_parser():
    parser = argparse.ArgumentParser(description='Unet3D Trainer')
    parser.add_argument('-p', '--path', required=True, type=str, help='Path to data folder')
    parser.add_argument('-o', '--output', default='.', type=str, help='Output folder to save training result')
    parser.add_argument('-lr', '--learning_rate', default=1e-2, type=float, help='Learning rate of optimzer')
    parser.add_argument('-b', '--batch_size', default=2, type=int, help='Batch size of dataloader')
    # parser.add_argument('-w', '--worker', default=6, type=int, help='Number of workers')
    parser.add_argument('-g', '--gpu', default=0, help='Index of GPU to train on, None for CPU training (not recommended)')
    parser.add_argument('-e', '--epoch', default=1000, type=int, help='Epoch to train model')
    parser.add_argument('-n', '--n_classes', default=3, type=int, help='Number of class to segmentation')
    parser.add_argument('-c', '--n_channels', default=1, type=int, help='Number of channels')
    parser.add_argument('-s', '--seed', default=31, type=str, help='Random seed')
    parser.add_argument('-gr', '--group', default=False, type=bool, help='Use group convolutional layer or not')
    parser.add_argument('-ckpt', '--checkpoint', default=None, type=str, help='Path to checkpoint if continue')
    args = parser.parse_args()
    return args

def create_folders(path='.'):
    if not os.path.isdir(path):
        os.mkdir(path)
    if not os.path.isdir(os.path.join(path, 'checkpoints')):
        os.mkdir(os.path.join(path, 'checkpoints'))
    if not os.path.isdir(os.path.join(path, 'logs')):
        os.mkdir(os.path.join(path, 'logs'))
    if not os.path.isdir(os.path.join(path, 'plots')):
        os.mkdir(os.path.join(path, 'plots')) 

class Trainer(object):
    def __init__(self, args, deterministic=True, batch_dice=False, fp16=True):
        self.args = args
        create_folders(self.args.output)
        self.logger = Logger('%s/logs' % self.args.output)
        self.device = torch.device(f'cuda:{self.args.gpu}' if self.args.gpu else 'cpu')

        self.fp16 = fp16
        self.amp_grad_scaler = None

        # Set random seed
        if deterministic:
            np.random.seed(self.args.seed)
            torch.manual_seed(self.args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.args.seed)
            cudnn.deterministic = True
            cudnn.benchmark = False
        else:
            cudnn.deterministic = False
            cudnn.benchmark = True
        
        # Set in self.initialize()
        self.batch_dice = batch_dice
        self.model = None
        self.loss = None

        self.optimizer = None
        self.scheduler = None

        self.train_dataset = self.val_dataset = None
        self.train_loader = self.val_loader = None
        self.train_gen = self.val_gen = None

        self.was_intitialized = False

        # Param for training
        self.batch_size = self.args.batch_size
        self.patch_size = np.array((128, 128, 128))

        self.max_epoch = self.args.epoch
        self.epoch = 1

        self.train_batches_per_epoch = 250
        self.val_batches_per_epoch = 50

        self.lr = args.learning_rate
        self.weight_decay = 3e-5

        self.oversample_foreground_percent = 0.33

        # Result
        self.best_val_loss = 1e8 # just a large number
        self.best_val_dice = 0.0

        self.train_loss = []
        self.train_dice_per_class = []
        self.train_dice_avg = []
        self.val_loss = []
        self.val_dice_per_class = []
        self.val_dice_avg = []

    def initialize(self):
        self.initialize_model_and_loss()
        self.initialize_optimizer_and_scheduler()
        self.initialize_loaders()
        self.was_intitialized = True

        if self.fp16:
            self.initialize_amp()

    def initialize_model_and_loss(self):
        if self.args.group:
            self.model = GUnet3D(n_channels=self.args.n_channels, n_classes=self.args.n_classes)
        else:
            self.model = Unet3D(n_channels=self.args.n_channels, n_classes=self.args.n_classes)
        self.model.to(self.device)

        self.loss = DiceCELoss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})
        self.loss.to(self.device)

    def initialize_optimizer_and_scheduler(self):
        assert self.model, 'self.initialize_model must be called first'

        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        self.scheduler = None

    def initialize_loaders(self):
        self.setup_DA_params()
        self.train_dataset = Dataset3D(self.args.path, mode='train')
        self.val_dataset = Dataset3D(self.args.path, mode='val')
        self.train_loader = DataLoader3D(
            self.train_dataset, self.patch_size, self.basic_generator_patch_size, self.batch_size, 
            oversample_foreground_percent=self.oversample_foreground_percent,
            pad_mode='constant'    
        )
        self.val_loader = DataLoader3D(
            self.val_dataset, self.patch_size, self.patch_size, self.batch_size,
            oversample_foreground_percent=self.oversample_foreground_percent,
            pad_mode='constant'
        )

        self.train_gen, self.val_gen = get_default_augmentation(
            self.train_loader, self.val_loader, 
            self.patch_size,
            self.data_aug_params
        )
        # self.train_gen, self.val_gen = get_moreDA_augmentation(
        #     self.train_loader, self.val_loader, 
        #     self.patch_size,
        #     self.data_aug_params
        # )

    def setup_DA_params(self):
        self.data_aug_params = default_3D_augmentation_params
        # - we increase roation angle from [-15, 15] to [-30, 30]
        # - scale range is now (0.7, 1.4), was (0.85, 1.25)
        # - we don't do elastic deformation anymore
        self.data_aug_params['rotation_x'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
        self.data_aug_params['rotation_y'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
        self.data_aug_params['rotation_z'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)

        self.basic_generator_patch_size = get_patch_size(
            self.patch_size, 
            self.data_aug_params['rotation_x'],
            self.data_aug_params['rotation_y'],
            self.data_aug_params['rotation_z'],
            self.data_aug_params['scale_range']
        )

        self.data_aug_params['scale_range'] = (0.7, 1.4)
        self.data_aug_params['do_elastic'] = False
        self.data_aug_params['selected_seg_channels'] = [0]

        self.data_aug_params['num_cached_per_thread'] = 2
        

    def initialize_amp(self):
        if self.amp_grad_scaler is None:
            self.amp_grad_scaler = GradScaler()
    
    def run(self):
        if self.device == 'cpu':
            self.logger.log('WARNING!! You are attempting to run training on a CPU. This can be VERY slow!')

        if cudnn.benchmark and cudnn.deterministic:
            warn(
                'torch.backends.cudnn.deterministic is True indicating a deterministic training is desired. '
                'But torch.backends.cudnn.benchmark is True as well and this will prevent deterministic training! '
                'If you want deterministic then set benchmark=False'
            )

        if not self.was_intitialized:
            self.initialize()
        
        _ = self.train_gen.next()
        _ = self.val_gen.next()

        if not self.device == 'cpu':
            torch.cuda.empty_cache()

        self.logger.log('Total number of parameters: %s' % (self.model.total_params()))
        self.logger.log('Total number of trainable parameters: %s' % (self.model.total_trainable_params()))

        while self.epoch <= self.max_epoch:
            self.logger.log('\nEpoch %d/%d' % (self.epoch, self.max_epoch))

            # Train
            self.model.train()
            epoch_start_time = time()

            train_loss_epoch = []
            train_result = {'tp': [], 'fp': [], 'fn': []}
            train_bar = trange(self.train_batches_per_epoch)
            for _ in train_bar:
                _loss, _tp, _fp, _fn = self.run_iteration(self.train_gen)
                train_loss_epoch.append(_loss)
                train_result['tp'] += (_tp)
                train_result['fp'] += (_fp)
                train_result['fn'] += (_fn)
                tmp_dice_per_class, tmp_dice_avg = self.dice_eval(train_result)
                train_bar.set_description('[%d/%d] Loss: %.4f, Dice: %s, Avg: %f' % (
                    self.epoch, self.max_epoch,
                    np.mean(train_loss_epoch), tmp_dice_per_class, tmp_dice_avg
                ))

            train_dice_per_class, train_dice_avg = self.dice_eval(train_result)
            self.train_loss.append(np.mean(train_loss_epoch))
            self.train_dice_per_class.append(train_dice_per_class)
            self.train_dice_avg.append(np.mean(train_dice_avg))
            self.logger.log('Train loss: %.4f, dice: %s, avg: %f' % (
                self.train_loss[-1], 
                self.train_dice_per_class[-1], 
                self.train_dice_avg[-1]
            ), print_console=False)

            # Val
            with torch.no_grad():
                self.model.eval()

                val_loss_epoch = []
                val_result = {'tp': [], 'fp': [], 'fn': []}
                val_bar = trange(self.val_batches_per_epoch)
                for _ in val_bar:
                    _loss, _tp, _fp, _fn = self.run_iteration(self.val_gen, do_backprop=False)
                    val_loss_epoch.append(_loss)
                    val_result['tp'] += (_tp)
                    val_result['fp'] += (_fp)
                    val_result['fn'] += (_fn)
                    tmp_dice_per_class, tmp_dice_avg = self.dice_eval(val_result)
                    val_bar.set_description('Loss: %.4f, Dice: %s, Avg: %f' % (
                        np.mean(val_loss_epoch), tmp_dice_per_class, tmp_dice_avg
                    ))
                
                val_dice_per_class, val_dice_avg = self.dice_eval(val_result)
                self.val_loss.append(np.mean(val_loss_epoch))
                self.val_dice_per_class.append(val_dice_per_class)
                self.val_dice_avg.append(np.mean(val_dice_avg))
                self.logger.log('Val loss: %.4f, dice: %s, avg: %f' % (
                    self.val_loss[-1], 
                    self.val_dice_per_class[-1], 
                    self.val_dice_avg[-1]
                ), print_console=False)

            continue_training = self.on_epoch_end()
            if not continue_training:
                break

            epoch_end_time = time()
            self.logger.log('This epoch took %.2f s\n' % (epoch_end_time - epoch_start_time))
            self.epoch += 1

    def run_iteration(self, dataloader, do_backprop=True):
        data_dict = next(dataloader)
        data = data_dict['data'].to(self.device)
        seg = data_dict['seg'].to(self.device)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                pred = self.model(data)
                del data
                _loss = self.loss(pred, seg)
            if do_backprop:
                self.amp_grad_scaler.scale(_loss).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            pred = self.model(data)
            del data
            _loss = self.loss(pred, seg)

            if do_backprop:
                _loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 12)
                self.optimizer.step()

        _tp, _fp, _fn = self.run_eval(pred, seg)
        del seg

        return _loss.item(), _tp, _fp, _fn

    def run_eval(self, pred, seg):
        with torch.no_grad():
            pred_softmax = F.softmax(pred, 1)
            pred_argmax = pred_softmax.argmax(1)
            seg = seg[:, 0]
            axis = tuple(range(1, len(seg.shape)))
            _tp = torch.zeros((seg.shape[0], self.args.n_classes - 1)).to(pred_argmax.device)
            _fp = torch.zeros((seg.shape[0], self.args.n_classes - 1)).to(pred_argmax.device)
            _fn = torch.zeros((seg.shape[0], self.args.n_classes - 1)).to(pred_argmax.device)
            for c in range(1, self.args.n_classes):
                _tp[:, c - 1] = sum_tensor((pred_argmax == c).float() * (seg == c).float(), axis=axis)
                _fp[:, c - 1] = sum_tensor((pred_argmax == c).float() * (seg != c).float(), axis=axis)
                _fn[:, c - 1] = sum_tensor((pred_argmax != c).float() * (seg == c).float(), axis=axis)

            _tp = _tp.detach().cpu().numpy()
            _fp = _fp.detach().cpu().numpy()
            _fn = _fn.detach().cpu().numpy()

        return list(_tp), list(_fp), list(_fn)

    def dice_eval(self, result):
        tp = np.sum(result['tp'], axis=0)
        fp = np.sum(result['fp'], axis=0)
        fn = np.sum(result['fn'], axis=0)

        # Exclude background
        dice_per_class = dice_score(tp, fp, fn, smooth=1e-5)
        # dice_per_class = [idx for idx in [
        #     dice_score(i, j, k, smooth=1e-5) for i, j, k in zip(tp, fp, fn)
        # ] if not np.isnan(idx)]
        dice_avg = np.mean(dice_per_class)

        return dice_per_class, dice_avg

    def on_epoch_end(self):
        self.plot_progress('%s/plots/training_%s.png' % (self.args.output, self.logger.timestamp))

        self.update_lr()

        self.save_checkpoint('%s/checkpoints/latest_%s.pth' % (self.args.output, self.logger.timestamp))
        if self.best_val_loss > self.val_loss[-1]:
            self.best_val_loss = self.val_loss[-1]
            os.system('cp %s/checkpoints/latest_%s.pth %s/checkpoints/best_loss_%s.pth' % (
                self.args.output, self.logger.timestamp,
                self.args.output, self.logger.timestamp
            ))
            self.logger.log('Better loss')
        if self.best_val_dice < self.val_dice_avg[-1]:
            self.best_val_dice = self.val_dice_avg[-1]
            os.system('cp %s/checkpoints/latest_%s.pth %s/checkpoints/best_dice_%s.pth' % (
                self.args.output, self.logger.timestamp,
                self.args.output, self.logger.timestamp
            ))
            self.logger.log('Better dice')

        continue_training = self.epoch <= self.max_epoch
        # It can rarely happen that the momentum is too high for some dataset. 
        # If at epoch 100 the estimated validation Dice is still 0 then we reduce the momentum from 0.99 to 0.95
        if self.epoch == 100:
            if self.val_dice_avg[-1] == 0:
                self.optimizer.param_groups[0]["momentum"] = 0.95
                self.logger.log(
                    "At epoch 100, the mean foreground Dice was 0. This can be caused by a too "
                    "high momentum. High momentum (0.99) is good for datasets where it works, but "
                    "sometimes causes issues such as this one. Momentum has now been reduced to "
                    "0.95 and network weights have been reinitialized"
                )
        return continue_training

    def update_lr(self):
        self.optimizer.param_groups[0]['lr'] = poly_lr(self.epoch, self.max_epoch, self.lr, 0.9)
        self.logger.log('lr:', np.round(self.optimizer.param_groups[0]['lr'], decimals=6))

    def save_checkpoint(self, fname, save_optimizer=True):
        start_time = time()
        state_dict = self.model.state_dict()
        
        if self.scheduler and hasattr(self.scheduler, 'state_dict'):
            scheduler_state_dict = self.scheduler.state_dict()
        else:
            scheduler_state_dict = None

        if save_optimizer:
            optimizer_state_dict = self.optimizer.state_dict()
        else:
            optimizer_state_dict = None

        print('Saving checkpoint...')
        save_this = {
            'epoch': self.epoch,
            'state_dict': state_dict,
            'optimizer': optimizer_state_dict,
            'scheduler': scheduler_state_dict,
            'plot_stuff': (self.train_loss, self.val_loss, self.train_dice_avg, self.val_dice_avg),
            'best': (self.best_val_loss, self.best_val_dice)
        }
        if self.amp_grad_scaler is not None:
            save_this['amp_grad_scaler'] = self.amp_grad_scaler.state_dict()
        torch.save(save_this, fname)
        print('Done, saving took %.2f seconds' % (time() - start_time))

    def load_checkpoint(self, fname):
        print('Loading checkpoint...')
        if not self.was_initialized:
            self.initialize()

        checkpoint = torch.load(fname)
        if self.fp16:
            self.initialize_amp()
            if 'amp_grad_scaler' in checkpoint.keys():
                self.amp_grad_scaler.load_state_dict(checkpoint['amp_grad_scaler'])
        self.epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['state_dict'])
        
        if checkpoint['optimizer'] is not None:
            self.optimizer.load_stat_dict(checkpoint['optimizer'])

        if self.scheduler is not None and hasattr(self.scheduler, 'load_state_dict') and checkpoint['scheduler'] is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler'])

        if issubclass(self.scheduler.__class__, _LRScheduler):
            self.scheduler.step(self.epoch)

        self.train_loss, self.val_loss, self.train_dice_avg, self.val_dice_avg = checkpoint['plot_stuff']
        self.best_val_loss, self.best_val_dice = checkpoint['best']

    def plot_progress(self, fname):
        try:
            font = {
                'weight': 'normal',
                'size': 10
            }
            matplotlib.rc('font', **font)

            fig = plt.figure(figsize=(30, 24))
            ax1 = fig.add_subplot(111)
            ax2 = ax1.twinx()

            epochs = list(range(self.epoch))

            ax1.plot(epochs, self.train_loss, color='b', ls='-', label='Train loss')
            ax1.plot(epochs, self.val_loss, color='r', ls='-', label='Val loss')
            ax2.plot(epochs, self.train_dice_avg, color='y', ls='--', label='Train dice')
            ax2.plot(epochs, self.val_dice_avg, color='g', ls='--', label='Val dice')

            plt.title('Training summary')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax2.set_ylabel('Dice score')
            ax1.legend()
            ax2.legend(loc=9)
            
            fig.savefig(fname)
            plt.close()

        except IOError:
            print('Failed to plot: ', sys.exc_info())

    def save_stats(self, fname):
        pass


if __name__ == '__main__':
    args = args_parser()

    trainer = Trainer(args)
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    trainer.run()
