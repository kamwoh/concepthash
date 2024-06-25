import logging
from collections import defaultdict

import numpy as np
import torch
from models.loss.mae import MAELoss
from tqdm import tqdm

import engine
from trainers.base import BaseTrainer
from utils.misc import AverageMeter

ALLOWED_TRANSFORM = ['mae']


class MAETrainer(BaseTrainer):
    def __init__(self, config):
        super(MAETrainer, self).__init__(config)

        if self.config['dataset_kwargs']['transform'] not in ALLOWED_TRANSFORM:
            logging.info(f'Input transform is {self.config["dataset_kwargs"]["transform"]}')
            logging.info(f'Force changing transform to {ALLOWED_TRANSFORM[0]}')
            self.config['dataset_kwargs']['transform'] = ALLOWED_TRANSFORM[0]

        self.mask_ratio = self.config['method']['param'].get('mask_ratio', 0.75)

    def get_param_groups(self, is_backbone=True):
        regularized = []
        not_regularized = []

        if is_backbone:
            module = self.model.get_backbone()
        else:
            module = self.model.get_training_modules()

        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            # we do not regularize biases nor Norm parameters
            if name.endswith(".bias") or len(param.shape) == 1:
                not_regularized.append(param)
            else:
                regularized.append(param)

        return regularized, not_regularized

    def load_optimizer_and_scheduler(self):
        assert self.model is not None
        lr = self.config['optim_kwargs']['lr']
        backbone_lr_scale = self.config['optim_kwargs']['backbone_lr_scale']

        # regularize only certain groups
        regularized, not_regularized = self.get_param_groups(is_backbone=True)
        regularized_training, not_regularized_training = self.get_param_groups(is_backbone=False)

        params = [{'params': regularized,
                   'lr': lr * backbone_lr_scale},
                  {'params': not_regularized,
                   'lr': lr * backbone_lr_scale,
                   'weight_decay': 0.},
                  {'params': regularized_training},
                  {'params': not_regularized_training,
                   'weight_decay': 0.}]

        if backbone_lr_scale == 0:  # if not training backbone, freeze it
            for _ in range(2):  # pop twice
                backbone_params = params.pop(0)['params']
                for p in backbone_params:
                    p.requires_grad_(False)

        self.optimizer = engine.optimizer(self.config, params)
        self.scheduler = engine.scheduler(self.config, self.optimizer, first_n=2)

    def load_criterion(self):
        self.criterion = MAELoss(**self.config['method']['param'])

    def inference_one_batch(self, *args, **kwargs):
        device = self.device

        data, meters = args
        images, labels, index = data
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        with torch.no_grad():
            codes = self.model(images)

        return {
            'codes': codes,
            'labels': labels
        }

    def update_weight_decay_rate(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            if i % 2 == 0:  # only the first group is regularized  ( the order is [ r, not r, r, not r, ... ] )
                param_group["weight_decay"] = self.cosine_scheduler(epoch,
                                                                    self.config['epochs'],
                                                                    0,
                                                                    self.config['optim_kwargs']['weight_decay'],
                                                                    self.config['optim_kwargs']['final_weight_decay'])

    def cosine_scheduler(self, epoch, epochs, warmup_epochs, base_value, final_value):
        # epoch = global iterations (ep + it / niters)
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return base_value * (epoch / warmup_epochs)
        else:
            scale = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            return final_value + (0.5 * (base_value - final_value) * (1 + np.cos(np.pi * scale)))

    def train_one_batch(self, *args, **kwargs):
        device = self.device

        ep, bidx, blength = kwargs['ep'], kwargs['bidx'], kwargs['blength']

        data, meters = args
        images, labels, index = data

        images = images.to(device, non_blocking=True)

        # clear gradient
        self.optimizer.zero_grad()

        # only the 2 global views pass through the teacher
        codes, pred, mask = self.model.forward_train(images, self.mask_ratio)
        loss = self.criterion(images, pred, mask, self.model)

        # backward and update
        loss.backward()
        self.scheduler.step(ep + bidx / blength)
        self.optimizer.step()

        # store results
        meters['loss'].update(loss.item())
        for key in self.criterion.losses:
            meters[key].update(self.criterion.losses[key].item())

    def train_one_epoch(self, **kwargs):
        assert self.is_ready_for_training()

        self.model.train()
        self.criterion.train()
        meters = defaultdict(AverageMeter)
        blength = len(self.dataloader['train'])

        with tqdm(self.dataloader['train'], bar_format='{l_bar}{bar:10}{r_bar}') as tepoch:
            for i, data in enumerate(tepoch):
                self.train_one_batch(data, meters, bidx=i, blength=blength, **kwargs)
                tepoch.set_postfix({k: v.avg for k, v in meters.items()})

        return meters
