import logging
import math
from collections import defaultdict

import torch
from models.loss.moco import MoCoV3Loss
from tqdm import tqdm

from trainers.base_contrastive import ContrastiveTrainer
from utils.misc import AverageMeter

ALLOWED_TRANSFORM = ['byol']


class MoCoV3Trainer(ContrastiveTrainer):
    def __init__(self, config):
        super(MoCoV3Trainer, self).__init__(config)

        if self.config['dataset_kwargs']['transform'] not in ALLOWED_TRANSFORM:
            logging.info(f'Input transform is {self.config["dataset_kwargs"]["transform"]}')
            logging.info(f'Force changing transform to {ALLOWED_TRANSFORM[0]}')
            self.config['dataset_kwargs']['transform'] = ALLOWED_TRANSFORM[0]

        self.moco_m = self.config['method']['param'].get('moco_m', 0.99)
        self.moco_m_cos = self.config['method']['param'].get('moco_m_cos', True)
        self.current_moco_m = self.moco_m

    def load_criterion(self):
        self.criterion = MoCoV3Loss(multiclass=self.config['multiclass'],
                                    **self.config['method']['param'])

    def inference_one_batch(self, *args, **kwargs):
        device = self.device

        data, meters = args
        images, labels, index = data
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            codes, _ = self.model(images, use_head=False)

        return {
            'codes': codes,
            'labels': labels
        }

    def adjust_moco_momentum(self, epoch):
        """
        Adjust moco momentum based on current epoch

        Copy from https://github.com/facebookresearch/moco-v3
        """
        m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / self.config['epochs'])) * (1. - self.moco_m)
        return m

    def train_one_batch(self, *args, **kwargs):
        device = self.device

        ep, bidx, blength = kwargs['ep'], kwargs['bidx'], kwargs['blength']

        data, meters = args
        images, labels, index = data
        images_0, images_1 = images
        images_0, images_1, labels = images_0.to(device), images_1.to(device), labels.to(device)

        # clear gradient
        self.optimizer.zero_grad()

        codes_0, feats_0 = self.model(images_0, use_head=True, use_momentum=False)
        codes_1, feats_1 = self.model(images_1, use_head=True, use_momentum=False)

        # momemtum stuff here
        with torch.no_grad():
            if self.moco_m_cos:
                self.current_moco_m = self.adjust_moco_momentum(ep + bidx / blength)

            self.model.update_momentum_encoder(self.current_moco_m)

            _, momentum_feats_0 = self.model(images_0, use_head=True, use_momentum=True)
            _, momentum_feats_1 = self.model(images_1, use_head=True, use_momentum=True)

        loss = self.criterion(codes_0, codes_1,
                              feats_0, feats_1,
                              momentum_feats_0, momentum_feats_1,
                              labels, index)

        # backward and update
        loss.backward()
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

        self.scheduler.step()

        return meters
