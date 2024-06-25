import math
from collections import defaultdict

import hydra
import numpy as np
import torch
from tqdm import tqdm

from trainers.base_contrastive import ContrastiveTrainer
from utils import io
from utils.misc import AverageMeter


class DINOTrainer(ContrastiveTrainer):
    def __init__(self, config):
        super(DINOTrainer, self).__init__(config)

        self.not_regularized_optimizer = None
        self.teacher_momentum = self.config.criterion.teacher_momentum

    def to_device(self, device=None):
        if device is None:
            device = self.device

        if self.model is not None:
            self.model = self.model.to(device)

        if self.optimizer is not None:
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)

        if self.not_regularized_optimizer is not None:
            for state in self.not_regularized_optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)

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
        lr = self.config.optim.lr
        backbone_lr_scale = self.config.backbone_lr_scale

        # regularize only certain groups
        regularized, not_regularized = self.get_param_groups(is_backbone=True)
        regularized_training, not_regularized_training = self.get_param_groups(is_backbone=False)

        regularized_params = [{'params': regularized,
                               'lr': lr * backbone_lr_scale},
                              {'params': regularized_training}]
        not_regularized_params = [{'params': not_regularized,
                                   'lr': lr * backbone_lr_scale,
                                   'weight_decay': 0.},
                                  {'params': not_regularized_training,
                                   'weight_decay': 0.}]

        if backbone_lr_scale == 0:  # if not training backbone, freeze it
            for params in [regularized_params, not_regularized_params]:
                backbone_params = params.pop(0)['params']
                for p in backbone_params:
                    p.requires_grad_(False)

        self.optimizer = hydra.utils.instantiate(self.config.optim, regularized_params)
        self.not_regularized_optimizer = hydra.utils.instantiate(self.config.optim, not_regularized_params)

        self.scheduler = hydra.utils.instantiate(self.config.scheduler, self.optimizer)

    def save_training_state(self, fn):
        optimsd = self.optimizer.state_dict()
        nr_optimsd = self.not_regularized_optimizer.state_dict()
        schedulersd = self.scheduler.state_dict()
        io.fast_save({'optim': optimsd,
                      'nr_optim': nr_optimsd,
                      'scheduler': schedulersd}, fn)

    def load_training_state(self, fn):
        sd = torch.load(fn, map_location='cpu')
        self.optimizer.load_state_dict(sd['optim'])
        self.not_regularized_optimizer.load_state_dict(sd['nr_optim'])
        self.scheduler.load_state_dict(sd['scheduler'])

    def inference_one_batch(self, *args, **kwargs):
        device = self.device

        data, meters = args
        images, labels, index = data
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        with torch.no_grad():
            _, codes, _ = self.model(images, forward_student=False)

        return {
            'codes': codes,
            'labels': labels
        }

    def adjust_momentum(self, epoch):
        """
        Adjust moco momentum based on current epoch

        Copy from https://github.com/facebookresearch/moco-v3
        """
        m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / self.config.epochs)) * (1. - self.teacher_momentum)
        return m

    def clip_gradients(self, clip):
        if clip == 0:
            return

        norms = []
        for name, p in self.model.student_backbone.named_parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                norms.append(param_norm.item())
                clip_coef = clip / (param_norm + 1e-6)
                if clip_coef < 1:
                    p.grad.data.mul_(clip_coef)
        return norms

    def cancel_gradients_last_layer(self, epoch, freeze_last_layer):
        if epoch >= freeze_last_layer:
            return
        for n, p in self.model.student_backbone.named_parameters():
            if "last_layer" in n:
                p.grad = None

    def update_weight_decay_rate(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['weight_decay'] = self.cosine_scheduler(epoch,
                                                                self.config.epochs,
                                                                0,
                                                                self.config.method_params.weight_decay,
                                                                self.config.method_params.final_weight_decay)

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

        images = [img.to(device, non_blocking=True) for img in images]

        # clear gradient
        self.optimizer.zero_grad()
        self.not_regularized_optimizer.zero_grad()

        # only the 2 global views pass through the teacher
        _, teacher_codes, teacher_output = self.model(images[:2], forward_student=False)
        _, student_codes, student_output = self.model(images, forward_student=True)
        loss = self.criterion(student_output, teacher_output, ep, self.model)

        # backward and update
        loss.backward()
        self.clip_gradients(clip=self.config.method_params.gradient_clipping)
        self.cancel_gradients_last_layer(ep, 1)
        self.update_weight_decay_rate(ep + bidx / blength)
        self.scheduler.step(ep + bidx / blength)
        self.optimizer.step()
        self.not_regularized_optimizer.step()

        # EMA update for the teacher
        with torch.no_grad():
            m = self.cosine_scheduler(ep + bidx / blength,
                                      self.config.epochs,
                                      0,
                                      base_value=self.teacher_momentum,
                                      final_value=1)
            for param_q, param_k in zip(self.model.student_backbone.parameters(),
                                        self.model.backbone.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
            for param_q, param_k in zip(self.model.student_hash_fc.parameters(),
                                        self.model.teacher_hash_fc.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
            for param_q, param_k in zip(self.model.student_head.parameters(),
                                        self.model.teacher_head.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

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
