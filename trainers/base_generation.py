import logging

import hydra
import torch
import torch.nn.functional as F
import wandb

import engine
from trainers.base import BaseTrainer
from utils import transforms
from utils.logger import wandb_log


class GenerationTrainer(BaseTrainer):

    def load_dataset(self, load_db=False):
        logging.info('Creating Datasets')
        train_dataset = hydra.utils.instantiate(self.config.dataset.train_dataset)
        test_dataset = hydra.utils.instantiate(self.config.dataset.test_dataset)

        logging.info(f'Number of Train data: {len(train_dataset)}')
        logging.info(f'Number of Test data: {len(test_dataset)}')

        self.dataset = {
            'train': train_dataset,
            'test': test_dataset
        }

        if load_db:
            db_dataset = hydra.utils.instantiate(self.config.dataset.db_dataset)
            self.dataset['db'] = db_dataset
            logging.info(f'Number of DB data: {len(db_dataset)}')

    def load_dataloader(self):
        assert self.dataset is not None
        bs = self.config.batch_size
        train_loader = engine.dataloader(self.dataset['train'], bs, shuffle=True, drop_last=True)
        test_loader = engine.dataloader(self.dataset['test'], self.config.batch_size, shuffle=False, drop_last=False)

        self.dataloader = {
            'train': train_loader,
            'test': test_loader
        }

        if 'db' in self.dataset:
            db_loader = engine.dataloader(self.dataset['db'], self.config.batch_size, shuffle=False, drop_last=False)
            self.dataloader['db'] = db_loader

    def record_wandb(self, image, recs, kwargs, prefix='train'):
        if self.config.wandb and kwargs['bidx'] == 0:
            n = 8  # maximum first n images
            unnormalize = transforms.unnormalize_transform(self.config.dataset.norm)
            to_pil = transforms.to_pil()

            def process_image(inp):
                inp = inp.cpu().detach()
                inp = unnormalize(inp)
                inp = inp.clamp(0, 1)
                inp = to_pil(inp)
                inp = wandb.Image(inp)
                return inp

            inps = [process_image(inp) for inp in image[:n]]
            oups = [process_image(oup) for oup in recs[:n]]

            wandb_log({f'{prefix}/inps': inps, f'{prefix}/oups': oups})

    def inference_one_batch(self, *args, **kwargs):
        device = self.device

        data, meters = args
        image, labels, index = data
        image, labels = image.to(device), labels.to(device)

        with torch.no_grad():
            latents, recs = self.model(image)
            loss = self.criterion(latents, image, recs, index)

            # store results
            meters['loss'].update(loss.item(), image.size(0))
            for key in self.criterion.losses:
                meters[key].update(self.criterion.losses[key].item(), image.size(0))

        if kwargs.get('codes_callback') is not None:
            codes_callback = kwargs['codes_callback']
            codes = codes_callback(latents)
        else:
            codes = latents[0]  # take mu/featmap as codes
            if len(codes.size()) == 4:
                if kwargs.get('avg_pool', True):  # if it is a featmap, take avgpool
                    codes = F.adaptive_avg_pool2d(codes, (1, 1))
                    codes = torch.flatten(codes, 1)
                else:  # save index only
                    codes = self.model.vq.get_encoding_ind(codes)  # take index
                    codes = torch.flatten(codes, 1)

        self.record_wandb(image, recs, kwargs, 'test')

        return {
            'codes': codes.cpu(),
            'labels': labels.cpu()
        }

    def train_one_batch(self, *args, **kwargs):
        device = self.device

        data, meters = args
        image, labels, index = data
        image, labels, index = image.to(device), labels.to(device), index.to(device)

        # clear gradient
        self.optimizer.zero_grad()

        latents, recs = self.model(image)
        loss = self.criterion(latents, image, recs, index)

        # backward and update
        loss.backward()
        self.optimizer.step()

        # store results
        meters['loss'].update(loss.item(), image.size(0))
        for key in self.criterion.losses:
            meters[key].update(self.criterion.losses[key].item(), image.size(0))

        self.record_wandb(image, recs, kwargs)
