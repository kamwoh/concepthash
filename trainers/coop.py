import gc
import logging
import os

import hydra
import torch
from omegaconf import DictConfig

import engine
from trainers.base import BaseTrainer


class COOPTrainer(BaseTrainer):

    def __init__(self, config: DictConfig):
        super().__init__(config)

        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    def load_dataset(self, load_db=True):
        logging.info('Creating Datasets')
        train_dataset = hydra.utils.instantiate(self.config.dataset.train_dataset)
        test_dataset = hydra.utils.instantiate(self.config.dataset.test_dataset)
        db_dataset = hydra.utils.instantiate(self.config.dataset.db_dataset)

        logging.info(f'Number of Train data: {len(train_dataset)}')
        logging.info(f'Number of Query data: {len(test_dataset)}')
        logging.info(f'Number of Database data: {len(db_dataset)}')

        self.dataset = {
            'train': train_dataset,
            'test': test_dataset,
            'db': db_dataset
        }

    def load_dataloader(self):
        assert self.dataset is not None
        bs = self.config.batch_size
        train_loader = engine.dataloader(self.dataset['train'], bs, shuffle=True, drop_last=True)
        test_loader = engine.dataloader(self.dataset['test'], bs, shuffle=False, drop_last=False)
        db_loader = engine.dataloader(self.dataset['db'], bs, shuffle=False, drop_last=False)

        self.dataloader = {
            'train': train_loader,
            'test': test_loader,
            'db': db_loader
        }

    def parse_model_output(self, output):
        codes, logits = output
        if isinstance(logits, dict):
            return logits

        return {
            'codes': codes,
            'logits': logits
        }

    def compute_features_one_batch(self, data):
        device = self.device

        image, labels, index = data
        image = image.to(device)
        labels = labels.to(device)
        # if self.config.model.get('upt_config', {}).get('v3'):
        if self.config.model.get('pass_labels'):
            output = self.model(image, labels)
        else:
            output = self.model(image)

        return (image, labels, index), self.parse_model_output(output)

    def inference_one_batch(self, *args, **kwargs):
        data, meters = args

        with torch.no_grad():
            data, output = self.compute_features_one_batch(data)

            image, labels, index = data
            if self.config.dataset.multiclass:
                loss = self.criterion(output, labels)
            else:
                loss = self.criterion(output, labels.argmax(1))

            # store results
            meters['loss'].update(loss.item(), image.size(0))
            for key in self.criterion.losses:
                meters[key].update(self.criterion.losses[key].item(), image.size(0))

            for key in output:
                if 'logits' in key and not self.config.model.get('upt_config', {}).get('v3'):
                    if len(output[key].size()) == 3:
                        acc = (output[key].mean(dim=0).argmax(1) == labels.argmax(1)).float().mean()
                    else:
                        acc = (output[key].argmax(1) == labels.argmax(1)).float().mean()
                    if len(key.split('_')) == 1:
                        store_key = 'acc'
                    else:
                        store_key = f'acc_{key.split("_")[1]}'

                    meters[store_key].update(acc.item(), image.size(0))

        return {
            'codes': output['codes'],
            'labels': labels
        }

    def train_one_batch(self, *args, **kwargs):
        """
        Args:
            args: [data, meters]
            kwargs: {'ep': current epoch, 'bidx': current batch index}
        """
        data, meters = args
        # clear gradient
        self.optimizer.zero_grad()

        # print(torch.cuda.memory_summary())
        # prints currently alive Tensors and Variables

        data, output = self.compute_features_one_batch(data)
        image, labels, index = data
        if self.config.dataset.multiclass:
            loss = self.criterion(output, labels)
        else:
            loss = self.criterion(output, labels.argmax(1))

        # backward and update
        loss.backward()
        self.optimizer.step()

        # store results
        meters['loss'].update(loss.item(), image.size(0))
        for key in self.criterion.losses:
            meters[key].update(self.criterion.losses[key].item(), image.size(0))

        for key in output:
            if 'logits' in key and not self.config.model.get('upt_config', {}).get('v3'):
                if len(output[key].size()) == 3:
                    acc = (output[key].mean(dim=0).argmax(1) == labels.argmax(1)).float().mean()
                else:
                    acc = (output[key].argmax(1) == labels.argmax(1)).float().mean()
                if len(key.split('_')) == 1:
                    store_key = 'acc'
                else:
                    store_key = f'acc_{key.split("_")[1]}'

                meters[store_key].update(acc.item(), image.size(0))

        if self.config.bypass_oom_error:
            # this make training super slow but solve the unknown OOM error
            # (the tensor is accumulating somewhere inside the clip vision model)
            gc.collect()
            torch.cuda.empty_cache()
