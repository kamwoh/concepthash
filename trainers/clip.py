import logging
import os

import hydra
import torch
from omegaconf import DictConfig
from transformers import CLIPProcessor

import engine
from trainers.base import BaseTrainer


class FinetuneCLIPTrainer(BaseTrainer):
    """
    same as BaseTrainer
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)

        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        self.processor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')

    def load_dataset(self, load_db=True):
        logging.info('Creating Datasets')
        train_dataset = hydra.utils.instantiate(self.config.dataset.train_dataset)
        test_dataset = hydra.utils.instantiate(self.config.dataset.test_dataset)

        logging.info(f'Number of Train data: {len(train_dataset)}')
        logging.info(f'Number of Query data: {len(test_dataset)}')

        self.dataset = {
            'train': train_dataset,
            'test': test_dataset,
        }

    def load_dataloader(self):
        assert self.dataset is not None
        bs = self.config.batch_size
        train_loader = engine.dataloader(self.dataset['train'], bs, shuffle=True, drop_last=True)
        test_loader = engine.dataloader(self.dataset['test'], self.config.batch_size, shuffle=False, drop_last=False)

        self.dataloader = {
            'train': train_loader,
            'test': test_loader,
        }

    def load_optimizer_and_scheduler(self):
        assert self.model is not None

        params = [{'params': self.model.get_training_modules().parameters()}]

        self.model.model.requires_grad_(False)
        self.model.model.logit_scale.requires_grad_(True)

        self.optimizer = hydra.utils.instantiate(self.config.optim, params)
        self.scheduler = hydra.utils.instantiate(self.config.scheduler, self.optimizer)

    def parse_model_output(self, output):
        logits = output
        return {
            'logits': logits
        }

    def compute_features_one_batch(self, data):
        device = self.device

        image, text, index = data
        image = image.to(device)

        text_inputs = self.processor(text=text, return_tensors="pt", padding=True)
        input_ids = text_inputs['input_ids'].to(device)
        attention_mask = text_inputs['attention_mask'].to(device)

        output = self.model(input_ids, image, attention_mask)

        return data, self.parse_model_output(output)

    def inference_one_batch(self, *args, **kwargs):
        data, meters = args

        with torch.no_grad():
            data, output = self.compute_features_one_batch(data)

            image, labels, index = data
            logits = output['logits']
            loss = self.criterion(logits)

            # store results
            meters['loss'].update(loss.item(), image.size(0))
            for key in self.criterion.losses:
                meters[key].update(self.criterion.losses[key].item(), image.size(0))

        return {}

    def train_one_batch(self, *args, **kwargs):
        """
        Args:
            args: [data, meters]
            kwargs: {'ep': current epoch, 'bidx': current batch index}
        """
        data, meters = args
        # clear gradient
        self.optimizer.zero_grad()

        data, output = self.compute_features_one_batch(data)
        logits = output['logits']
        loss = self.criterion(logits)

        # backward and update
        loss.backward()
        self.optimizer.step()

        # store results
        meters['loss'].update(loss.item(), data[0].size(0))
        for key in self.criterion.losses:
            meters[key].update(self.criterion.losses[key].item(), data[0].size(0))
