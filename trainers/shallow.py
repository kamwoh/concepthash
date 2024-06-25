import logging
from collections import defaultdict

import hydra
import numpy as np
import torch
from tqdm import tqdm

from trainers.base import BaseTrainer
from utils import io
from utils.misc import AverageMeter


class BaseShallowTrainer(BaseTrainer):

    def is_ready_for_training(self):
        r = True

        for item in [self.dataset,
                     self.dataloader,
                     self.model]:
            r = r and item is not None

        return r

    def is_ready_for_inference(self):
        r = True

        for item in [self.dataset,
                     self.dataloader,
                     self.model]:
            r = r and item is not None

        return r

    def load_optimizer_and_scheduler(self):
        pass

    def save_model_state(self, fn):
        modelsd = self.criterion.state_dict()
        try:
            modelsd = {k: v.clone().cpu() for k, v in modelsd.items()}
        except:
            modelsd = {k: v for k, v in modelsd.items()}
        io.fast_save(modelsd, fn)

    def load_model_state(self, fn):
        modelsd = torch.load(fn, map_location='cpu')
        self.criterion.load_state_dict(modelsd)

    def save_training_state(self, fn):
        pass

    def load_training_state(self, fn):
        pass

    def inference_one_batch(self, *args, **kwargs):
        device = self.device

        data, meters = args
        image, labels, index = data
        image, labels = image.to(device), labels.to(device)

        with torch.no_grad():
            codes = self.criterion(self.model(image))

        return {
            'codes': codes,
            'labels': labels
        }

    def inference_one_epoch(self, datakey='test', return_codes=False, **kwargs):
        assert self.is_ready_for_inference()

        self.model.eval()
        self.criterion.eval()
        meters = defaultdict(AverageMeter)

        ret = defaultdict(list)

        with tqdm(self.dataloader[datakey], bar_format='{l_bar}{bar:10}{r_bar}') as tepoch:
            for i, data in enumerate(tepoch):
                output = self.inference_one_batch(data, meters, bidx=i, **kwargs)
                tepoch.set_postfix({k: v.avg for k, v in meters.items()})

                if return_codes:
                    for key in output:
                        ret[key].append(output[key])

        if return_codes:
            res = {}
            for key in ret:
                if isinstance(ret[key][0], torch.Tensor):
                    res[key] = torch.cat(ret[key])
                else:
                    res[key] = np.concatenate(ret[key])
            return meters, res

        return meters

    def train_one_batch(self, *args, **kwargs):
        pass

    def train_one_epoch(self, *args, **kwargs):
        assert self.is_ready_for_training()

        self.model.eval()  # shallow no need set to train mode
        self.criterion.train()

        meters = defaultdict(AverageMeter)

        with torch.no_grad():
            train_data = []
            with tqdm(self.dataloader['train'], bar_format='{l_bar}{bar:10}{r_bar}') as tepoch:
                for x in tepoch:
                    train_data.append(self.model(x[0].to(self.device)).cpu())
            train_data = torch.cat(train_data)
            train_out, quan_error = self.criterion(train_data)
        meters['quan'].update(quan_error.item())

        return meters

    def load_model(self):
        logging.info('Creating Model')
        model = hydra.utils.instantiate(self.config.backbone)
        self.model = model
