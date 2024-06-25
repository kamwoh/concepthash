import torch

from trainers.base import BaseTrainer


class SEMICONCETrainer(BaseTrainer):

    def inference_one_batch(self, *args, **kwargs):
        device = self.device

        data, meters = args
        image, labels, index = data

        image, labels = image.to(device), labels.to(device)

        with torch.no_grad():
            codes, local_features, logits = self.model(image)

        return {
            'codes': codes,
            'labels': labels
        }

    def train_one_batch(self, *args, **kwargs):
        device = self.device

        data, meters = args
        image, labels, index = data
        image, labels = image.to(device), labels.to(device)

        # clear gradient
        self.optimizer.zero_grad()

        codes, local_features, logits = self.model(image)
        loss = self.criterion(codes, logits, labels.argmax(1))

        # backward and update
        loss.backward()
        self.optimizer.step()

        # store results
        meters['loss'].update(loss.item(), image.size(0))
        for key in self.criterion.losses:
            meters[key].update(self.criterion.losses[key].item(), image.size(0))
