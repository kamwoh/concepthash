import torch

from trainers.base import BaseTrainer


class A2NetCETrainer(BaseTrainer):

    def inference_one_batch(self, *args, **kwargs):
        device = self.device

        data, meters = args
        image, labels, index = data

        image, labels = image.to(device), labels.to(device)

        with torch.no_grad():
            codes, codes_tanh, logits, all_x, rec_all_x = self.model(image)
            loss = self.criterion(codes, codes_tanh, logits, all_x, rec_all_x, labels.argmax(1))

            # store results
            meters['loss'].update(loss.item(), image.size(0))
            for key in self.criterion.losses:
                meters[key].update(self.criterion.losses[key].item(), image.size(0))

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

        codes, codes_tanh, logits, all_x, rec_all_x = self.model(image)
        loss = self.criterion(codes, codes_tanh, logits, all_x, rec_all_x, labels.argmax(1))

        # backward and update
        loss.backward()
        self.optimizer.step()

        # store results
        meters['loss'].update(loss.item(), image.size(0))
        for key in self.criterion.losses:
            meters[key].update(self.criterion.losses[key].item(), image.size(0))
        meters['acc'].update((logits.argmax(1) == labels.argmax(1)).float().mean().item(), image.size(0))
