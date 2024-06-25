import torch

from trainers.base import BaseTrainer
from utils.metrics import calculate_accuracy


class DELGTrainer(BaseTrainer):

    def __init__(self, config):
        super(DELGTrainer, self).__init__(config)

        self.multiclass = self.config.dataset.multiclass
        self.onehot = self.config.dataset_name not in ['gldv2', 'gldv2_delg']

    def train_one_batch(self, *args, **kwargs):
        device = self.device

        data, meters = args
        image, labels, index = data
        image, labels = image.to(device), labels.to(device)

        # clear gradient
        self.optimizer.zero_grad()

        outputs = self.model(image)
        loss = self.criterion(outputs, labels, onehot=self.onehot)

        (global_feat, local_feat, local_feat_reduced,
         attn_probs, global_logits, local_logits, layer3, rec_layer3) = outputs

        # backward and update
        loss.backward()
        self.optimizer.step()

        acc_global = calculate_accuracy(global_logits, labels, self.onehot, self.multiclass)
        acc_local = calculate_accuracy(local_logits, labels, self.onehot, self.multiclass)

        # store results
        meters['loss'].update(loss.item(), image.size(0))
        for key in self.criterion.losses:
            meters[key].update(self.criterion.losses[key].item(), image.size(0))
        meters['acc_g'].update(acc_global.item(), image.size(0))
        meters['acc_l'].update(acc_local.item(), image.size(0))

    def inference_one_batch(self, *args, **kwargs):
        device = self.device

        data, meters = args
        image, labels, index = data
        image, labels = image.to(device), labels.to(device)

        with torch.no_grad():
            outputs = self.model(image)
            loss = self.criterion(outputs, labels, onehot=self.onehot)

            (global_feat, local_feat, local_feat_reduced,
             attn_probs, global_logits, local_logits, layer3, rec_layer3) = outputs

            acc_global = calculate_accuracy(global_logits, labels, self.onehot, self.multiclass)
            acc_local = calculate_accuracy(local_logits, labels, self.onehot, self.multiclass)

            # store results
            meters['loss'].update(loss.item(), image.size(0))
            for key in self.criterion.losses:
                meters[key].update(self.criterion.losses[key].item(), image.size(0))
            meters['acc_g'].update(acc_global.item(), image.size(0))
            meters['acc_l'].update(acc_local.item(), image.size(0))

        if self.config['codes_for_retrieval'] == 'global':
            feat = global_feat
        else:
            feat = local_feat

        return {
            'codes': feat,
            'labels': labels
        }
