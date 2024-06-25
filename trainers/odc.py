import faiss
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from trainers.base import BaseTrainer
from utils import io


class Memory:
    def __init__(self, momentum=0.1, nclusters=10, in_dim=64):
        super().__init__()
        self.memory_features = None  # torch.Tensor
        self.memory_labels = None
        self.momentum = momentum
        self.centroids = torch.zeros(nclusters, in_dim)  # torch.Tensor

    def init(self, memory_features, memory_labels):
        self.memory_features = memory_features
        self.memory_labels = memory_labels

        self.update_centroids()

    def update(self, index, new_features):
        curr_features = self.memory_features.data[index, ...]

        difference = curr_features - F.normalize(new_features, p=2, dim=-1)
        difference.mul_(self.momentum)
        curr_features.sub_(difference)

        # (B, 1, D) - (1, C, D) = (B, C)
        dist_to_centroids = ((curr_features.unsqueeze(1) - self.centroids.unsqueeze(0)) ** 2).sum(dim=2)  # (B, C)
        new_labels = dist_to_centroids.argmin(dim=1)  # (B,)
        self.memory_labels.data[index].copy_(new_labels)

    def update_centroids(self):
        memory_features = self.memory_features
        memory_labels = self.memory_labels

        for c in range(self.centroids.size(0)):
            cmask = memory_labels == c
            cfeatures = memory_features[cmask].mean(dim=0)
            self.centroids.data[c].copy_(cfeatures)

    def save(self, fn):
        pass

    def load(self, fn):
        pass


class ODCTrainer(BaseTrainer):
    """
    online deep clustering
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)

        self.memory = Memory(momentum=0.5,
                             nclusters=self.config.model.nclusters,
                             in_dim=self.config.model.head_dim)
        self.global_step = 0
        self.update_interval = config.method_params.update_interval

    def save_training_state(self, fn):
        optimsd = self.optimizer.state_dict()
        schedulersd = self.scheduler.state_dict()
        criterionsd = self.criterion.state_dict()
        io.fast_save({'optim': optimsd,
                      'scheduler': schedulersd,
                      'criterion': criterionsd}, fn)

    def load_training_state(self, fn):
        sd = torch.load(fn, map_location='cpu')
        self.optimizer.load_state_dict(sd['optim'])
        self.scheduler.load_state_dict(sd['scheduler'])
        self.criterion.load_state_dict(sd['criterion'])

    def parse_model_output(self, output):
        logits, codes, feats = output
        return {
            'logits': logits,
            'codes': codes,
            'feats': feats
        }

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

        if self.criterion is not None:
            self.criterion = self.criterion.to(device)

    def prepare_before_first_epoch(self):
        self.to_device(self.device)
        results = self.compute_features('train_no_shuffle')
        init_features = results['codes'].cpu()
        k = self.config.model.nclusters

        kmeans = faiss.Kmeans(init_features.shape[1], k, niter=50, verbose=True)
        kmeans.train(init_features.numpy())

        _, pseudo_labels = kmeans.index.search(init_features.numpy(), 1)
        pseudo_labels = pseudo_labels.ravel()

        self.memory.init(memory_features=init_features,
                         memory_labels=torch.tensor(pseudo_labels))
        self.criterion.set_reweight(self.memory.memory_labels)
        # self.memory.centroids = self.memory.centroids.to(self.device)

    def inference_one_batch(self, *args, **kwargs):
        data, meters = args

        with torch.no_grad():
            data, output = self.compute_features_one_batch(data)

            image, labels, index = data
            logits, codes, feats = output['logits'], output['codes'], output['feats']

        return {
            'codes': feats,
            'labels': labels
        }

    def train_one_batch(self, *args, **kwargs):
        """
        Args:
            args: [data, meters]
            kwargs: {'ep': current epoch, 'bidx': current batch index}
        """
        device = self.device

        data, meters = args
        image, labels, index = data
        image, labels = image.to(device), labels.to(device)
        pseudo_labels = self.memory.memory_labels[index].clone()

        # clear gradient
        self.optimizer.zero_grad()

        logits, codes, feats = self.model(image)
        loss = self.criterion(logits, codes, pseudo_labels.to(device))

        # backward and update
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            self.memory.update(index, codes.cpu())

            update_now = self.global_step % self.update_interval == 0
            if update_now:
                self.memory.update_centroids()
                self.criterion.set_reweight(self.memory.memory_labels)

        self.global_step += 1

        # store results
        meters['loss'].update(loss.item(), image.size(0))
        for key in self.criterion.losses:
            meters[key].update(self.criterion.losses[key].item(), image.size(0))
