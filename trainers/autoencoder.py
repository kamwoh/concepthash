import hydra
import torch
import wandb
from omegaconf import DictConfig

from trainers.base_generation import GenerationTrainer
from utils import transforms
from utils.logger import wandb_log, wandb_commit


class AutoencoderTrainer(GenerationTrainer):

    def __init__(self, config: DictConfig):
        super().__init__(config)

        self.global_step = 0
        self.cached_images = {}

    def _record_wandb(self, image, recs, prefix=''):
        if prefix != '' and prefix[-1] != '/':
            prefix = prefix + '/'

        to_pil = transforms.to_pil()
        unnormalize = transforms.unnormalize_transform(self.config.dataset.norm)

        def process_image(inp):
            inp = inp.cpu().detach()
            inp = unnormalize(inp)
            inp = inp.clamp(0, 1)
            inp = to_pil(inp)
            inp = wandb.Image(inp)
            return inp

        n = 8
        inps = [process_image(inp) for inp in image[:n]]
        oups = [process_image(oup) for oup in recs[:n]]
        wandb_log({f'{prefix}reconstruction/inps': inps, f'{prefix}reconstruction/oups': oups})

    def load_optimizer_and_scheduler(self):
        assert self.model is not None
        params = [{'params': self.model.parameters()}]

        self.optimizer = hydra.utils.instantiate(self.config.optim, params)
        self.scheduler = hydra.utils.instantiate(self.config.scheduler, self.optimizer)

    def inference_one_batch(self, *args, **kwargs):
        device = self.device

        data, meters = args
        image, labels, index = data
        image, labels = image.to(device), labels.to(device)

        with torch.no_grad():
            latents, recs = self.model(image)
            loss = self.criterion(image, recs)

            # store results
            meters['loss'].update(loss.item(), image.size(0))
            for key in self.criterion.losses:
                meters[key].update(self.criterion.losses[key].item(), image.size(0))

        if self.config.wandb and kwargs['bidx'] == 0:
            self._record_wandb(image, recs, 'test')

        return {}

    def train_one_batch(self, *args, **kwargs):
        device = self.device

        data, meters = args
        image, labels, index = data
        image, labels, index = image.to(device), labels.to(device), index.to(device)

        # clear gradient
        self.optimizer.zero_grad()

        latents, recs = self.model(image)
        loss = self.criterion(image, recs)

        # backward and update
        loss.backward()

        if self.config.get('grad_clip') > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

        self.optimizer.step()

        # store results
        meters['loss'].update(loss.item(), image.size(0))
        for key in self.criterion.losses:
            meters[key].update(self.criterion.losses[key].item(), image.size(0))

        global_step_eval_interval = self.config.global_step_eval_interval
        if global_step_eval_interval != 0 and self.global_step % global_step_eval_interval == 0:
            self._record_wandb(image, recs, 'train')
            wandb_commit()

        self.global_step += 1
