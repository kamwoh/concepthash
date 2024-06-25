import logging
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.linalg import hadamard
from tqdm import tqdm

from trainers.base import BaseTrainer
from utils import io
from utils.hashing import get_hamm_dist
from utils.metrics import calculate_accuracy_hamm_dist, calculate_accuracy


def get_adaptive_scale(nclass):
    return math.sqrt(2) * math.log(nclass - 1)


def get_hadamard(nclass, nbit, fast=True):
    """
    copy from CSQ
    """
    H_K = hadamard(nbit)
    H_2K = np.concatenate((H_K, -H_K), 0)
    hash_targets = torch.from_numpy(H_2K[:nclass]).float()

    if H_2K.shape[0] < nclass:
        hash_targets.resize_(nclass, nbit)
        for k in range(20):
            for index in range(H_2K.shape[0], nclass):
                ones = torch.ones(nbit)
                # Bernouli distribution
                sa = random.sample(list(range(nbit)), nbit // 2)
                ones[sa] = -1
                hash_targets[index] = ones

            if fast:
                return hash_targets

            # to find average/min  pairwise distance
            c = []
            # print()
            # print(n_class)
            TF = (hash_targets.view(1, -1, nbit) != hash_targets.view(-1, 1, nbit)).sum(dim=2).float()
            TF_mask = torch.triu(torch.ones_like(TF), 1).bool()
            c = TF[TF_mask]

            # choose min(c) in the range of K/4 to K/3
            # see in https://github.com/yuanli2333/Hadamard-Matrix-for-hashing/issues/1
            # but it is hard when bit is  small
            if c.min() > nbit / 4 and c.mean() >= nbit / 2:
                print(c.min(), c.mean())
                break

    return hash_targets


def get_codebook(codebook_method, nclass, nbit, **kwargs):
    assert codebook_method in ['N', 'B', 'H', 'O', 'L']

    if codebook_method == 'N':  # normal
        codebook = torch.randn(nclass, nbit)
    elif codebook_method == 'B':  # bernoulli
        prob = torch.ones(nclass, nbit) * 0.5
        codebook = torch.bernoulli(prob) * 2. - 1.
    elif codebook_method == 'H':  # hadamard
        codebook = get_hadamard(nclass, nbit)
    elif codebook_method == 'O':  # O: optim
        codebook = optimize_codebook(nclass, nbit)
    else:
        codebook = language_guided_codebook(nbit=nbit, **kwargs)

    return codebook.sign()


class InducedEncoder(nn.Module):
    def __init__(self, n=1000, d=768):
        super().__init__()

        self.queries = nn.Parameter(torch.randn(n, d))

    def forward(self, x):
        attn = F.normalize(self.queries, p=2, dim=-1) @ F.normalize(x, p=2, dim=-1).t()  # (n, x)
        attn = F.normalize(attn, p=2, dim=-1)

        return attn @ attn.t()  # (n, n)


def language_guided_codebook(class_name_path, nbit,
                             model_id="openai/clip-vit-large-patch14",
                             binary_method='itq',
                             **kwargs):
    from transformers import CLIPModel, CLIPProcessor
    from models.loss.itq import ITQLoss
    from sklearn.decomposition import PCA

    text_model = CLIPModel.from_pretrained(model_id).text_model.eval()
    processor = CLIPProcessor.from_pretrained(model_id)

    class_names = open(class_name_path).readlines()
    class_names = [c.replace("_", " ").strip() for c in class_names]  # replace all _
    nclass = len(class_names)

    prompt_prefix = kwargs.get('prompt_prefix', 'a photo of a ')
    prompt_postfix = kwargs.get('prompt_postfix', '')
    if len(prompt_prefix) != 0 and prompt_prefix[-1] != ' ':
        prompt_prefix = prompt_prefix + ' '

    prompts = [prompt_prefix + name + prompt_postfix for name in class_names]
    tokenized_prompts = processor(text=prompts, return_tensors="pt", padding=True, truncation=True)['input_ids']
    device = torch.device('cuda')

    logging.info(f'sample prompt: {prompts[0]}')
    logging.info('compute text embeddings')

    with torch.no_grad():
        text_model = text_model.to(device)
        text_batch_size = min(nclass, 100)
        embedding = []

        pbar = tqdm(range(len(tokenized_prompts) // text_batch_size + 1),
                    desc='Query',
                    bar_format='{l_bar}{bar:10}{r_bar}')

        for batch_idx in pbar:
            curr_start = batch_idx * text_batch_size
            curr_end = min(curr_start + text_batch_size, len(tokenized_prompts))
            if curr_start >= nclass:
                break
            embedding.append(text_model(tokenized_prompts[curr_start:curr_end].to(device)).pooler_output.cpu())
        embedding = torch.cat(embedding, dim=0)
        logging.info(f"embedding: {embedding.size()}")

    # manual clear memory
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # in case tokenizer warning
    del processor
    del text_model

    if not kwargs.get('quantized', True):
        return embedding

    if binary_method == 'itq':
        itq_loss = ITQLoss(nbit, 100)
        binary_target, quan_loss = itq_loss(embedding)
    elif binary_method == 'pca':
        pca = PCA(nbit)
        binary_target = torch.from_numpy(pca.fit_transform(embedding.numpy()))
        quan_loss = (1 - F.cosine_similarity(binary_target, binary_target.sign())).mean()
    elif binary_method == 'pcaw':
        pca = PCA(nbit, whiten=True)
        binary_target = torch.from_numpy(pca.fit_transform(embedding.numpy()))
        quan_loss = (1 - F.cosine_similarity(binary_target, binary_target.sign())).mean()
    elif binary_method == 'rand':
        rand_idx = torch.randperm(embedding.size(1))[:nbit]
        binary_target = embedding[:, rand_idx]
        quan_loss = (1 - F.cosine_similarity(binary_target, binary_target.sign(), dim=-1)).mean()
        binary_target = binary_target.sign()
    else:  # ae
        if 'non' in binary_method:
            encoder = nn.Sequential(
                nn.Linear(embedding.size(1), embedding.size(1)),
                nn.GELU(),
                nn.Linear(embedding.size(1), nbit),
            )
            decoder = nn.Sequential(
                nn.Linear(nbit, embedding.size(1)),
                nn.GELU(),
                nn.Linear(embedding.size(1), embedding.size(1))
            )
        else:
            encoder = nn.Linear(embedding.size(1), nbit)
            decoder = nn.Linear(nbit, embedding.size(1))

        if 'induced_' in binary_method:
            binary_method = binary_method.replace('induced_', '')
            induced_encoder = InducedEncoder(1000, embedding.size(1))
        else:
            induced_encoder = nn.Identity()

        encoder = encoder.to(device)
        decoder = decoder.to(device)
        induced_encoder = induced_encoder.to(device)
        embedding = embedding.to(device)

        criterion = nn.MSELoss(reduction='none')
        quan_criterion = nn.CosineSimilarity(dim=-1)
        optimizer = optim.Adam(list(encoder.parameters()) +
                               list(decoder.parameters()) +
                               list(induced_encoder.parameters()),
                               lr=0.0001)
        pbar = tqdm(range(kwargs.get('ae_iters', 10000)),
                    desc='Query',
                    bar_format='{l_bar}{bar:10}{r_bar}')

        binary_method = binary_method.replace('non', '')  # nonae -> ae

        if binary_method == 'ae_cossim':
            if isinstance(induced_encoder, nn.Identity):
                l2_embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
                cossim = l2_embedding @ l2_embedding.t()
            else:
                cossim = None
        elif binary_method == 'ae_norm_cossim':
            if isinstance(induced_encoder, nn.Identity):
                l2_embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
                cossim = l2_embedding @ l2_embedding.t()
            else:
                cossim = induced_encoder(embedding)
            cossim = (cossim - cossim.min()) / (cossim.max() - cossim.min()) * 2. - 1
        else:
            cossim = torch.eye(nclass)

        t = kwargs.get('t', 1)
        identity_scale = kwargs.get('identity_scale', 1)

        for _ in pbar:
            optimizer.zero_grad()

            binary_target = encoder(embedding)
            l2_binary_target = F.normalize(binary_target, p=2, dim=-1)
            rec_embedding = decoder(binary_target)

            loss = criterion(embedding, rec_embedding).mean(dim=-1)

            if isinstance(induced_encoder, nn.Identity):
                identity_loss = (cossim - l2_binary_target @ l2_binary_target.t()).pow(2).mean()
            else:
                cossim = induced_encoder(embedding)
                query_targets = encoder(induced_encoder.queries)
                attn = F.normalize(query_targets, p=2, dim=-1) @ l2_binary_target.t()
                attn = F.normalize(attn, p=2, dim=-1)
                binary_cossim = attn @ attn.t()

                identity_loss = (cossim - binary_cossim).pow(2).mean()

            quan_loss = (1 - quan_criterion(binary_target, binary_target.sign()))
            total_loss = loss.mean() + (torch.exp(-loss / t) * quan_loss).mean() + identity_loss * identity_scale
            total_loss.backward()

            pbar.set_postfix({'total_loss': total_loss.item(),
                              'loss': loss.mean().item(),
                              'quan_loss': quan_loss.mean().item(),
                              'identity_loss': identity_loss.item()})

            optimizer.step()

        quan_loss = quan_loss.mean()

    binary_target = binary_target.detach().sign()
    identity = 0
    if nclass < 1000:
        identity = (torch.eye(nclass) - (binary_target @ binary_target.t()) / nbit).pow(2).mean()
    logging.info(f"binary target: {binary_target.size()}; quan loss: {quan_loss:.2f}; identity: {identity:.2f}")

    return binary_target.cpu()


def get_hd(a, b):
    return 0.5 * (a.size(0) - a @ b.t()) / a.size(0)


def optimize_codebook(nclass, nbit, maxtries=10000, initdist=0.61, mindist=0.2, reducedist=0.05):
    """
    brute force to find centroid with furthest distance
    :param nclass:
    :param nbit:
    :param maxtries:
    :param initdist:
    :param mindist:
    :param reducedist:
    :return:
    """
    codebook = torch.zeros(nclass, nbit)
    i = 0
    count = 0
    currdist = initdist
    while i < nclass:
        print(i, end='\r')
        c = torch.randn(nbit).sign()
        nobreak = True
        for j in range(i):
            if get_hd(c, codebook[j]) < currdist:
                i -= 1
                nobreak = False
                break
        if nobreak:
            codebook[i] = c
        else:
            count += 1

        if count >= maxtries:
            count = 0
            currdist -= reducedist
            print('reduce', currdist, i)
            if currdist < mindist:
                raise ValueError('cannot find')

        i += 1
    codebook = codebook[torch.randperm(nclass)]
    return codebook


class OrthoHashTrainer(BaseTrainer):
    def __init__(self, config):
        super(OrthoHashTrainer, self).__init__(config)

        self.codebook = None

    def load_model(self):
        super(OrthoHashTrainer, self).load_model()
        self.codebook = self.model.codebook

    def save_codebook(self, fn):
        io.fast_save(self.codebook, fn)

    def load_codebook(self, fn):
        self.codebook = torch.load(fn)

    def load_for_inference(self, logdir):
        self.load_codebook(f'{logdir}/outputs/codebook.pth')

    def to_device(self, device=None):
        super(OrthoHashTrainer, self).to_device(device)

        if device is None:
            device = self.device

        self.codebook = self.codebook.to(device)

    def is_ready_for_inference(self):
        ready = super(OrthoHashTrainer, self).is_ready_for_inference()
        ready = ready and self.codebook is not None
        return ready

    def is_ready_for_training(self):
        ready = super(OrthoHashTrainer, self).is_ready_for_training()
        ready = ready and self.codebook is not None
        return ready

    def save_before_training(self, logdir):
        super(OrthoHashTrainer, self).save_before_training(logdir)
        self.save_codebook(f'{logdir}/outputs/codebook.pth')

    def inference_one_batch(self, *args, **kwargs):
        device = self.device

        data, meters = args

        with torch.no_grad():
            data, output = self.compute_features_one_batch(data)
            image, labels, index = data
            logits, codes = output['logits'], output['codes']

            loss = self.criterion(logits, codes, labels)
            acc = calculate_accuracy(logits, labels)

            hamm_dist = get_hamm_dist(codes, self.codebook, normalize=True)
            hacc = calculate_accuracy_hamm_dist(hamm_dist, labels)

            # store results
            meters['loss'].update(loss.item(), image.size(0))
            for key in self.criterion.losses:
                meters[key].update(self.criterion.losses[key].item(), image.size(0))
            meters['acc'].update(acc.item(), image.size(0))
            meters['hacc'].update(hacc.item(), image.size(0))

        return {
            'codes': codes,
            'labels': labels
        }

    def train_one_batch(self, *args, **kwargs):
        device = self.device

        data, meters = args

        # clear gradient
        self.optimizer.zero_grad()

        data, output = self.compute_features_one_batch(data)
        image, labels, index = data
        logits, codes = output['logits'], output['codes']

        loss = self.criterion(logits, codes, labels)

        # backward and update
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            acc = calculate_accuracy(logits, labels)
            hamm_dist = get_hamm_dist(codes, self.codebook, normalize=True)
            hacc = calculate_accuracy_hamm_dist(hamm_dist, labels)

        # store results
        meters['loss'].update(loss.item(), image.size(0))
        for key in self.criterion.losses:
            meters[key].update(self.criterion.losses[key].item(), image.size(0))
        meters['acc'].update(acc.item(), image.size(0))
        meters['hacc'].update(hacc.item(), image.size(0))


class OrthoHashWithBCSTrainer(OrthoHashTrainer):
    def parse_model_output(self, output):
        logits, logits_2, codes = output
        return {
            'logits': logits,
            'logits_2': logits_2,
            'codes': codes,
        }

    def inference_one_batch(self, *args, **kwargs):
        device = self.device

        data, meters = args

        with torch.no_grad():
            data, output = self.compute_features_one_batch(data)
            image, labels, index = data
            logits, codes = output['logits'], output['codes']

            loss = self.criterion(logits, codes, labels)
            acc = calculate_accuracy(logits, labels)

            hamm_dist = get_hamm_dist(codes, self.codebook, normalize=True)
            hacc = calculate_accuracy_hamm_dist(hamm_dist, labels)

            # store results
            meters['loss'].update(loss.item(), image.size(0))
            for key in self.criterion.losses:
                meters[key].update(self.criterion.losses[key].item(), image.size(0))
            meters['acc'].update(acc.item(), image.size(0))
            meters['hacc'].update(hacc.item(), image.size(0))

        return {
            'codes': codes,
            'labels': labels
        }

    def train_one_batch(self, *args, **kwargs):
        device = self.device

        data, meters = args

        # clear gradient
        self.optimizer.zero_grad()

        data, output = self.compute_features_one_batch(data)
        image, labels, index = data
        logits, logits_2, codes = output['logits'], output['logits_2'], output['codes']

        loss = self.criterion(logits, codes, labels)

        # backward and update
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            acc = calculate_accuracy(logits, labels)
            hamm_dist = get_hamm_dist(codes, self.codebook, normalize=True)
            hacc = calculate_accuracy_hamm_dist(hamm_dist, labels)

        # store results
        meters['loss'].update(loss.item(), image.size(0))
        for key in self.criterion.losses:
            meters[key].update(self.criterion.losses[key].item(), image.size(0))
        meters['acc'].update(acc.item(), image.size(0))
        meters['hacc'].update(hacc.item(), image.size(0))
