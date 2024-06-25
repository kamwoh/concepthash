import os
import uuid

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

import engine
from experiments.test_hashing import RetrievalEvaluation
from experiments.train_helper import RetrievalExperiment
from experiments.train_no_eval import GeneralExperiment


@hydra.main(config_path="configs/", config_name="train.yaml", version_base=None)
def main(config: DictConfig):
    ##### training #####
    if config.exp == 'general':
        experiment = GeneralExperiment(config)  # this will skip mAP evaluation
    elif config.exp == 'hashing':
        experiment = RetrievalExperiment(config)
    elif config.exp == 'validation':
        load_config = OmegaConf.load(config.logdir + '/config.yaml')
        load_config.dataset = config.dataset
        load_config.data_dir = config.data_dir
        load_config.work_dir = config.work_dir
        load_config.eval_logdir = config.eval_logdir
        load_config.logdir = f'{config.work_dir}/{config.logdir}'
        load_config.R = config.R
        load_config.PRs = config.PRs
        load_config.use_last = config.use_last
        load_config.compute_mAP = config.compute_mAP
        load_config.ternary_threshold = config.ternary_threshold
        load_config.dist_metric = config.dist_metric
        load_config.batch_size = config.batch_size
        load_config.save_code = config.save_code
        load_config.wandb = False
        load_config.sub_code_eval = config.sub_code_eval
        load_config.sub_code_eval_setting = config.sub_code_eval_setting
        load_config.zero_mean_eval = config.zero_mean_eval
        load_config.test_as_database = config.test_as_database
        experiment = RetrievalEvaluation(load_config)
    elif config.exp in ['descriptor', 'extract']:
        experiment = RetrievalEvaluation(config)
    else:
        raise ValueError(f'Unknown exp value: "{config.exp}"')

    experiment.main()


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.backends.cudnn.benchmark = False
    # ROOTDIR = os.environ.get('ROOTDIR', '.')
    os.environ['HYDRA_FULL_ERROR'] = '1'
    os.environ['OC_CAUSE'] = '1'

    engine.default_workers = min(16, os.cpu_count())

    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver("uuid4", lambda _: str(uuid.uuid4())[-4:])

    # if evaluation:  --config-path val.yaml

    main()
