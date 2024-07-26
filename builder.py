from mmcv.utils import Registry
from utils import build_from_cfg
import copy
import pandas as pd



TRAINERS = Registry('trainer')
def build_trainer(cfg, default_args=None):
    cp_cfg = dict(cfg.trainer)
    trainer = build_from_cfg(cp_cfg, TRAINERS, default_args)
    return trainer

AGENTS = Registry('agent')
def build_agent(cfg, default_args = None):
    cp_cfg = copy.deepcopy(cfg.agent)
    agent = build_from_cfg(cp_cfg, AGENTS, default_args)
    return agent

ENVIRONMENTS = Registry('environment')
def build_environment(cfg, default_args=None):
    cp_cfg = copy.deepcopy(cfg.environment)
    environment = build_from_cfg(cp_cfg, ENVIRONMENTS, default_args)
    return environment

DATASETS = Registry('dataset')
def build_dataset(cfg, default_args=None):

    cp_cfg = copy.deepcopy(cfg.data)
    dataset = build_from_cfg(cp_cfg, DATASETS, default_args)
    return dataset


# def build_dataset(cfg, default_args=None):
#     cp_cfg = copy.deepcopy(cfg.data)
#
#     train_data = pd.read_csv(cp_cfg['train_path'])
#     valid_data = pd.read_csv(cp_cfg['valid_path'])
#     test_data = pd.read_csv(cp_cfg['test_path'])
#
#     datasets = {
#         'train': build_from_cfg(dict(type='PortfolioManagementDataset', data=train_data, timesteps=cp_cfg['timesteps']),
#                                 DATASETS, default_args),
#         'valid': build_from_cfg(dict(type='PortfolioManagementDataset', data=valid_data, timesteps=cp_cfg['timesteps']),
#                                 DATASETS, default_args),
#         'test': build_from_cfg(dict(type='PortfolioManagementDataset', data=test_data, timesteps=cp_cfg['timesteps']),
#                                DATASETS, default_args)
#     }
#
#     return datasets

NETS = Registry('net')
def build_net(cfg, default_args=None):
    cp_cfg = copy.deepcopy(cfg)
    net = build_from_cfg(cp_cfg, NETS, default_args)
    return net

OPTIMIZERS = Registry('optimizer')

def build_optimizer(cfg, default_args=None):
    cp_cfg = copy.deepcopy(cfg.optimizer)
    optimizer = build_from_cfg(cp_cfg, OPTIMIZERS, default_args)
    return optimizer

LOSSES = Registry('loss')

def build_loss(cfg, default_args=None):
    cp_cfg = copy.deepcopy(cfg.loss)
    loss = build_from_cfg(cp_cfg, LOSSES, default_args)
    return loss

TRANSITIONS = Registry('transition')

def build_transition(cfg, default_args=None):
    cp_cfg = copy.deepcopy(cfg.transition)
    transition = build_from_cfg(cp_cfg, TRANSITIONS, default_args)
    return transition