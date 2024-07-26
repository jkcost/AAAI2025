import inspect
import os
import re
import mmcv
from mmcv import Config
from mmcv.utils import Registry
import prettytable
from iopath.common.file_io import g_pathmgr as pathmgr
from math import inf
import pickle as pkl
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch.distributed as dist
from collections import namedtuple, OrderedDict
import os.path as osp
from typing import List

import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset




def build_from_cfg(cfg, registry, default_args=None):
    """Build a module from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.

    Returns:
        object: The constructed object.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
    if 'type' not in cfg:
        if default_args is None or 'type' not in default_args:
            raise KeyError(
                '`cfg` or `default_args` must contain the key "type", '
                f'but got {cfg}\n{default_args}')
    if not isinstance(registry, Registry):
        raise TypeError('registry must be an mmcv.Registry object, '
                        f'but got {type(registry)}')
    if not (isinstance(default_args, dict) or default_args is None):
        raise TypeError('default_args must be a dict or None, '
                        f'but got {type(default_args)}')
    args = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    obj_type = args.pop('type')
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(
                f'{obj_type} is not in the {registry.name} registry')
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(
            f'type must be a str or valid type, but got {type(obj_type)}')
    try:
        return obj_cls(**args)
    except Exception as e:
        # Normal TypeError does not print class name.
        raise type(e)(f'{obj_cls.__name__}: {e}')



def replace_cfg_vals(ori_cfg):
    """Replace the string "${key}" with the corresponding value.

    Replace the "${key}" with the value of ori_cfg.key in the config. And
    support replacing the chained ${key}. Such as, replace "${key0.key1}"
    with the value of cfg.key0.key1. Code is modified from `vars.py
    < https://github.com/microsoft/SoftTeacher/blob/main/ssod/utils/vars.py>`_  # noqa: E501

    Args:
        ori_cfg (mmcv.utils.config.Config):
            The origin config with "${key}" generated from a file.

    Returns:
        updated_cfg [mmcv.utils.config.Config]:
            The config with "${key}" replaced by the corresponding value.
    """

    def get_value(cfg, key):
        for k in key.split('.'):
            cfg = cfg[k]
        return cfg

    def replace_value(cfg):
        if isinstance(cfg, dict):
            return {key: replace_value(value) for key, value in cfg.items()}
        elif isinstance(cfg, list):
            return [replace_value(item) for item in cfg]
        elif isinstance(cfg, tuple):
            return tuple([replace_value(item) for item in cfg])
        elif isinstance(cfg, str):
            # the format of string cfg may be:
            # 1) "${key}", which will be replaced with cfg.key directly
            # 2) "xxx${key}xxx" or "xxx${key1}xxx${key2}xxx",
            # which will be replaced with the string of the cfg.key
            keys = pattern_key.findall(cfg)
            values = [get_value(ori_cfg, key[2:-1]) for key in keys]
            if len(keys) == 1 and keys[0] == cfg:
                # the format of string cfg is "${key}"
                cfg = values[0]
            else:
                for key, value in zip(keys, values):
                    # the format of string cfg is
                    # "xxx${key}xxx" or "xxx${key1}xxx${key2}xxx"
                    assert not isinstance(value, (dict, list, tuple)), \
                        f'for the format of string cfg is ' \
                        f"'xxxxx${key}xxxxx' or 'xxx${key}xxx${key}xxx', " \
                        f"the type of the value of '${key}' " \
                        f'can not be dict, list, or tuple' \
                        f'but you input {type(value)} in {cfg}'
                    cfg = cfg.replace(key, str(value))
            return cfg
        else:
            return cfg

    # the pattern of string "${key}"
    pattern_key = re.compile(r'\$\{[a-zA-Z\d_.]*\}')
    # the type of ori_cfg._cfg_dict is mmcv.utils.config.ConfigDict
    updated_cfg = Config(
        replace_value(ori_cfg._cfg_dict), filename=ori_cfg.filename)
    # replace the model with model_wrapper
    if updated_cfg.get('model_wrapper', None) is not None:
        updated_cfg.model = updated_cfg.model_wrapper
        updated_cfg.pop('model_wrapper')
    return updated_cfg

def get_attr(args, key=None, default_value=None):
    if isinstance(args, dict):
        return args[key] if key in args else default_value
    elif isinstance(args, object):
        return getattr(args, key, default_value) if key is not None else default_value


def print_metrics(stats):
    table = prettytable.PrettyTable()
    # table.add_row(['' for _ in range(len(stats))])
    for key, value in stats.items():
        table.add_column(key, value)
    return table

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0

def save_on_master(state, path):
    if is_main_process():
        #print(f"save path {path}")
        with pathmgr.open(path, "wb") as f:
            torch.save(state, f)


def save_model(output_dir,
               epoch,
               save):
    checkpoint_path = os.path.join(output_dir,"checkpoint-{:05d}.pth".format(epoch))

    to_save = dict()
    for name, model in save["models"].items():
        if model:
            to_save[name] = model.state_dict()

    for name, optimizer in save["optimizers"].items():
        if optimizer:
            to_save[name] = optimizer.state_dict()
    to_save["epoch"] = epoch

    save_on_master(to_save, checkpoint_path)
    return checkpoint_path


def save_best_model(output_dir,
                    save,
                    epoch = None):
    checkpoint_path = os.path.join(output_dir, "best.pth")

    to_save = dict()
    for name, model in save["models"].items():
        if model:
            to_save[name] = model.state_dict()

    for name, optimizer in save["optimizers"].items():
        if optimizer:
            to_save[name] = optimizer.state_dict()
    to_save["epoch"] = epoch

    save_on_master(to_save, checkpoint_path)
    return checkpoint_path

def save_best_model_trial(output_dir,
               trial_number,
               save,
               epoch = None):
    checkpoint_path = os.path.join(output_dir, "trial-{:05d}.pth".format(trial_number))

    to_save = dict()
    for name, model in save["models"].items():
        if model:
            to_save[name] = model.state_dict()

    for name, optimizer in save["optimizers"].items():
        if optimizer:
            to_save[name] = optimizer.state_dict()
    to_save["epoch"] = epoch

    save_on_master(to_save, checkpoint_path)
    return checkpoint_path

def get_last_checkpoint(output_dir):
    """
    Get the last checkpoint from the checkpointing folder.
    Args:
        path_to_job (string): the path to the folder of the current job.
    """
    d = output_dir
    names = pathmgr.ls(d) if pathmgr.exists(d) else []
    names = [f for f in names if "checkpoint" in f]
    if len(names) == 0:
        print("No checkpoints found in '{}'.".format(d))
        return None
    else:
        # Sort the checkpoints by epoch.
        name = sorted(names)[-1]
        return os.path.join(d, name)


def load_model(output_dir,
               save,
               epoch = None,
               resume = None,
               is_train = True,verbose = False):

    if resume is None:
        resume = get_last_checkpoint(output_dir)
    if epoch:
        resume = os.path.join(output_dir, "checkpoint-{:05d}.pth".format(epoch))

    with pathmgr.open(resume, "rb") as f:
        checkpoint = torch.load(f, map_location="cpu")

    for name, model in save["models"].items():
        if model:
            model.load_state_dict(checkpoint[name])
    if verbose:
        print("Resume checkpoint %s" % resume)
    if is_train:
        for name, optimizer in save["optimizers"].items():
            optimizer.load_state_dict(checkpoint[name])
        if verbose:
            print("With optim & sched!")

def load_best_model(output_dir,
               save,
               resume = None,
               is_train = True,verbose = False):
    if not resume:
        resume = os.path.join(output_dir, "best.pth")

    if resume:
        with pathmgr.open(resume, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")

        for name, model in save["models"].items():
            if model:
                model.load_state_dict(checkpoint[name])
        if verbose:
            print("Resume checkpoint %s" % resume)
        if is_train:
            for name, optimizer in save["optimizers"].items():
                optimizer.load_state_dict(checkpoint[name])
            print("With optim & sched!")

def plot_metric_against_baseline(total_asset,buy_and_hold,alg,task,color='darkcyan',save_dir=None,metric_name='Total asset'):
    # print('total_asset shape is:',total_asset.shape)
    # print(total_asset)

    #normalize total_asset and buy_and_hold by the first value
    # print('total_asset shape is:',total_asset.shape,total_asset)
    if buy_and_hold is not None:
        buy_and_hold = buy_and_hold / total_asset[0]
    total_asset=total_asset/total_asset[0]

    x = range(len(total_asset))
    # print('total_asset shape is:',total_asset.shape)
    # print('x shape is:',len(x))
    # set figure size
    plt.figure(figsize=(10, 6))
    y=total_asset
    plt.plot(x, y, color, label=alg)
    plt.xlabel('Trading times',size=18)
    plt.ylabel(metric_name,size=18)
    if buy_and_hold is not None:
        # print('buy and hold shape is:',buy_and_hold.shape)
        plt.plot(x, buy_and_hold, 'r', label='Buy and Hold')
    plt.grid(ls='--')
    plt.legend(fancybox=True, ncol=1)
    # set title
    plt.title(f'{metric_name} of {alg} in {task}')
    if save_dir is not None:
        plt.savefig(osp.join(save_dir,f"Visualization_{task}.png"))
    # plt.show()


Transition = namedtuple("Transition", ['state', 'action', 'reward', 'undone','next_state'])

class GeneralReplayBuffer:
    def __init__(self,
                 transition: namedtuple,
                 shapes: dict,
                 max_size: int,
                 num_seqs: int,
                 device: torch.device
                 ):

        self.p = 0  # pointer
        self.if_full = False
        self.cur_size = 0
        self.add_size = 0
        self.add_item = None
        self.max_size = max_size
        self.num_seqs = num_seqs
        self.device = device

        # initialize
        self.transition = transition
        self.names = self.transition._fields
        self.shapes = shapes
        self.storage = OrderedDict()
        for name in self.names:
            assert name in self.shapes
            # (max_size, num_seqs, dim1, dim2, ...)
            self.storage[name] = torch.empty(self.shapes[name], dtype=torch.float32, device=self.device)

    def update(self, items: namedtuple):
        # check shape
        for name in self.names:
            assert name in self.storage
            assert getattr(items, name).shape[1:] == self.storage[name].shape[1:]

        # add size
        self.add_size = getattr(items, self.names[0]).shape[0]

        p = self.p + self.add_size  # pointer
        if p > self.max_size:
            self.if_full = True
            p0 = self.p
            p1 = self.max_size
            p2 = self.max_size - self.p
            p = p - self.max_size

            for name in self.names:
                self.storage[name][p0:p1], self.storage[name][0:p] = \
                    getattr(items, name)[:p2], getattr(items, name)[-p:]
        else:
            for name in self.names:
                self.storage[name][self.p:p] = getattr(items, name)
        self.p = p
        self.cur_size = self.max_size if self.if_full else self.p

    def clear(self):
        for name in self.names:
            assert name in self.shapes
            # (max_size, num_seqs, dim1, dim2, ...)
            self.storage[name] = torch.empty(self.shapes[name], dtype=torch.float32, device=self.device)
    def sample(self, batch_size: int) -> namedtuple:
        sample_len = self.cur_size - 1

        ids = torch.randint(sample_len * self.num_seqs, size=(batch_size,), requires_grad=False)
        ids0 = torch.fmod(ids, sample_len)  # ids % sample_len
        ids1 = torch.div(ids, sample_len, rounding_mode='floor')  # ids // sample_len

        sample_data = OrderedDict()
        for name in self.names:
            sample_data[name] = self.storage[name][ids0, ids1]
        return self.transition(**sample_data)

    def save_or_load_history(self, cwd: str, if_save: bool):
        if if_save:
            for name, item in self.storage.items():
                if self.cur_size == self.p:
                    buf_item = item[:self.cur_size]
                else:
                    buf_item = torch.vstack((item[self.p:self.cur_size], item[0:self.p]))
                file_path = f"{cwd}/replay_buffer_{name}.pt"
                print(f"| {self.__class__.__name__}: Save {file_path}")
                torch.save(buf_item, file_path)

        elif all([os.path.isfile(f"{cwd}/replay_buffer_{name}.pt") for name, item in self.storage.items()]):
            max_sizes = []
            for name, item in self.storage.items():
                file_path = f"{cwd}/replay_buffer_{name}.pt"
                print(f"| {self.__class__.__name__}: Load {file_path}")
                buf_item = torch.load(file_path)

                max_size = buf_item.shape[0]
                item[:max_size] = buf_item
                max_sizes.append(max_size)
            assert all([max_size == max_sizes[0] for max_size in max_sizes])
            self.cur_size = max_sizes[0]


class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 - 0.5


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """

    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)


def time_features(dates, freq='h'):
    return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)])

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.ARR_min = np.Inf
        self.delta = delta

    def __call__(self, ARR, model, path):
        score = -ARR
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(ARR, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(ARR, model, path)
            self.counter = 0

    def save_checkpoint(self, ARR, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.ARR_min:.6f} --> {ARR:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path)
        self.ARR_min = ARR


