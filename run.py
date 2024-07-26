import argparse
import os
import sys
import warnings
import os.path as osp
warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.abspath("."))
sys.path.append(ROOT)

import torch
from utils import replace_cfg_vals
from mmcv import Config
from sklearn.preprocessing import StandardScaler
from builder import *
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from datasets import dataset,AAAI_mse
from environments import deeptrader_environment,AAAI_environment
from nets import  ASU,MSU,AAAI,AAAI_mse
from losses.custom import MSELoss
from optimizers.custom import Adam,Adagrad,Adadelta,AdamW
from agents import deeptrader,AAAI
from trainers import deeptrader_trainer,AAAI_reinforce,AAAI_mse
from transition.custom import TransitionDeepTrader
torch.autograd.set_detect_anomaly(True)


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
torch.autograd.set_detect_anomaly(True)
torch.set_num_threads(1)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = "1"


import wandb

def setup(rank, world_size,gpu_ids):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)#nccl
    torch.cuda.set_device(gpu_ids[rank])

def cleanup():
    dist.destroy_process_group()

def calculate_return(tic_data):
    # close 가격의 비율 계산
    close_current = tic_data['close']
    close_next = close_current.shift(-1)
    returns = close_next / close_current - 1
    return returns

def add_returns_label(df):
    # 각 티커별로 return 계산 후 'label' 칼럼 추가
    returns = df.groupby(level='tic').apply(calculate_return).droplevel(0)
    df['returns'] = returns
    df.dropna(subset=['returns'], inplace=True)  # NaN 값 제거 (마지막 행)
    df['label'] = df.groupby(level='date')['returns'].transform(lambda x: (x - x.mean()) / x.std())
    return df

def normalize_ticker_data(df, tech_indicator_list):


    # 각 티커별 데이터프레임 저장을 위한 딕셔너리
    ticker_data = {}
    scalers = {}

    # 멀티인덱스에서 티커별로 데이터 분리
    for ticker in df.index.get_level_values('tic').unique():
        ticker_df = df.xs(ticker, level='tic')[tech_indicator_list].copy()

        # StandardScaler를 사용하여 각 티커별 특징 표준화
        scaler = StandardScaler()
        ticker_df_scaled = scaler.fit_transform(ticker_df)

        # 스케일러와 표준화된 데이터 저장
        scalers[ticker] = scaler
        ticker_data[ticker] = pd.DataFrame(ticker_df_scaled, index=ticker_df.index, columns=ticker_df.columns)

    # 표준화된 데이터프레임 다시 결합
    normalized_df = pd.concat(ticker_data, keys=ticker_data.keys(), names=['tic', 'date'])
    normalized_df = normalized_df.swaplevel('date', 'tic').sort_index()
    # label 칼럼을 추가
    normalized_df['returns'] = df['returns']
    normalized_df['label'] = df['label']
    return normalized_df, scalers

def apply_scaler_to_dataset(data_set, scalers, tech_indicator_list):
    ticker_data = {}

    for ticker in data_set.data.index.get_level_values('tic').unique():
        ticker_df = data_set.data.xs(ticker, level='tic')[tech_indicator_list].copy()
        scaler = scalers[ticker]
        ticker_df_scaled = scaler.transform(ticker_df)
        ticker_data[ticker] = pd.DataFrame(ticker_df_scaled, index=ticker_df.index, columns=ticker_df.columns)

    normalized_df = pd.concat(ticker_data, keys=ticker_data.keys(), names=['tic', 'date'])
    normalized_df = normalized_df.swaplevel('date', 'tic').sort_index()

    # 기존의 'label' 칼럼을 다시 추가
    normalized_df['returns'] = data_set.data['returns']
    normalized_df['label'] = data_set.data['label']
    data_set.data = normalized_df




def normalize_datewise(df, tech_indicator_list):
    # 날짜별로 데이터를 그룹화하여 정규화 작업 수행
    def z_score(group):
        return (group - group.mean()) / group.std()

    df[tech_indicator_list] = df.groupby(level='date')[tech_indicator_list].transform(z_score)

    return df

def apply_datewise_normalization_to_dataset(data_set, tech_indicator_list):
    data_set.data = normalize_datewise(data_set.data, tech_indicator_list)
    return data_set


def set_environments(cfg):
    # Build datasets
    train_data_set = build_dataset(cfg, default_args=dict(flag='train'))
    valid_data_set = build_dataset(cfg, default_args=dict(flag='valid'))
    test_data_set = build_dataset(cfg, default_args=dict(flag='test'))

    tech_indicator_list = train_data_set.tech_indicator_list
    # Add return column to train, valid, and test datasets
    train_data_set.data = add_returns_label(train_data_set.data)
    valid_data_set.data = add_returns_label(valid_data_set.data)
    test_data_set.data = add_returns_label(test_data_set.data)

    # 정규화 방법 선택
    normalization_method = cfg['common_params']['norm_method']

    if normalization_method == 'ticker':
        # Feature별 정규화
        train_data_set.data, scalers = normalize_ticker_data(train_data_set.data, tech_indicator_list)
        apply_scaler_to_dataset(valid_data_set, scalers, tech_indicator_list)
        apply_scaler_to_dataset(test_data_set, scalers, tech_indicator_list)
    elif normalization_method == 'date':
        # Date별 정규화
        train_data_set = apply_datewise_normalization_to_dataset(train_data_set, tech_indicator_list)
        valid_data_set = apply_datewise_normalization_to_dataset(valid_data_set, tech_indicator_list)
        test_data_set = apply_datewise_normalization_to_dataset(test_data_set, tech_indicator_list)

    # Build environments
    train_environment = build_environment(cfg, default_args=dict(dataset=train_data_set, task="train"))
    valid_environment = build_environment(cfg, default_args=dict(dataset=valid_data_set, task="valid"))
    test_environment = build_environment(cfg, default_args=dict(dataset=test_data_set, task="test"))

    return train_environment, valid_environment, test_environment

def run(cfg):
    train_environment, valid_environment, test_environment = set_environments(cfg)

    action_dim = train_environment.action_dim
    state_dim = train_environment.state_dim
    input_dim = len(train_environment.tech_indicator_list)
    criterion = build_loss(cfg)
    transition = build_transition(cfg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    act_net = build_net(cfg.act_net).to(device)
    optimizer = build_optimizer(cfg, default_args=dict(params=act_net.parameters()))

    agent = build_agent(cfg, default_args=dict(action_dim=action_dim, state_dim=state_dim, act=act_net,
                                               network_optimizer=optimizer, criterion=criterion,
                                               transition=transition,
                                               device=device))

    trainer = build_trainer(cfg,
                            default_args=dict(train_environment=train_environment, valid_environment=valid_environment,
                                              test_environment=test_environment, agent=agent, device=device))
    wandb.init(project=cfg.common_params['wandb_project_name'], group=cfg.common_params['wandb_group_name'],
               name=cfg.common_params['wandb_session_name'])
    wandb.config.update(cfg.act_net)

    trainer.train_and_valid()

    trainer.test()
    wandb.finish()
def run_ddp(rank, world_size, cfg,gpu_ids):
    setup(rank, world_size,gpu_ids)

    train_environment, valid_environment, test_environment = set_environments(cfg)

    action_dim = train_environment.action_dim
    state_dim = train_environment.state_dim
    input_dim = len(train_environment.tech_indicator_list)
    criterion = build_loss(cfg)
    transition = build_transition(cfg)
    
    device = torch.device(f'cuda:{gpu_ids[rank]}')
    model = build_net(cfg.act_net).to(device)
    ddp_model = DDP(model, device_ids=[gpu_ids[rank]],output_device=device,find_unused_parameters=True)
    print(f"Rank {rank}: DDP model initialized on device {device}")
    print('ddp_modeL_device_ids:',ddp_model.device_ids,'ddp_model_output_device:',ddp_model.output_device)
    optimizer = build_optimizer(cfg, default_args=dict(params=ddp_model.parameters()))
  
    agent = build_agent(cfg, default_args=dict(action_dim=action_dim, state_dim=state_dim, act=ddp_model,
                                               network_optimizer=optimizer, criterion=criterion,
                                               transition=transition,
                                               device=device))
   
    trainer = build_trainer(cfg, default_args=dict(train_environment=train_environment, valid_environment=valid_environment,
                                                   test_environment=test_environment, agent=agent, device=device))
    if rank ==0:
        wandb.init(project=cfg.common_params['wandb_project_name'], group=cfg.common_params['wandb_group_name'], name=cfg.common_params['wandb_session_name'])
        wandb.config.update(cfg)
   
    trainer.train_and_valid(rank, world_size)
    trainer.test(rank, world_size)
    if rank ==0:
        wandb.finish()
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=osp.join(ROOT, "2025_AAAI", "configs", "dj30_AAAI_mse.py"),
                        help="download datasets config file path")
    parser.add_argument('--use_ddp', action='store_true', help='Use DDP for training',default=True)
    args, _ = parser.parse_known_args()
    cfg = Config.fromfile(args.config)
    cfg = replace_cfg_vals(cfg)

    if args.use_ddp:
        # gpu_ids = cfg.gpu_ids
        gpu_ids = [0]
        world_size = len(gpu_ids)
        print(f"Using GPU IDs: {gpu_ids}")
        mp.spawn(run_ddp, args=(world_size, cfg,gpu_ids), nprocs=world_size, join=True)
    else:
        run(cfg)