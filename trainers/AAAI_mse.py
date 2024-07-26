import random
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
# from custom import Trainer
from builder import TRAINERS
from utils import get_attr, save_model, save_best_model, load_model, load_best_model, plot_metric_against_baseline, \
    GeneralReplayBuffer, print_metrics, EarlyStopping
import numpy as np
import os
import pandas as pd
from collections import OrderedDict
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Sampler
import copy
import gc
import multiprocessing
from torch.cuda.amp import autocast, GradScaler
from datetime import datetime
import wandb

"""this algorithms is based on the paper 'DeepTrader: A Deep Reinforcement Learning Approach for Risk-Return Balanced Portfolio Management with Market Conditions Embedding'
and code https://github.com/CMACH508/DeepTrader. However, since the data is open-souce, we make some modification
the tic-level tabular data follows the rest algrithms  in portfolio management while we use the corriance matrix to represents the graph information
and average tic-level tabular data as the market information 
"""
torch.autograd.set_detect_anomaly(True)
def calc_ic(pred, label):
    if pred.ndim > 1:
        pred = pred.flatten()
    if label.ndim > 1:
        label = label.flatten()
    df = pd.DataFrame({'pred':pred, 'label':label})
    ic = df['pred'].corr(df['label'])
    ric = df['pred'].corr(df['label'], method='spearman')
    return ic, ric

def custom_collate_fn(batch):
    seq_x = torch.tensor([item[0] for item in batch], dtype=torch.float32)
    seq_y = torch.tensor([item[1] for item in batch], dtype=torch.float32)
    seq_x_mark = torch.tensor([item[2] for item in batch], dtype=torch.float32)
    seq_y_mark = torch.tensor([item[3] for item in batch], dtype=torch.float32)
    return_data = torch.tensor([item[4] for item in batch], dtype=torch.float32)
    date_data = [item[5] for item in batch]
    return seq_x, seq_y, seq_x_mark, seq_y_mark, return_data, date_data


@TRAINERS.register_module()
class AAAI_mse:
    def __init__(self, **kwargs):
        super(AAAI_mse, self).__init__()

        self.pred_len = get_attr(kwargs,'pred_len',5)
        self.num_envs = int(get_attr(kwargs, "num_envs", 1))
        self.device = get_attr(kwargs, "device", None)
        self.gamma = get_attr(kwargs, 'gamma', 0.05)
        self.epochs = get_attr(kwargs, "epochs", 20)
        self.batch_size = get_attr(kwargs, 'batch_size', 1)
        self.temperature = get_attr(kwargs, 'temperature', 1)
        self.train_environment = get_attr(kwargs, "train_environment", None)
        self.valid_environment = get_attr(kwargs, "valid_environment", None)
        self.test_environment = get_attr(kwargs, "test_environment", None)
        self.agent = get_attr(kwargs, "agent", None)
        self.work_dir = get_attr(kwargs, "work_dir", None)
        self.wandb_project_name = get_attr(kwargs, "wandb_project_name", None)
        self.wandb_group_name = get_attr(kwargs, "work_group_name", None)
        self.wandb_session_name = get_attr(kwargs, "wandb_session_name", None)
        start_time = datetime.now().strftime('%m%d/%H%M%S')
        work_base = os.path.join(ROOT, self.work_dir)
        WANDB_NAME = f'{self.wandb_project_name}_{self.wandb_group_name}_{self.wandb_session_name}'
        self.work_dir = os.path.join(work_base, WANDB_NAME + '_' + start_time)

        self.seeds_list = get_attr(kwargs, "seeds_list",
                                   (12345, 32, 412, 123, 13123, 93434, 1234, 1201, 1206, 419, 23153, 4341))
        self.random_seed = random.choice(self.seeds_list)

        self.num_threads = int(get_attr(kwargs, "num_threads", 8))

        self.if_remove = get_attr(kwargs, "if_remove", False)
        self.if_discrete = get_attr(kwargs, "if_discrete", False)
        self.if_off_policy = get_attr(kwargs, "if_off_policy", True)
        self.if_keep_save = get_attr(kwargs, "if_keep_save", True)
        self.if_over_write = get_attr(kwargs, "if_over_write", False)
        self.if_save_buffer = get_attr(kwargs, "if_save_buffer", False)

        if self.if_off_policy:  # off-policy
            self.batch_size = int(get_attr(kwargs, "batch_size", 1))
            self.horizon_len = int(get_attr(kwargs, "horizon_len", 512))
            self.buffer_size = int(get_attr(kwargs, "buffer_size", 1000))
        else:  # on-policy
            self.batch_size = int(get_attr(kwargs, "batch_size", 1))
            self.horizon_len = int(get_attr(kwargs, "horizon_len", 512))
            self.buffer_size = int(get_attr(kwargs, "buffer_size", 128))
        self.epochs = int(get_attr(kwargs, "epochs", 20))

        self.state_dim = self.agent.state_dim
        self.action_dim = self.agent.action_dim
        self.timesteps = self.agent.timesteps
        self.transition = self.agent.transition

        self.transition_shapes = OrderedDict({
            'state': (self.buffer_size, self.num_envs,
                      self.action_dim, self.state_dim,
                      self.timesteps),
            'action': (self.buffer_size, self.num_envs, self.action_dim),
            'reward': (self.buffer_size, self.num_envs),
            'undone': (self.buffer_size, self.num_envs),
            'next_state': (self.buffer_size, self.num_envs,
                           self.action_dim, self.state_dim,
                           self.timesteps),
            'correlation_matrix': (self.buffer_size, self.num_envs,
                                   self.action_dim, self.action_dim),
            'next_correlation_matrix': (self.buffer_size, self.num_envs,
                                        self.action_dim, self.action_dim),
            'state_market': (self.buffer_size, self.num_envs, self.timesteps, self.state_dim),
            'roh_bar_market': (self.buffer_size, self.num_envs),
        })

    # def _init_data_loader(self, data,rank,world_size, shuffle=False, drop_last=True):
    #     # sampler = DailyBatchSamplerRandom(data, shuffle)
    #     sampler = DistributedSampler(data, num_replicas=world_size, rank=rank,shuffle=shuffle)
    #     data_loader = DataLoader(data, shuffle=shuffle, sampler=sampler,drop_last=drop_last, collate_fn=custom_collate_fn)
    #     # num_workers = multiprocessing.cpu_count() // 2
    #     # data_loader = DataLoader(data,shuffle=shuffle, drop_last=drop_last,collate_fn=custom_collate_fn,num_workers=num_workers)
    #     return data_loader
    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)
        loss = (pred[mask] - label[mask]) ** 2
        return torch.mean(loss)
    def _init_data_loader(self, dataset, rank, world_size, shuffle=False, drop_last=True):
        if rank != None:
            total_size = len(dataset)
            chunk_size = total_size // world_size
            indices = list(range(rank * chunk_size, (rank + 1) * chunk_size))
            subset = torch.utils.data.Subset(dataset, indices)
            data_loader = DataLoader(subset, batch_size=self.batch_size, collate_fn=custom_collate_fn, shuffle=shuffle,
                                     drop_last=drop_last)
        else:
            data_loader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=custom_collate_fn, shuffle=shuffle,
                                     drop_last=drop_last)
        return data_loader

    def init_before_training(self, rank):
        random.seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.backends.cudnn.benckmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_num_threads(self.num_threads)
        torch.set_default_dtype(torch.float32)
        self.checkpoints_path = os.path.join(self.work_dir, "checkpoints")
        if rank == 0 or rank == None:
            # if not os.path.exists(self.work_dir):
            #    os.makedirs(self.work_dir)
            '''remove history'''
            if self.if_remove is None:
                self.if_remove = bool(input(f"| Arguments PRESS 'y' to REMOVE: {self.work_dir}? ") == 'y')
            if self.if_remove:
                import shutil
                shutil.rmtree(self.work_dir, ignore_errors=True)
                print(f"| Arguments Remove work_dir: {self.work_dir}")
            else:
                print(f"| Arguments Keep work_dir: {self.work_dir}",end='\n')
            # os.makedirs(self.work_dir, exist_ok=True)

            if not os.path.exists(self.checkpoints_path):
                os.makedirs(self.checkpoints_path, exist_ok=True)
        if rank != None:
            dist.barrier()

    def train_and_valid(self, rank=None, world_size=None):
        self.init_before_training(rank)
        '''init agent.last_state'''
        train_loader = self._init_data_loader(self.train_environment.dataset, rank, world_size, shuffle=False,
                                              drop_last=True)
        valid_loader = self._init_data_loader(self.valid_environment.dataset, rank, world_size, shuffle=False,
                                              drop_last=True)

        '''init buffer'''
        if self.if_off_policy:
            buffer = GeneralReplayBuffer(
                transition=self.transition,
                shapes=self.transition_shapes,
                num_seqs=self.num_envs,
                max_size=self.buffer_size,
                device=self.device,
            )
            # buffer_items = self.agent.explore_env(self.train_environment, self.horizon_len)
            # buffer.update(buffer_items)
        else:
            buffer = []

        early_stopping = EarlyStopping(patience=10, verbose=True)
        for epoch in range(1, self.epochs + 1):
            # if rank == 0 or rank == None:
            #     print("Train Episode: [{}/{}]".format(epoch, self.epochs))
            train_loss = self.train_epoch(train_loader, rank, world_size)
            valid_loss = self.valid_epoch(valid_loader, rank, world_size)

            # print("Epoch %d, train_loss %.6f, valid_loss %.6f " % (epoch, train_loss, val_loss))
            if rank == 0 or rank == None:
                print("Epoch %d, train_loss %.6f, valid_loss %.6f " % (epoch, train_loss, valid_loss))
                wandb.log({'epoch': epoch, 'train_loss': train_loss, 'valid_loss': valid_loss})
                early_stopping(valid_loss, self.agent.act,
                               os.path.join(self.checkpoints_path, 'best_arr.pkl'))
                stop = early_stopping.early_stop
                if stop:
                    print("Early stopping")
            else:
                stop = False
            if rank != None:
                stop_tensor = torch.tensor(int(stop), dtype=torch.int).to(self.device)
                dist.broadcast(stop_tensor, src=0)
                stop = bool(stop_tensor.item())
                if stop:
                    break

    def test(self, rank=None, world_size=None):
        self.agent.act.load_state_dict(torch.load(os.path.join(self.checkpoints_path, 'best_arr.pkl')))
        print("Successfully loaded best checkpoint...")
        test_loader = self._init_data_loader(self.test_environment.dataset, rank, world_size, shuffle=False,
                                             drop_last=True)

        print("Test Best Episode")
        self.agent.act.eval()
        local_weights = []
        local_return_data = []
        local_date_data = []
        local_preds = []
        local_ic = []
        local_ric = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, return_data, date_data) in enumerate(test_loader):
            # iter_count += 1
            # model_optim.zero_grad()
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

            return_data = return_data.float().to(self.device)
            with torch.no_grad():
                pred = self.agent.act(batch_x, batch_x_mark, batch_y, batch_y_mark)
                label = torch.squeeze(batch_y, dim=0)[:, -self.pred_len:, -1]
                local_preds.append(pred.cpu().numpy().ravel())
                daily_ic, daily_ric = calc_ic(pred.cpu().numpy(), label.cpu().detach().numpy())
                local_ic.append(daily_ic)
                local_ric.append(daily_ric)
                #for portfolio
                pred_port = pred.mean(dim=1)
                weights = torch.softmax(pred_port / self.temperature, dim=0)
                local_weights.append(weights)
                local_return_data.append(return_data)
                local_date_data.append(date_data)



            torch.cuda.empty_cache()
            # Stack the weights and return_data

        local_preds_tensor = torch.tensor(local_preds, device=self.device)
        local_ic_tensor = torch.tensor(local_ic, device=self.device)
        local_ric_tensor = torch.tensor(local_ric, device=self.device)

        all_weights = torch.cat(local_weights, dim=0).to(self.device)
        all_return_data = torch.cat(local_return_data, dim=0).to(self.device)
        all_date_data = sum(local_date_data, [])

        if rank is not None:
            # Synchronize before gathering
            dist.barrier()
            # Gather data from all ranks
            gathered_preds = [torch.zeros_like(local_preds_tensor) for _ in range(world_size)]
            gathered_ic = [torch.zeros_like(local_ic_tensor) for _ in range(world_size)]
            gathered_ric = [torch.zeros_like(local_ric_tensor) for _ in range(world_size)]

            dist.all_gather(gathered_preds, local_preds_tensor)
            dist.all_gather(gathered_ic, local_ic_tensor)
            dist.all_gather(gathered_ric, local_ric_tensor)
            # Gather all weights and return_data from all ranks
            gathered_weights = [torch.zeros_like(all_weights) for _ in range(world_size)]
            gathered_return_data = [torch.zeros_like(all_return_data) for _ in range(world_size)]
            gathered_date_data = [None for _ in range(world_size)]

            dist.all_gather(gathered_weights, all_weights)
            dist.all_gather(gathered_return_data, all_return_data)
            dist.all_gather_object(gathered_date_data, all_date_data)
            if rank ==0:
                gathered_preds = torch.cat(gathered_preds).cpu().numpy()
                gathered_ic = torch.cat(gathered_ic).cpu().numpy()
                gathered_ric = torch.cat(gathered_ric).cpu().numpy()
                # Concatenate the gathered weights and return_data
                gathered_weights = torch.cat(gathered_weights, dim=0).reshape(-1, batch_x.shape[1])
                gathered_return_data = torch.cat(gathered_return_data, dim=0)
                gathered_date_data = sum(gathered_date_data, [])

        else:
            # DDP가 비활성화된 경우 로컬에서 직접 처리
            gathered_preds = local_preds_tensor.cpu().numpy()
            gathered_ic = local_ic_tensor.cpu().numpy()
            gathered_ric = local_ric_tensor.cpu().numpy()
            gathered_weights = all_weights
            gathered_return_data = all_return_data
            gathered_date_data = all_date_data


        if rank == 0 or rank == None:
            for weights, return_data, date_data in zip(gathered_weights, gathered_return_data, gathered_date_data):
                self.test_environment.step(weights, return_data, [date_data])

            # predictions = np.concatenate(gathered_preds)
            ic_mean = np.mean(gathered_ic)
            icir = np.mean(gathered_ic) / np.std(gathered_ic)
            ric_mean = np.mean(gathered_ric)
            ricir = np.mean(gathered_ric) / np.std(gathered_ric)

            metrics = {
                'IC': ic_mean,
                'ICIR': icir,
                'RIC': ric_mean,
                'RICIR': ricir
            }

            start_date, end_date, tr, sharpe_ratio, vol, mdd, cr, sor = self.test_environment.analysis_result()
            stats = OrderedDict(
                {"Start Date": [start_date],
                 "End Date": [end_date],
                 "Total Return": ["{:04f}%".format((tr * 100).item())],
                 "Sharp Ratio": ["{:04f}".format(sharpe_ratio)],
                 "Volatility": ["{:04f}%".format(vol * 100)],
                 "Max Drawdown": ["{:04f}%".format(mdd * 100)],
                 "Calmar Ratio": ["{:04f}".format(cr.item())],
                 "Sortino Ratio": ["{:04f}".format(sor.item())],
                 }
            )

            wandb.log({'Total_Return': tr * 100, 'Sharp Ratio': sharpe_ratio, 'Volatility': vol * 100,
                       'Max Drawdown': mdd * 100,
                       'Calmar Ratio': cr, 'Sortino Ratio': sor})

            print('test result')
            print('-' * 100)
            print(metrics)
            print('-' * 100)
            table_port = print_metrics(stats)
            print('test result')
            print('-' * 100)
            print(table_port)


        if rank is not None:
            dist.barrier()





    def train_epoch(self, data_loader, rank, world_size):
        self.agent.act.train()
        self.train_environment.reset()


        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, return_data, date_data) in enumerate(data_loader):
            if i == 0:
                print(f'rank:{rank} start_date:{date_data}')
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)
            return_data = return_data.float().to(self.device)
            losses = []
            with autocast():
                pred = self.agent.act(batch_x, batch_x_mark, batch_y, batch_y_mark)
                label = torch.squeeze(batch_y,dim=0)[:,-self.pred_len:,-1]
                loss = self.loss_fn(pred, label)
                losses.append(loss.item())

                self.agent.network_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.agent.act.parameters(), 3.0)
                self.agent.network_optimizer.step()



            gc.collect()
            torch.cuda.empty_cache()




        # Convert list to tensor
        local_losses = torch.tensor(losses).to(self.device)
        if rank != None:
            dist.barrier()
            # Gather losses from all GPUs
            gathered_losses = [torch.zeros_like(local_losses) for _ in range(world_size)]
            dist.all_gather(gathered_losses, local_losses)
            # Calculate the combined loss on rank 0
            if rank == 0:
                total_loss = torch.cat(gathered_losses).mean()

        else:
            total_loss = float(np.mean(losses))

        return total_loss

    def valid_epoch(self, data_loader, rank, world_size):
        self.agent.act.eval()
        self.valid_environment.reset()

        local_losses = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, return_data, date_data) in enumerate(data_loader):

            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)
            return_data = return_data.float().to(self.device)
            with torch.no_grad():
                pred = self.agent.act(batch_x, batch_x_mark, batch_y, batch_y_mark)
                label = torch.squeeze(batch_y, dim=0)[:, -self.pred_len:, -1]
                loss = self.loss_fn(pred, label)
                local_losses.append(loss.item())

            torch.cuda.empty_cache()
            # Convert local losses to a tensor and gather them from all ranks
        local_losses_tensor = torch.tensor(local_losses).to(self.device)


        if rank != None:
            dist.barrier()
            gathered_losses = [torch.zeros_like(local_losses_tensor) for _ in range(world_size)]
            dist.all_gather(gathered_losses, local_losses_tensor)

            if rank == 0:
                all_losses = torch.cat(gathered_losses)
                total_loss = float(torch.mean(all_losses))

        else:
            total_loss = float(np.mean(local_losses))

        return total_loss

