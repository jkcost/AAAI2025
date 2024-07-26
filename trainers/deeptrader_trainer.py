import random
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
# from custom import Trainer
from builder import TRAINERS
from utils import get_attr,save_model, save_best_model, load_model,load_best_model,plot_metric_against_baseline,GeneralReplayBuffer
import numpy as np
import os
import pandas as pd
from collections import OrderedDict
from torch.cuda.amp import autocast, GradScaler
"""this algorithms is based on the paper 'DeepTrader: A Deep Reinforcement Learning Approach for Risk-Return Balanced Portfolio Management with Market Conditions Embedding'
and code https://github.com/CMACH508/DeepTrader. However, since the data is open-souce, we make some modification
the tic-level tabular data follows the rest algrithms  in portfolio management while we use the corriance matrix to represents the graph information
and average tic-level tabular data as the market information 
"""
torch.autograd.set_detect_anomaly(True)

def make_market_information(df, technical_indicator):
    # based on the information, calculate the average for technical_indicator to present the market average
    all_dataframe_list = []
    index_list = df.index.unique().tolist()
    index_list.sort()
    for i in index_list:
        information = df[df.index == i]
        new_dataframe = []
        for tech in technical_indicator:
            tech_value = np.mean(information[tech])
            new_dataframe.append(tech_value)
        all_dataframe_list.append(new_dataframe)
    new_df = pd.DataFrame(all_dataframe_list,
                          columns=technical_indicator).values
    # new_df.to_csv(store_path)
    return new_df


def make_correlation_information(df: pd.DataFrame, feature="adjclose"):
    # based on the information, we are making the correlation matrix(which is N*N matric where N is the number of tickers) based on the specific
    # feature here,  as default is adjclose
    df.sort_values(by='tic', ascending=True, inplace=True)
    array_symbols = df['tic'].values

    # get data, put into dictionary then dataframe
    dict_sym_ac = {}  # key=symbol, value=array of adj close
    for sym in array_symbols:
        dftemp = df[df['tic'] == sym]
        dict_sym_ac[sym] = dftemp['adjcp'].values

    # create correlation coeff df
    dfdata = pd.DataFrame.from_dict(dict_sym_ac)
    dfcc = dfdata.corr().round(2)
    dfcc = dfcc.values
    return dfcc

@TRAINERS.register_module()
class DeepTraderTrainer:
    def __init__(self, **kwargs):
        super(DeepTraderTrainer, self).__init__()

        self.num_envs = int(get_attr(kwargs, "num_envs", 1))
        self.device = get_attr(kwargs, "device", None)
        self.gamma = get_attr(kwargs,'gamma', 0.05)
        self.epochs = get_attr(kwargs, "epochs", 20)
        self.batch_size = get_attr(kwargs,'batch_size',1)
        self.train_environment = get_attr(kwargs, "train_environment", None)
        self.valid_environment = get_attr(kwargs, "valid_environment", None)
        self.test_environment = get_attr(kwargs, "test_environment", None)
        self.agent = get_attr(kwargs, "agent", None)
        self.work_dir = get_attr(kwargs, "work_dir", None)
        self.work_dir = os.path.join(ROOT, self.work_dir)
        self.scaler = GradScaler()
        self.seeds_list = get_attr(kwargs, "seeds_list", (12345,))
        self.random_seed = random.choice(self.seeds_list)

        self.num_threads = int(get_attr(kwargs, "num_threads", 8))

        self.if_remove = get_attr(kwargs, "if_remove", False)
        self.if_discrete = get_attr(kwargs, "if_discrete", False)
        self.if_off_policy = get_attr(kwargs, "if_off_policy", True)
        self.if_keep_save = get_attr(kwargs, "if_keep_save", True)
        self.if_over_write = get_attr(kwargs, "if_over_write", False)
        self.if_save_buffer = get_attr(kwargs, "if_save_buffer", False)

        if self.if_off_policy:  # off-policy
            self.batch_size = int(get_attr(kwargs, "batch_size", 64))
            self.horizon_len = int(get_attr(kwargs, "horizon_len", 512))
            self.buffer_size = int(get_attr(kwargs, "buffer_size", 1000))
        else:  # on-policy
            self.batch_size = int(get_attr(kwargs, "batch_size", 128))
            self.horizon_len = int(get_attr(kwargs, "horizon_len", 512))
            self.buffer_size = int(get_attr(kwargs, "buffer_size", 128))
        self.epochs = int(get_attr(kwargs, "epochs", 20))

        self.state_dim = self.agent.state_dim
        self.action_dim = self.agent.action_dim
        self.timesteps = self.agent.timesteps
        self.transition = self.agent.transition

        self.transition_shapes = OrderedDict({
            'state': (self.buffer_size, self.num_envs,
                      self.action_dim,self.state_dim,
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
            'state_market': (self.buffer_size, self.num_envs,self.timesteps , self.state_dim),
            'roh_bar_market':(self.buffer_size, self.num_envs),
        })

        self.init_before_training()

    def init_before_training(self):
        random.seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.backends.cudnn.benckmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_num_threads(self.num_threads)
        torch.set_default_dtype(torch.float32)

        '''remove history'''
        if self.if_remove is None:
            self.if_remove = bool(input(f"| Arguments PRESS 'y' to REMOVE: {self.work_dir}? ") == 'y')
        if self.if_remove:
            import shutil
            shutil.rmtree(self.work_dir, ignore_errors=True)
            print(f"| Arguments Remove work_dir: {self.work_dir}")
        else:
            print(f"| Arguments Keep work_dir: {self.work_dir}")
        os.makedirs(self.work_dir, exist_ok=True)

        self.checkpoints_path = os.path.join(self.work_dir, "checkpoints")
        if not os.path.exists(self.checkpoints_path):
            os.makedirs(self.checkpoints_path, exist_ok=True)

    def train_and_valid(self):

        '''init agent.last_state'''
        state,market_state,A = self.train_environment.reset()
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        market_state = torch.tensor(market_state,dtype=torch.float32,device=self.device)
        A = torch.tensor(A,dtype=torch.float32,device=self.device)
        assert state.shape == (self.batch_size, self.action_dim, self.state_dim, self.timesteps)
        assert isinstance(state, torch.Tensor)
        # if self.num_envs == 1:
        #     assert state.shape == (self.action_dim, self.state_dim, self.timesteps)
        #     assert isinstance(state, np.ndarray)
        #     state = torch.tensor(state, dtype=torch.float32, device=self.device)
        # else:
        #     assert state.shape == (self.num_envs, self.state_dim, self.timesteps)
        #     assert isinstance(state, torch.Tensor)
        #     state = state.to(self.device)
        # assert state.shape == (self.num_envs, self.action_dim, self.state_dim, self.timesteps)
        # assert isinstance(state, torch.Tensor)
        self.agent.last_state = state.detach()
        self.agent.market_last_state = market_state.detach()
        self.agent.A = A.detach()

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

        valid_score_list = []
        save_dict_list = []
        for epoch in range(1, self.epochs+1):

            print("Train Episode: [{}/{}]".format(epoch, self.epochs))

            count = 0
            s,s_m,A = self.train_environment.reset()

            episode_reward_sum = []
            episode_loss = 0
            steps_asu_grad = []
            steps_log_p_rho = []
            steps_reward_total = []
            while True:
                old_asset_state = torch.from_numpy(s).float().to(self.device)
                old_market_state = torch.from_numpy(s_m).float().to(self.device)
                corr_matrix_old = torch.from_numpy(A).float().to(self.device)
                with autocast():
                    weights, roh_log_p, action_asset, action_market = self.agent.compute_weights_train(
                        old_asset_state, old_market_state, corr_matrix_old)
                    s, s_m, A, asu_grad_, reward, done, save_dict = self.train_environment.step(weights)
                    asu_grad = torch.sum(
                        torch.from_numpy(asu_grad_).to(self.device) * torch.softmax(action_asset, axis=1), axis=1)
                    steps_asu_grad.append(asu_grad)
                    steps_log_p_rho.append(roh_log_p)
                    steps_reward_total.append(reward)
                    episode_reward_sum.append(reward)

                # action_asset = self.agent.act(
                #     torch.from_numpy(old_asset_state).float().to(self.device),
                #     corr_matrix_old)
                # action_asset = self.agent.act_net(
                #     torch.from_numpy(old_asset_state).float().to(self.device),
                #     corr_matrix_old)
                # action_market = self.agent.market_net(old_market_state)
                # action_market = self.agent.market(old_market_state)


                new_asset_state = s
                new_market_state = torch.from_numpy(s_m).float().to(self.device)
                corr_matrix_new = torch.from_numpy(A).float().to(self.device)
                # self.agent.store_transition(
                #     torch.from_numpy(old_asset_state).float().to(self.device),
                #     asu_grad,
                #     torch.tensor(reward).float().to(self.device),
                #     torch.from_numpy(new_asset_state).float().to(self.device),
                #     old_market_state, action_market, new_market_state,
                #     corr_matrix_old, corr_matrix_new, roh_log_p)
                count = count + 1
                if done.any():
                    steps_log_p_rho = torch.stack(steps_log_p_rho, dim=-1)
                    gradient_asu = torch.stack(steps_asu_grad, dim=-1)
                    steps_reward_total = np.array(steps_reward_total).transpose((1, 0))
                    rewards_total = torch.from_numpy(steps_reward_total).to(self.device)
                    # rewards_total = torch.stack(steps_reward_total, dim=-1)
                    # steps_reward_total = np.array(steps_reward_total).transpose((1, 0))



                    rewards_totals = (rewards_total - torch.mean(rewards_total, dim=-1, keepdim=True)) \
                                     / torch.std(rewards_total, dim=-1, keepdim=True)





                    gradient_rho = (rewards_totals * steps_log_p_rho)
                    loss = - (self.gamma * gradient_rho + gradient_asu)
                    loss = loss.mean()
                    assert not torch.isnan(loss)

                    self.agent.network_optimizer.zero_grad()
                    self.scaler.scale(loss).backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(
                        list(self.agent.act.parameters()) + list(self.agent.market.parameters()), max_norm=1.0)
                    self.scaler.step(self.agent.network_optimizer)
                    self.scaler.update()
                    episode_loss += loss

                    # if count % 100 == 10:
                    #     self.agent.learn()
                    print("Train Episode Reward Sum: {:04f}".format(np.mean(np.array(episode_reward_sum))))
                    print("Train Episode Loss Sum: {:04f}".format(episode_loss))
                    break


            save_model(self.checkpoints_path,
                       epoch=epoch,
                       save=self.agent.get_save())

            print("Valid Episode: [{}/{}]".format(epoch, self.epochs))
            s,s_m,A = self.valid_environment.reset()
            episode_reward_sum = 0
            while True:
                old_asset_state = torch.from_numpy(s).float().to(self.device)
                old_market_state = torch.from_numpy(s_m).float().to(self.device)
                corr_matrix_old = torch.from_numpy(A).float().to(self.device)
                weights = self.agent.compute_weights_test(old_asset_state,old_market_state,corr_matrix_old)
                s,s_m,A,asu_grad_, reward, done, save_dict = self.valid_environment.step(weights)
                episode_reward_sum += reward
                if done:
                    print("Valid Episode Reward Sum: {:04f}".format(episode_reward_sum.item()))
                    break
            valid_score_list.append(episode_reward_sum.item())
            save_dict_list.append(save_dict)
            if torch.cuda.is_available():
                print(torch.cuda.memory_summary(device=self.device))
        max_index = np.argmax(valid_score_list)
        plot_metric_against_baseline(total_asset=save_dict_list[max_index]['total_assets'],
                                     buy_and_hold=None, alg='Deeptrader',
                                     task='valid', color='darkcyan', save_dir=self.work_dir)
        load_model(self.checkpoints_path,
                   epoch=max_index + 1,
                   save=self.agent.get_save())
        save_best_model(
            output_dir=self.checkpoints_path,
            epoch=max_index + 1,
            save=self.agent.get_save()
        )

    def test(self):
        load_best_model(self.checkpoints_path, save=self.agent.get_save(), is_train=False)

        print("Test Best Episode")
        s, s_m, A = self.test_environment.reset()

        episode_reward_sum = 0
        while True:
            old_asset_state = torch.from_numpy(s).float().to(self.device)
            old_market_state = torch.from_numpy(s_m).float().to(self.device)
            corr_matrix_old = torch.from_numpy(A).float().to(self.device)
            weights = self.agent.compute_weights_test(old_asset_state, old_market_state,corr_matrix_old)
            s,s_m,A,asu_grad_, reward, done, save_dict = self.test_environment.step(weights)
            episode_reward_sum += reward
            if done:
                plot_metric_against_baseline(total_asset=save_dict['total_assets'],
                                             buy_and_hold=None, alg='Deeptrader',
                                             task='test', color='darkcyan', save_dir=self.work_dir)
                # print("Test Best Episode Reward Sum: {:04f}".format(episode_reward_sum))
                break

        df_return = self.test_environment.save_portfolio_return_memory()
        df_assets = self.test_environment.save_asset_memory()
        assets = df_assets["total assets"].values
        daily_return = df_return.daily_return.values
        df = pd.DataFrame()
        df["daily_return"] = daily_return
        df["total assets"] = assets
        df.to_csv(os.path.join(self.work_dir, "test_result.csv"), index=False)
        daily_return = df.daily_return.values
        # return daily_return