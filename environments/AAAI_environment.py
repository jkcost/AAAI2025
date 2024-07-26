from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = str(Path(__file__).resolve().parents[2])
sys.path.append(ROOT)
import numpy as np
from utils import get_attr, print_metrics,time_features
import pandas as pd
# from ..custom import Environments
from builder import ENVIRONMENTS
import gymnasium as gym
from collections import OrderedDict
from torch.utils.data import DataLoader,Sampler

class DailyBatchSamplerRandom(Sampler):
    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        # calculate number of samples in each batch
        self.daily_count = pd.Series(index=self.data_source.get_index()).groupby("datetime").size().values
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)  # calculate begin index of each batch
        self.daily_index[0] = 0

    def __iter__(self):
        if self.shuffle:
            index = np.arange(len(self.daily_count))
            np.random.shuffle(index)
            for i in index:
                yield np.arange(self.daily_index[i], self.daily_index[i] + self.daily_count[i])
        else:
            for idx, count in zip(self.daily_index, self.daily_count):
                yield np.arange(idx, idx + count)

    def __len__(self):
        return len(self.data_source)

@ENVIRONMENTS.register_module()
class AAAIEnvironment:
    def __init__(self, **kwargs):
        super(AAAIEnvironment, self).__init__()

        self.dataset = get_attr(kwargs, "dataset", None)
        self.task = get_attr(kwargs, "task", "train")
        self.batch_size = get_attr(kwargs, 'batch_size', 1)
        timesteps = get_attr(self.dataset, "timesteps", 10)
        self.trade_len = get_attr(kwargs, 'trade_len', 1)
        self.day = timesteps - 1



        self.initial_amount = get_attr(self.dataset, "initial_amount", 100000)
        self.transaction_cost_pct = get_attr(self.dataset, "transaction_cost_pct", 0.001)
        self.tech_indicator_list = get_attr(self.dataset, "tech_indicator_list", [])

        self.stock_dim = len(self.dataset.data.index.get_level_values('tic').unique())
        self.state_space_shape = self.stock_dim
        self.action_space_shape = self.stock_dim
        self.timesteps = timesteps

        self.action_space = gym.spaces.Box(low=-5,
                                           high=5,
                                           shape=(self.action_space_shape,))
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.tech_indicator_list),
                   self.state_space_shape,
                   self.timesteps))

        self.action_dim = self.action_space.shape[0]
        self.state_dim = self.observation_space.shape[0]


        self.terminal = False
        self.portfolio_value = self.initial_amount
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        self.weights_memory = [np.array([1 / (self.stock_dim)] * (self.stock_dim))]
        self.date_memory = [self.dataset.data.index.get_level_values('date').unique()[0]]
        self.transaction_cost_memory = []

    def reset(self):
        self.portfolio_value = self.initial_amount
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        self.weights_memory = [np.array([1 / (self.stock_dim)] * (self.stock_dim))]
        self.date_memory = [self.dataset.data.index.get_level_values('date').unique()[0]]
        self.transaction_cost_memory = []


    def step(self, weights,returns,date):
        # make judgement about whether our data is running ou
        weights = np.array(weights.detach().cpu())
        returns = np.array(returns.detach().cpu()).squeeze()
        self.weights_memory.append(weights)
        portfolio_return = np.sum(weights * returns)
        change_ratio = returns + 1
        weights_brandnew = self.normalization(weights * change_ratio)
        self.weights_memory.append(weights_brandnew)
        weights_old = (self.weights_memory[-3])
        weights_new = (self.weights_memory[-2])
        diff_weights = np.sum(np.abs(weights_old - weights_new), axis=-1)
        transcationfee = diff_weights * self.transaction_cost_pct * self.portfolio_value
        new_portfolio_value = (self.portfolio_value -transcationfee) * (1 + portfolio_return)
        portfolio_return = (new_portfolio_value - self.portfolio_value) / self.portfolio_value
        ###reward
        reward = portfolio_return

        self.portfolio_value = new_portfolio_value
        self.portfolio_return_memory.append(portfolio_return)
        self.date_memory.append(date[0])
        self.asset_memory.append(new_portfolio_value)

        return reward



    def normalization(self, actions):
        # a normalization function not only for actions to transfer into weights but also for the weights of the
        # portfolios whose prices have been changed through time
        sum = np.sum(actions, axis=-1, keepdims=True)
        actions = actions / sum
        return actions

    def save_portfolio_return_memory(self):
        # a record of return for each time stamp
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ['date']

        return_list = self.portfolio_return_memory
        df_return = pd.DataFrame(return_list)
        df_return.columns = ["daily_return"]
        df_return.index = df_date.date

        return df_return

    def save_asset_memory(self):
        # a record of asset values for each time stamp
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ['date']

        assets_list = self.asset_memory
        df_value = pd.DataFrame(assets_list)
        df_value.columns = ["total assets"]
        df_value.index = df_date.date

        return df_value

    def analysis_result(self):
        # A simpler API for the environment to analysis itself when coming to terminal
        df_return = self.save_portfolio_return_memory()
        daily_return = df_return.daily_return.values
        df_value = self.save_asset_memory()
        assets = df_value["total assets"].values
        df = pd.DataFrame()
        df['date'] = pd.to_datetime(df_return.index)
        df["daily_return"] = daily_return
        df["total assets"] = assets
        return self.evaualte(df)

    def get_daily_return_rate(self, price_list: list):
        return_rate_list = []
        for i in range(len(price_list) - 1):
            return_rate = (price_list[i + 1] / price_list[i]) - 1
            return_rate_list.append(return_rate)
        return return_rate_list

    def evaualte(self, df):
        start_date = df['date'].iloc[0]
        end_date = df['date'].iloc[-1]
        daily_return = df["daily_return"]
        # print(df, df.shape, len(df),len(daily_return))
        neg_ret_lst = df[df["daily_return"] < 0]["daily_return"]
        # neg_ret_lst = neg_ret_lst.apply(lambda x:x.item())
        tr = df["total assets"].values[-1] / (df["total assets"].values[0] + 1e-10) - 1
        return_rate_list = self.get_daily_return_rate(df["total assets"].values)

        sharpe_ratio = np.mean(return_rate_list) * (252) ** 0.5 / (np.std(return_rate_list) + 1e-10)
        vol = np.std(return_rate_list)
        mdd = 0
        peak = df["total assets"][0]
        for value in df["total assets"]:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > mdd:
                mdd = dd
        cr = np.sum(daily_return) / (mdd + 1e-10)
        sor = np.sum(daily_return) / (np.nan_to_num(np.std(neg_ret_lst), 0) + 1e-10) / (
                    np.sqrt(len(daily_return)) + 1e-10)
        return start_date,end_date,tr, sharpe_ratio, vol, mdd, cr, sor