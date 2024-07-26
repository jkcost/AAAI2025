from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = str(Path(__file__).resolve().parents[2])
sys.path.append(ROOT)
import numpy as np
from utils import get_attr, print_metrics
import pandas as pd
# from ..custom import Environments
from builder import ENVIRONMENTS
import gymnasium as gym
from collections import OrderedDict


@ENVIRONMENTS.register_module()
class DeepTraderEnvironment():
    def __init__(self, **kwargs):
        super(DeepTraderEnvironment, self).__init__()

        self.dataset = get_attr(kwargs, "dataset", None)
        self.task = get_attr(kwargs, "task", "train")
        self.batch_size = get_attr(kwargs,'batch_size',1)
        timesteps = get_attr(self.dataset, "timesteps", 10)
        self.trade_len = get_attr(kwargs,'trade_len',1)
        self.day = timesteps - 1

        self.df_path = None
        if self.task.startswith("train"):
            self.df_path = get_attr(self.dataset, "train_path", None)
        elif self.task.startswith("valid"):
            self.df_path = get_attr(self.dataset, "valid_path", None)
        else:
            self.df_path = get_attr(self.dataset, "test_path", None)

        self.initial_amount = get_attr(self.dataset, "initial_amount", 100000)
        self.transaction_cost_pct = get_attr(self.dataset, "transaction_cost_pct", 0.001)
        self.tech_indicator_list = get_attr(self.dataset, "tech_indicator_list", [])

        if self.task.startswith("test_dynamic"):
            dynamics_test_path = get_attr(kwargs, "dynamics_test_path", None)
            self.df = pd.read_csv(dynamics_test_path, index_col=0)
            self.start_date = self.df.loc[:, 'date'].iloc[0]
            self.end_date = self.df.loc[:, 'date'].iloc[-1]
        else:
            self.df = pd.read_csv(self.df_path, index_col=0)
        self.stock_dim = len(self.df.tic.unique())
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

        self.data = self.df.loc[self.day - self.timesteps + 1:self.day, :]
        self.state = np.array([[
            self.data[self.data.tic == tic][tech].values.tolist()
            for tech in self.tech_indicator_list
        ] for tic in self.data.tic.unique()])
        self.terminal = False
        self.portfolio_value = self.initial_amount
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        self.weights_memory = [(np.array([[1 / (self.stock_dim)] * (self.stock_dim)]*self.batch_size))]
        self.date_memory = [self.data.date.unique()[0]]
        self.transaction_cost_memory = []

    def get_data(self,cursor):


        tmp_states = []
        tmp_market_states = []

        for i, idx in enumerate(cursor):
            start_idx = idx -self.timesteps +1
            end_idx = idx
            self.data = self.df.loc[start_idx:end_idx,:]
            tmp_state = np.array([[self.data[self.data.tic == tic][tech].values.tolist() for tech in self.tech_indicator_list] for tic in self.data.tic.unique()])
            tmp_states.append(tmp_state)

            information_lst = self.df.loc[start_idx:end_idx,:].index.unique().tolist()
            information_lst.sort()

            tmp_market_state = []
            for index in information_lst:
                information = self.df[self.df.index == index]
                tmp_tech_market_lst = []
                for tech in self.tech_indicator_list:
                    tech_value = np.mean(information[tech])
                    tmp_tech_market_lst.append(tech_value)
                tmp_market_state.append(tmp_tech_market_lst)
            tmp_market_states.append(tmp_market_state)



        state = np.array(tmp_states)
        market_state = np.array(tmp_market_states)
        A = self.make_correlation_information(cursor)
        return state,market_state,A

    def make_correlation_information(self, cursor, feature="adjcp"):
        all_correlation_matrices = []

        for idx in cursor:
            start_idx = idx - self.timesteps + 1
            end_idx = idx
            self.data = self.df.loc[start_idx:end_idx, :]

            # 상관 관계 정보를 만들기 위한 데이터 프레임 정렬 및 준비
            self.data.sort_values(by='tic', ascending=True, inplace=True)
            array_symbols = self.data['tic'].values

            # 데이터 수집 및 딕셔너리에 넣기
            dict_sym_ac = {}
            for sym in array_symbols:
                dftemp = self.data[self.data['tic'] == sym]
                dict_sym_ac[sym] = dftemp['adjcp'].values

            # 상관 관계 계수 데이터 프레임 생성
            dfdata = pd.DataFrame.from_dict(dict_sym_ac)
            dfcc = dfdata.corr().round(2)
            correlation_matrix = dfcc.values

            all_correlation_matrices.append(correlation_matrix)

        return np.array(all_correlation_matrices)

    def reset(self):
        self.day = self.timesteps - 1
        if self.task =='train':
            # valid_indices = self.df.index.values[self.df.index.values >=self.day]
            valid_indices = np.arange(self.day,self.df.index[-1] -self.trade_len)
            self.cursor = np.random.permutation(valid_indices)[:self.batch_size]
        else:
            self.cursor = np.array([self.day])
        state,market_state,A = self.get_data(self.cursor)

        # self.data = self.df.loc[self.day - self.timesteps + 1:self.day, :]
        # # initially, the self.state's shape is stock_dim*len(tech_indicator_list)
        # self.state = np.array([[
        #     self.data[self.data.tic == tic][tech].values.tolist()
        #     for tech in self.tech_indicator_list
        # ] for tic in self.data.tic.unique()])
        # self.state = np.transpose(self.state, (2, 0, 1))
        self.terminal = False
        self.portfolio_value = self.initial_amount
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        self.weights_memory = [(np.array([[1 / (self.stock_dim)] * (self.stock_dim)]*self.batch_size))]
        self.date_memory = [self.data.date.unique()[0]]
        self.transaction_cost_memory = []

        return state,market_state,A

    def step(self, weights):
        # make judgement about whether our data is running out
        self.terminal =  self.cursor + self.trade_len >= self.df.index[-1] - 1
        weights = np.array(weights)


        if self.terminal.any() and self.task != 'train':
            if self.task.startswith("test_dynamic"):
                print(f'Date from {self.start_date} to {self.end_date}')
            tr, sharpe_ratio, vol, mdd, cr, sor = self.analysis_result()
            df_value = self.save_asset_memory()
            assets = df_value["total assets"].values
            stats = OrderedDict(
                {
                    "Total Return": ["{:04f}%".format((tr * 100).item())],
                    "Sharp Ratio": ["{:04f}".format(sharpe_ratio)],
                    "Volatility": ["{:04f}%".format(vol * 100)],
                    "Max Drawdown": ["{:04f}%".format((mdd * 100).item())],
                    "Calmar Ratio": ["{:04f}".format(cr.item())],
                    "Sortino Ratio": ["{:04f}".format(sor.item())],
                }
            )
            table = print_metrics(stats)
            print(table)
            return self.state,0, 0,0,0, self.terminal, {"sharpe_ratio": sharpe_ratio, 'total_assets': assets}

        else:  # directly use the process of
            self.weights_memory.append(weights)
            last_day_memory = self.df.loc[self.cursor, :]
            self.day += self.trade_len
            self.cursor += self.trade_len
            state, market_state, A = self.get_data(self.cursor)

            # self.data = self.df.loc[self.day - self.timesteps + 1:self.day, :]
            # self.state = np.array([[
            #     self.data[self.data.tic == tic][tech].values.tolist()
            #     for tech in self.tech_indicator_list
            # ] for tic in self.data.tic.unique()])
            # self.state = np.transpose(self.state, (2, 0, 1))
            new_price_memory = self.df.loc[self.cursor, :]
            portfolio_weights = weights
            if len(weights.shape) ==1:
                asu_grad_return = ((new_price_memory.close.values / last_day_memory.close.values) - 1)
                change_ratio = (new_price_memory.close.values / last_day_memory.close.values)
            else:
                asu_grad_return = ((new_price_memory.close.values / last_day_memory.close.values) - 1).reshape(weights.shape[0],-1)
                change_ratio = (new_price_memory.close.values / last_day_memory.close.values).reshape(weights.shape[0],
                                                                                                      -1)

            portfolio_return = np.sum(asu_grad_return * portfolio_weights, axis=-1)
            weights_brandnew = self.normalization(weights * change_ratio)
            self.weights_memory.append(weights_brandnew)

            weights_old = (self.weights_memory[-3])
            weights_new = (self.weights_memory[-2])
            diff_weights = np.sum(np.abs(weights_old - weights_new),axis=-1)
            transcationfee = diff_weights * self.transaction_cost_pct * self.portfolio_value
            new_portfolio_value = (self.portfolio_value -
                                   transcationfee) * (1 + portfolio_return)
            portfolio_return = (new_portfolio_value - self.portfolio_value) / self.portfolio_value
            self.reward = new_portfolio_value - self.portfolio_value
            self.portfolio_value = new_portfolio_value

            self.portfolio_return_memory.append(portfolio_return)
            self.date_memory.append(self.data.date.unique()[-1])
            self.asset_memory.append(new_portfolio_value)

            self.reward = self.reward

            return state,market_state,A,asu_grad_return, self.reward, self.terminal, {}

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
        daily_return = df["daily_return"]
        # print(df, df.shape, len(df),len(daily_return))
        neg_ret_lst = df[df["daily_return"] < 0]["daily_return"]
        neg_ret_lst = neg_ret_lst.apply(lambda x:x.item())
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
        return tr, sharpe_ratio, vol, mdd, cr, sor