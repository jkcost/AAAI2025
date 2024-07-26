import sys
import os
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])
sys.path.append(ROOT)

from builder import AGENTS
# from custom import AgentBase
from utils import get_attr
import torch
from torch.distributions import Normal
import random
import pandas as pd
import numpy as np
from collections import namedtuple
from torch import Tensor
from typing import Tuple
from torch.cuda.amp import autocast, GradScaler




def generate_portfolio(scores=torch.sigmoid(torch.randn(29, 1)), quantile=0.5):
    scores = scores.squeeze()
    length = scores.shape[-1]
    if scores.equal(torch.ones(length)):
        weights = (1 / length) * torch.ones(length)
        return weights
    if scores.equal(torch.zeros(length)):
        weights = (-1 / length) * torch.ones(length)
        return weights
    sorted_score, indices = torch.sort(scores, descending=True)

    value_hold = sorted_score[..., -1] + (sorted_score[..., 0] - sorted_score[..., -1]) * quantile
    if len(value_hold.shape) == 0:
        good_mask = scores > value_hold
    else:
        good_mask = scores > value_hold.unsqueeze(1)
    # good_mask = scores > value_hold
    bad_mask = ~good_mask
    good_scores = scores * good_mask.float()
    bad_scores = scores * bad_mask.float()
    exp_good_scores = torch.exp(good_scores)
    exp_bad_scores = torch.exp(1 - bad_scores)

    sum_exp_good_scores = torch.sum(exp_good_scores, dim=-1, keepdim=True)
    sum_exp_bad_scores = torch.sum(exp_bad_scores, dim=-1, keepdim=True)
    if len(quantile.shape) == 0:
        quantile_tensor = quantile
        inverse_quantile_tensor = (1 - quantile)
    else:
        quantile_tensor = quantile.unsqueeze(-1).expand(scores.shape[0], scores.shape[-1])
        inverse_quantile_tensor = (1 - quantile).unsqueeze(-1).expand(scores.shape[0], scores.shape[-1])

    good_portion = exp_good_scores / sum_exp_good_scores * quantile_tensor
    bad_portion = -exp_bad_scores / sum_exp_bad_scores * inverse_quantile_tensor

    # Combine good and bad portions
    final_portfolio = good_portion + bad_portion

    return final_portfolio


def generate_rho(mean: torch.tensor, std: torch.tensor):
    std = torch.log(1 + torch.exp(std))
    normal = Normal(mean, std)
    result = normal.sample()
    for i in range(result.size(0)):
        if result[i] <= 0:
            result[i] = torch.tensor(0)
        if result[i] >= 1:
            result[i] = torch.tensor(0.99)
    rho_log_p = normal.log_prob(result)
    return result, rho_log_p


@AGENTS.register_module()
class AAAI():
    def __init__(self, **kwargs):
        super(AAAI, self).__init__()
        self.scaler = GradScaler()
        self.num_envs = int(get_attr(kwargs, "num_envs", 1))
        self.device = get_attr(kwargs, "device", None)

        self.act = get_attr(kwargs, "act", None).to(self.device)
        # self.cri = get_attr(kwargs, "cri", None).to(self.device)


        self.act_optimizer = get_attr(kwargs, "act_optimizer", None)
        # self.cri_optimizer = get_attr(kwargs, "cri_optimizer", None)

        self.network_optimizer = get_attr(kwargs, "network_optimizer", None)
        self.timesteps = get_attr(kwargs, "timesteps", 10)

        self.criterion = get_attr(kwargs, "criterion", None)
        self.transition = get_attr(kwargs, "transition",
                                   namedtuple("TransitionDeepTrader",
                                              ['state',
                                               'action',
                                               'reward',
                                               'undone',
                                               'next_state',
                                               'correlation_matrix',
                                               'next_correlation_matrix',
                                               'state_market',
                                               'next_state_market',
                                               'a_market',
                                               'roh_bar_market'
                                               ]))

        self.action_dim = get_attr(kwargs, "action_dim", None)
        self.state_dim = get_attr(kwargs, "state_dim", None)

        self.memory_counter = 0  # for storing memory
        self.memory_capacity = get_attr(kwargs, "memory_capacity", 1000)
        self.gamma = get_attr(kwargs, "gamma", 0.9)
        self.data = []

        self.policy_update_frequency = get_attr(kwargs, "policy_update_frequency", 500)
        self.critic_learn_time = 0
        self.s_memory_asset = []
        self.a_memory_asset = []
        self.r_memory_asset = []
        self.sn_memory_asset = []
        self.correlation_matrix = []
        self.correlation_n_matrix = []
        self.s_memory_market = []
        self.a_memory_market = []
        self.sn_memory_market = []
        self.roh_bars = []

    def put_data(self,item):
        self.data.append(item)

    def train_net(self):
        R = 0
        self.network_optimizer.zero_grad()

        loss_list = []

        for reward, log_prob in self.data[::-1]:
            R = reward + self.gamma * R
            loss = -log_prob * R
            loss_list.append(loss)

        with autocast():
            total_loss = torch.stack(loss_list).sum()

            self.scaler.scale(total_loss).backward()
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.act.parameters(), max_norm=1.0)
            self.scaler.step(self.network_optimizer)
            self.scaler.update()


        self.data = []
        return total_loss.item()

    def get_save(self):
        models = {
            "act": self.act,
            "market": self.market
        }
        optimizers = {
            "network_optimizer": self.network_optimizer
        }
        res = {
            "models": models,
            "optimizers": optimizers
        }
        return res

    def get_action(self, state, state_market, corr_matrix):
        asset_scores = self.act(state, corr_matrix)
        output_market = self.market(state_market)
        roh_bar, roh_log_p = generate_rho(output_market[0], output_market[1])
        weights = generate_portfolio(asset_scores, roh_bar)
        action = weights.numpy()
        return action

    def explore_env(self, env, horizon_len: int) -> Tuple[Tensor, ...]:

        # states = torch.zeros((horizon_len,
        #                       self.num_envs,
        #                       self.action_dim,
        #                       self.timesteps,
        #                       self.state_dim), dtype=torch.float32).to(self.device)
        states = torch.zeros((horizon_len,
                              self.num_envs,
                              self.action_dim,
                              self.state_dim,
                              self.timesteps,), dtype=torch.float32).to(self.device)
        # actions = torch.zeros((horizon_len, self.num_envs, self.action_dim + 1), dtype=torch.int32).to(self.device)  # different
        actions = torch.zeros((horizon_len, self.num_envs, self.action_dim), dtype=torch.int32).to(self.device)
        rewards = torch.zeros((horizon_len, self.num_envs), dtype=torch.float32).to(self.device)
        dones = torch.zeros((horizon_len, self.num_envs), dtype=torch.bool).to(self.device)
        # next_states = torch.zeros((horizon_len,
        #                            self.num_envs,
        #                            self.action_dim,
        #                            self.timesteps,
        #                            self.state_dim), dtype=torch.float32).to(self.device)
        next_states = torch.zeros((horizon_len,
                                   self.num_envs,
                                   self.action_dim,
                                   self.state_dim,
                                   self.timesteps), dtype=torch.float32).to(self.device)
        correlation_matrixs = torch.zeros((
            horizon_len, self.num_envs, self.action_dim, self.action_dim
        ), dtype=torch.float32).to(self.device)
        next_correlation_matrixs = torch.zeros((
            horizon_len, self.num_envs, self.action_dim, self.action_dim
        ), dtype=torch.float32).to(self.device)
        state_markets = torch.zeros((horizon_len,
                                     self.num_envs,
                                     self.timesteps,
                                     self.state_dim), dtype=torch.float32).to(self.device)
        roh_bar_markets = torch.zeros((horizon_len, self.num_envs), dtype=torch.float32).to(self.device)

        state = self.last_state  # last_state.shape = (state_dim, ) for a single env.

        get_action = self.get_action

        for t in range(horizon_len):
            states[t] = state
            market_state = torch.from_numpy(make_market_information(env.data,
                                                                    technical_indicator=env.tech_indicator_list)).unsqueeze(
                0).float().to(self.device)
            # corr_matrix = make_correlation_information(env.data)
            corr_matrix = torch.from_numpy(make_correlation_information(env.data)).float().to(self.device)
            action = get_action(state, market_state, corr_matrix)
            # print(action.shape)
            # exit()
            # action = get_action(state.unsqueeze(0))

            # ary_action = action[0].detach().cpu().numpy()
            # ary_state, reward, done, _ = env.step(ary_action)  # next_state
            next_state, reward, done, _ = env.step(action)  # next_state
            state = torch.as_tensor(env.reset() if done else next_state, dtype=torch.float32, device=self.device)
            # actions[t] = action
            actions[t] = torch.tensor(action.reshape(self.num_envs, -1)).to(self.device)
            rewards[t] = torch.tensor(reward).to(self.device)
            dones[t] = torch.tensor(done).to(self.device)
            next_states[t] = state

        self.last_state = state

        # rewards *= self.reward_scale
        undones = 1.0 - dones.type(torch.float32)

        transition = self.transition(
            state=states,
            action=actions,
            reward=rewards,
            undone=undones,
            next_state=next_states
        )
        return transition

    def store_transition(self, s_asset,
                         a_asset,
                         r,
                         sn_asset,
                         s_market,
                         a_market,
                         sn_market,
                         A,
                         A_n,
                         roh_bar):  # 定义记忆存储函数 (这里输入为两套transition：asset和market)

        self.memory_counter = self.memory_counter + 1
        if self.memory_counter < self.memory_capacity:

            self.s_memory_asset.append(s_asset)
            self.a_memory_asset.append(a_asset)
            self.r_memory_asset.append(r)
            self.sn_memory_asset.append(sn_asset)
            self.correlation_matrix.append(A)
            self.correlation_n_matrix.append(A_n)
            self.s_memory_market.append(s_market)
            self.a_memory_market.append(a_market)
            self.sn_memory_market.append(sn_market)
            self.roh_bars.append(roh_bar)

        else:
            number = self.memory_counter % self.memory_capacity
            self.s_memory_asset[number - 1] = s_asset
            self.a_memory_asset[number - 1] = a_asset
            self.r_memory_asset[number - 1] = r
            self.sn_memory_asset[number - 1] = sn_asset
            self.correlation_matrix[number - 1] = A
            self.correlation_n_matrix[number - 1] = A_n

            self.s_memory_market[number - 1] = s_market
            self.a_memory_market[number - 1] = a_market
            # self.r_memory_market[number - 1] = r
            self.sn_memory_market[number - 1] = sn_market
            self.roh_bars[number - 1] = roh_bar

    def compute_weights_train(self, asset_state, market_state, A):
        # use the mean to compute roh
        asset_scores = self.act(asset_state, A)
        # input_market = torch.from_numpy(market_state).unsqueeze(0).to(
        #     torch.float32).to(self.device)
        output_market = self.market(market_state)
        roh_bar, roh_log_p = generate_rho(output_market[..., 0], output_market[..., 1])
        weights = generate_portfolio(asset_scores.cpu().detach(), roh_bar.cpu().detach())
        # weights = weights.detach().numpy()
        return weights, roh_log_p, asset_scores, output_market

    def compute_weights_test(self, asset_state, market_state, A):
        # use the mean to compute roh
        # asset_state = torch.from_numpy(asset_state).float().to(self.device)
        asset_scores = self.act(asset_state, A)
        # input_market = torch.from_numpy(market_state).unsqueeze(0).to(
        #     torch.float32).to(self.device)
        output_market = self.market(market_state)
        weights = generate_portfolio(asset_scores.cpu().detach(),
                                     output_market[..., 0].cpu().detach())
        weights = weights.detach().numpy()
        return weights

    def learn(self):
        length = len(self.s_memory_asset)
        out1 = random.sample(range(length), int(length / 10))
        # random sample
        s_learn_asset = []
        a_learn_asset = []
        r_learn_asset = []
        sn_learn_asset = []
        correlation_asset = []
        correlation_asset_n = []

        s_learn_market = []
        a_learn_market = []
        r_learn_market = []
        sn_learn_market = []
        roh_bar_market = []
        for number in out1:
            s_learn_asset.append(self.s_memory_asset[number])
            a_learn_asset.append(self.a_memory_asset[number])
            r_learn_asset.append(self.r_memory_asset[number])
            sn_learn_asset.append(self.sn_memory_asset[number])
            correlation_asset.append(self.correlation_matrix[number])
            correlation_asset_n.append(self.correlation_n_matrix[number])

            s_learn_market.append(self.s_memory_market[number])
            a_learn_market.append(self.a_memory_market[number])
            # r_learn_market.append(self.r_memory_market[number])
            sn_learn_market.append(self.sn_memory_market[number])
            roh_bar_market.append(self.roh_bars[number])
        self.critic_learn_time = self.critic_learn_time + 1
        # update the asset unit
        loss_sum = 0
        for rho_log_p, grad_asu in zip(roh_bar_market, a_learn_asset):
            self.network_optimizer.zero_grad()

            loss = -(rho_log_p * self.gamma + grad_asu)
            loss.backward(retain_graph=True)

            self.network_optimizer.step()
            loss_sum += loss
        print(f'step({length}) loss -> {loss_sum}')

        #
        # for bs, ba, br, bs_, correlation, correlation_n in zip(
        #         s_learn_asset, a_learn_asset, r_learn_asset, sn_learn_asset,
        #         correlation_asset, correlation_asset_n):
        #     # update actor
        #     a = self.act(bs, correlation)
        #     q = self.cri(bs, correlation, a)
        #     a_loss = -torch.mean(q)
        #     print('a_loss-> :',a_loss)
        #     self.act_optimizer.zero_grad()
        #     a_loss.backward(retain_graph=True)
        #     self.act_optimizer.step()
        #     # update critic
        #     a_ = self.act(bs_, correlation_n)
        #     q_ = self.cri(bs_, correlation_n, a_.detach())
        #     q_target = br + self.gamma * q_
        #     q_eval = self.cri(bs, correlation, ba.detach())
        #     # print(q_eval)
        #     # print(q_target)
        #     td_error = self.criterion(q_target.detach(), q_eval)
        #     print('td_error -> :',td_error)
        #     self.cri_optimizer.zero_grad()
        #     td_error.backward()
        #     self.cri_optimizer.step()
        # # update the asset unit
        # # 除了correlation以外都是tensor correlation是np.array 直接从make_correlation_information返回即可
        # loss_market = 0
        # for s, br, roh_bar in zip(s_learn_market, r_learn_asset,
        #                           roh_bar_market):
        #     output_market = self.market(s)
        #     normal = Normal(output_market[0], output_market[1])
        #     b_prob = -normal.log_prob(roh_bar)
        #
        #     loss_market += br * b_prob
        #
        # self.market_optimizer.zero_grad()
        # print('loss_market -> :',loss_market)
        # loss_market.backward()
        # self.market_optimizer.step()