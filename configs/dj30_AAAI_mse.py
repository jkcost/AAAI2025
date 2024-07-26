import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from configs.common_config  import common_params


# task_name = "portfolio_management"
dataset_name = "dj30"
net_name = "AAAI"
agent_name = "AAAI"
optimizer_name = "adam"
loss_name = "mse"
work_dir = f"work_dir/{dataset_name}_{net_name}_{agent_name}_{optimizer_name}_{loss_name}"


batch_size = 64
wandb_project_name =common_params['wandb_project_name'],
wandb_group_name =common_params['wandb_group_name'],
wandb_session_name =common_params['wandb_session_name'],
gpu_ids = common_params['gpu_ids']
data = dict(
    type='AAAI_mse_Dataset',
    data_path='data/dj30',
    train_path='train.csv',
    valid_path='valid.csv',
    test_path='test.csv',
    test_dynamic_path='test_with_label.csv',
    tech_indicator_list= common_params['tech_indicator_list'],
    size=[common_params['seq_len'], common_params['label_len'], common_params['pred_len']],
    features = 'MS',
    scale = True,
    timeenc = 1,
    freq = 'D',
    length_day=common_params['length_day'],
    timesteps=common_params['timesteps'],
    initial_amount=common_params['initial_amount'],
    transaction_cost_pct=common_params['transaction_cost_pct'])

environment = dict(type='AAAIEnvironment')

transition = dict(
    type = "Transition"
)
agent = dict(
    type='AAAI',
    memory_capacity=1000,
    gamma=0.99,
    policy_update_frequency=500,timesteps=5)

trainer = dict(
    type='AAAI_mse',
    pred_len = common_params['pred_len'],
    epochs=common_params['num_epochs'],
    gamma = 0.05,
    work_dir=work_dir,
    if_remove=False,
    wandb_project_name =common_params['wandb_project_name'],
    wandb_group_name =common_params['wandb_group_name'],
    wandb_session_name =common_params['wandb_session_name'],
    temperature = common_params['temperature'])

loss = dict(type='MSELoss')
optimizer = dict(type='Adam', lr=common_params['lr'])

act_net = dict(
    type='AAAI_mse',
    attention_bool='True',
    model = common_params['model'],
    dropout = 0.05,
    num_stocks = 29,
    seq_len = common_params['seq_len'],
    label_len = common_params['label_len'],
    pred_len = common_params['pred_len'],
    modes = 64,
    enc_in = 17,
    dec_in = 17,
    c_out = 17,
    d_model = common_params['d_model'],
    n_heads = 4,
    e_layers = common_params['e_layers'],
    d_layers = common_params['d_layers'],
    output_attention = True,
    embed= 'timeF',
    freq = 'd',
    factor = 1,
    d_ff = 512,
    activation = 'gelu',
    use_norm = True,
    moving_avg = [24],
    version = 'Fourier',
    mode_select = 'ranodm') # for FEDformer








