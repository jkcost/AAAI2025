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

# _base_ = [
#     f"../_base_/datasets/{task_name}/{dataset_name}.py",
#     f"../_base_/environments/{task_name}/env.py",
#     f"../_base_/agents/{task_name}/{agent_name}.py",
#     f"../_base_/trainers/{task_name}/deeptrader_trainer.py",
#     f"../_base_/losses/{loss_name}.py",
#     f"../_base_/optimizers/{optimizer_name}.py",
#     f"../_base_/nets/{net_name}.py",
#     f"../_base_/transition/transition.py"
# ]
batch_size = 64
wandb_project_name =common_params['wandb_project_name'],
wandb_group_name =common_params['wandb_group_name'],
wandb_session_name =common_params['wandb_session_name'],
gpu_ids = common_params['gpu_ids']
data = dict(
    type='AAAIDataset',
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
    type='AAAI_reinforce',
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
    type='AAAI',
    attention_bool='True',
    model = common_params['model'],
    dropout = 0.05,
    num_stocks = 29,
    seq_len = common_params['seq_len'],
    label_len = common_params['label_len'],
    pred_len = common_params['pred_len'],
    modes = 64,
    enc_in = 16,
    dec_in = 16,
    c_out = 16,
    d_model = 256,
    n_heads = 4,
    e_layers = 2,
    d_layers = 1,
    output_attention = True,
    embed= 'timeF',
    freq = 'd',
    factor = 1,
    d_ff = 512,
    activation = 'gelu',
    use_norm = True)








