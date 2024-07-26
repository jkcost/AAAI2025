# task_name = "portfolio_management"
dataset_name = "dj30"
net_name = "deeptrader"
agent_name = "deeptrader"
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
batch_size = 32

data = dict(
    type='PortfolioManagementDataset',
    data_path='data/dj30',
    train_path='data/dj30/train.csv',
    valid_path='data/dj30/valid.csv',
    test_path='data/dj30/test.csv',
    test_dynamic_path='data/dj30/test_with_label.csv',
    tech_indicator_list=[
        'high', 'low', 'open', 'close', 'adjcp', 'zopen', 'zhigh', 'zlow',
        'zadjcp', 'zclose', 'zd_5', 'zd_10', 'zd_15', 'zd_20', 'zd_25', 'zd_30'
    ],
    lenth_day = 10,
    timesteps=10,
    initial_amount=100000,
    transaction_cost_pct=0.001)

environment = dict(type='DeepTraderEnvironment')

transition = dict(
    type = "Transition"
)
agent = dict(
    type='DeepTrader',
    memory_capacity=1000,
    gamma=0.99,
    policy_update_frequency=500,timesteps=10)

trainer = dict(
    type='DeepTraderTrainer',
    epochs=10,
    gamma = 0.05,
    work_dir=work_dir,
    if_remove=False )

loss = dict(type='MSELoss')
optimizer = dict(type='Adam', lr=0.001)

act_net = dict(
    type='ASU',
    num_nodes=29,
    in_features = 16,
    hidden_dim =128,
    layers = 3)

cri_net = dict(
    type='ASU',
    num_nodes=29,
    in_features = 16,
    hidden_dim =128,
    layers = 3)

# cri_net = dict(
#     type='AssetScoringValueNet',
#     N=29,
#     K_l=10,
#     num_inputs=16,
#     num_channels=[12, 12, 12],
#     kernel_size=2,
#     dropout=0.2)
market_net = dict(type='MSU', in_features=16, window_len = 10,hidden_dim=12)