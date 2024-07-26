
# common_config.py

common_params = dict(
    initial_amount=100000,
    transaction_cost_pct=0.0,
    tech_indicator_list=[
        'high', 'low', 'open', 'close', 'adjcp', 'zopen', 'zhigh', 'zlow',
        'zadjcp', 'zclose', 'zd_5', 'zd_10', 'zd_15', 'zd_20', 'zd_25', 'zd_30'
    ],
    temperature=1,
    timesteps=5,
    batch_size=64,
    num_epochs=10,
    length_day=10,
    seq_len = 20,
    label_len = 5,
    pred_len = 5,
    model = 'FEDformer',
    wandb_project_name  = '2025_AAAI_Exp',
    wandb_group_name  = 'main_exp',
    wandb_session_name = 'exp_num',
    gpu_ids = [0,1,2,3,4,5],
    lr = 0.000001,
    d_model = 128,
    e_layers = 2,
    d_layers = 1,
    norm_method = 'ticker', # ticker,date


)
