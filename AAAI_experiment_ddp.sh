#!/bin/bash

# Define variables
experiment_count=1
learning_rates=(0.0001 0.00001 0.000001)
markets=("DJIA")
models=("Transformer" "Informer"  "Flashformer" "Flowformer" "Reformer")
e_layers_values=(1 2 3 4)
d_layers_values=(1 2)
d_model_values=(128 256 512)
dropouts=(0.05)
num_repeats=10

# Function to run the experiment
run_experiment() {
    local lr=$1
    local market=$2
    local model=$3
    local e_layers=$4
    local d_layers=$5
    local d_model=$6
    local dropout=$7
    local repeat=$8
    local experiment_id=$9

    # Create temporary common_config.py
    cat <<EOL > configs/common_config.py
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
    seq_len=20,
    label_len=5,
    pred_len=5,
    model='${model}',
    wandb_project_name='2025_AAAI_Exp',
    wandb_group_name='main_exp',
    wandb_session_name='setting_${experiment_id}_repeat_${repeat}',
    gpu_ids=[0,1,2,3,4,5],
    lr=${lr},
    d_model=${d_model},
    e_layers=${e_layers},
    d_layers=${d_layers},
    norm_method='ticker' # ticker,date
)
EOL

    # Run the experiment
    python -W ignore run.py \
        --config configs/dj30_AAAI.py \
        --market $market
}

# Loop through each combination and repeat the experiment
for lr in "${learning_rates[@]}"; do
    for market in "${markets[@]}"; do
        for model in "${models[@]}"; do
            for e_layers in "${e_layers_values[@]}"; do
                for d_layers in "${d_layers_values[@]}"; do
                    for d_model in "${d_model_values[@]}"; do
                        for dropout in "${dropouts[@]}"; do
                            for repeat in $(seq 1 $num_repeats); do
                                run_experiment $lr $market $model $e_layers $d_layers $d_model $dropout $repeat $experiment_count
                                experiment_count=$((experiment_count + 1))
                                wait
                            done
                        done
                    done
                done
            done
        done
    done
done
wait  # Wait for all experiments to complete before exiting
