#!/bin/bash

# Function to check GPU availability
check_gpu_availability() {
    while true; do
        for gpu_id in {0..7}; do
            gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits --id=$gpu_id | awk '{print $1}')
            memory_util=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=$gpu_id | awk '{print $1}')
            memory_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits --id=$gpu_id | awk '{print $1}')
            memory_ratio=$(echo "scale=2; $memory_util / $memory_total * 100" | bc)

            if [[ $gpu_util -lt 60 && $(echo "$memory_ratio < 60" | bc) -eq 1 ]]; then
                echo $gpu_id
                return
            fi
        done
        sleep 300
    done
}

# Define variables
experiment_count=1
learning_rates=(0.0001 0.00001 0.000001)
markets=("DJIA")
models=("Transformer" "iTransformer")
batch_sizes=(36)
dropouts=(0.05)

# Loop through each combination
for lr in "${learning_rates[@]}"; do
    for market in "${markets[@]}"; do
        for model in "${models[@]}"; do
            for batch_size in "${batch_sizes[@]}"; do
                for dropout in "${dropouts[@]}"; do
                    # Create temporary common_params.py
                    cat <<EOL > configs/common_config/common_params.py
common_params = dict(
    initial_amount=100000,
    transaction_cost_pct=0.001,
    tech_indicator_list=[
        'high', 'low', 'open', 'close', 'adjcp', 'zopen', 'zhigh', 'zlow',
        'zadjcp', 'zclose', 'zd_5', 'zd_10', 'zd_15', 'zd_20', 'zd_25', 'zd_30'
    ],
    temperature=1,
    timesteps=5,
    batch_size=${batch_size},
    num_epochs=10,
    length_day=10,
    seq_len=72,
    label_len=5,
    pred_len=5,
    model='${model}',
    wandb_project_name='2025_AAAI_Exp',
    wandb_group_name='main_exp',
    wandb_session_name='setting_${experiment_count}',
)
EOL

                    # Check GPU availability and set CUDA device
                    gpu=$(check_gpu_availability)
                    CUDA_VISIBLE_DEVICES=$gpu python -W ignore run.py \
                        --config configs/dj30_AAAI.py \
                        --market $market &

                    # Increment experiment count
                    experiment_count=$((experiment_count + 1))
                    sleep 10
                done
            done
        done
    done
done
