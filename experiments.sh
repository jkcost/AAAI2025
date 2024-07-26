#!/bin/bash
#'Flashformer' 'iFlashformer' 'Flowformer' 'iFlowformer'
check_gpu_availability() {
    while true; do
        for gpu_id in {0..7}; do
            gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits --id=$gpu_id | awk '{print $1}')
            memory_util=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=$gpu_id | awk '{print $1}')
            memory_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits --id=$gpu_id | awk '{print $1}')
            memory_ratio=$(echo "scale=2; $memory_util / $memory_total * 100" | bc)

            if [[ $gpu_util -lt 60 && $memory_ratio < 60 ]]; then
                echo $gpu_id
                return
            fi
        done
        sleep 300
    done
}

var=0

for experiment_count in {1..10}; do
    for VAR1 in 0.05 0.1 0.5 1 ; do
        for VAR2 in IXIC DJIA SNP; do
            for VAR3 in  'Transformer' 'iTransformer' 'Flashformer' 'iFlashformer'  ; do
                for VAR4 in 36 ; do
                    for VAR5 in 0.05; do
                        for VAR6 in 20; do
                            for VAR7 in  500 ; do
                                gpu=$(check_gpu_availability)
                                CUDA_VISIBLE_DEVICES=$gpu python -W ignore run.py \
                                    --temperature $VAR1 \
                                    --market $VAR2 \
                                    --tr_model $VAR3 \
                                    --batch_size $VAR4 \
                                    --dropout $VAR5 \
                                    --G $VAR6 \
                                    --r $VAR7 \
                                    --no_lora \
                                    --wandb_project_name "2024_CIKM_DeepClair_Transformer_nomsu_v7" \
                                    --wandb_group_name "preliminary_hyper_search" \
                                    --wandb_session_name "setting_${var}_exp${experiment_count}" &
                                var=$((var + 1))
                                sleep 10
                            done
                        done
                    done
                done
            done
        done
    done
done
