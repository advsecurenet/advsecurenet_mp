#!/bin/bash

export WANDB_MODE=disabled

# Optionally set the cache for transformers
# export TRANSFORMERS_CACHE='YOUR_PATH/huggingface'

export model=$1
export n=$2
export insert_loc=$3
export w_tar_1=$4
export w_tar_2=$5
export ctrl_temp=$6
export ctrl_prefix=${7:-vicuna}
export ctrl_suffix=${8:-empty}
export budget=${9:-3}
export n_sample_goals=${10:-0}  # randomly sample goals in optimizing each token

# Create results folder if it doesn't exist
if [ ! -d "../results" ]; then
    mkdir "../results"
    echo "Folder '../results' created."
else
    echo "Folder '../results' already exists."
fi

for data_offset in 0  # 10 20 30 40 50 60 70 80 90
do

    python -u ../main.py \
        --config="../configs/individual_${model}.py" \
        --config.attack=gcg \
        --config.train_data="../../data/advbench/harmful_behaviors.csv" \
        --config.result_prefix="../results/individual_${model}_${n}_${insert_loc}_${w_tar_1}_${w_tar_2}_${temp}" \
        --config.n_train_data=$n \
        --config.data_offset=$data_offset \
        --config.n_test_data=25 \
        --config.test_offset=25 \
        --config.n_steps=1000 \
        --config.test_steps=50 \
        --config.batch_size=512 \
        --config.topk=512 \
        --config.w_tar_1="${w_tar_1}" \
        --config.w_tar_2="${w_tar_2}" \
        --config.ctrl_temp="${ctrl_temp}" \
        --config.insert_loc="${insert_loc}" \
        --config.ctrl_suffix="${ctrl_suffix}" \
        --config.ctrl_prefix="${ctrl_prefix}" \
        --config.budget="${budget}"

done
