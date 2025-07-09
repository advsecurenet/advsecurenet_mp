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
export budget=${9:-1}
export n_sample_goals=${10:-0}  # randomly sample goals in optimizing each token

# Create results folder if it doesn't exist
if [ ! -d "../results" ]; then
    mkdir "../results"
    echo "Folder '../results' created."
else
    echo "Folder '../results' already exists."
fi

# note on arguments:
# 1. randomly sample batch_size from topk if batch_size < topk
python -u ../main.py \
    --config="../configs/mixture_${model}.py" \
    --config.train_data="../../data/advbench/harmful_behaviors.csv" \
    --config.result_prefix="../results/multiple_${model}_${n}_${insert_loc}_${w_tar_1}_${w_tar_2}_${temp}" \
    --config.progressive_goals=False \
    --config.stop_on_success=True \
    --config.num_train_models=2 \
    --config.allow_non_ascii=True \
    --config.n_train_data=$n \
    --config.n_test_data=25 \
    --config.n_steps=1000 \
    --config.test_steps=25 \
    --config.test_offset=25 \
    --config.batch_size=512 \
    --config.topk=512 \
    --config.w_tar_1="${w_tar_1}" \
    --config.w_tar_2="${w_tar_2}" \
    --config.ctrl_temp="${ctrl_temp}" \
    --config.insert_loc="${insert_loc}" \
    --config.ctrl_suffix="${ctrl_suffix}" \
    --config.ctrl_prefix="${ctrl_prefix}" \
    --config.budget="${budget}" \
    --config.n_sample_goals="${n_sample_goals}" \
