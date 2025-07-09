#!/bin/bash

export WANDB_MODE=disabled

# Optionally set the cache for transformers
# export TRANSFORMERS_CACHE='YOUR_PATH/huggingface'

export logfile=$1
export model=$2
export n=25
export insert_loc=tail
export w_tar_1=0.4
export w_tar_2=100
export ctrl_temp=1
export n_sample_goals=0
export ctrl_prefix=${3:-empty}
export ctrl_suffix=${4:-empty}
export budget=${5:-1}

# Create results folder if it doesn't exist
if [ ! -d "../results" ]; then
    mkdir "../results"
    echo "Folder '../results' created."
else
    echo "Folder '../results' already exists."
fi

python -u ../evaluate.py \
    --config="../configs/transfer_${model}.py" \
    --config.eval_model=${model} \
    --config.train_data="../../data/advbench/harmful_behaviors.csv" \
    --config.result_prefix="../results/multiple_${model}_${n}_${insert_loc}_${w_tar_1}_${w_tar_2}_${temp}" \
    --config.progressive_goals=False \
    --config.stop_on_success=True \
    --config.num_train_models=1 \
    --config.allow_non_ascii=True \
    --config.n_train_data=$n \
    --config.n_test_data=25 \
    --config.test_offset=25 \
    --config.n_steps=1000 \
    --config.test_steps=25 \
    --config.batch_size=64 \
    --config.topk=512 \
    --config.w_tar_1="${w_tar_1}" \
    --config.w_tar_2="${w_tar_2}" \
    --config.ctrl_temp="${ctrl_temp}" \
    --config.insert_loc="${insert_loc}" \
    --config.ctrl_suffix="${ctrl_suffix}" \
    --config.ctrl_prefix="${ctrl_prefix}" \
    --config.budget="${budget}" \
    --config.n_sample_goals="${n_sample_goals}" \
    --config.logfile="${logfile}" \
