#!/bin/bash

export WANDB_MODE=disabled

# Parse arguments
export experiment_type=$1  # individual, transfer, mixture, leak, evaluate
export model=$2
export n=$3
export insert_loc=${4:-tail}
export w_tar_1=${5:-0.4}
export w_tar_2=${6:-100}
export ctrl_temp=${7:-1}
export ctrl_prefix=${8:-vicuna}
export ctrl_suffix=${9:-empty}
export budget=${10:-1}
export n_sample_goals=${11:-0}

# Additional args for evaluation
export logfile=${12:-""}

# Create results folder if it doesn't exist
if [ ! -d "../results" ]; then
    mkdir "../results"
    echo "Folder '../results' created."
else
    echo "Folder '../results' already exists."
fi

# Set experiment-specific defaults
case $experiment_type in
    "individual")
        config_prefix="individual"
        attack="gcg"
        train_data="../../data/advbench/harmful_behaviors.csv"
        num_train_models=1
        budget=${budget:-3}
        test_steps=50
        script="../main.py"
        ;;
    "transfer")
        config_prefix="transfer"
        attack="autodan"
        train_data="../../data/advbench/harmful_behaviors.csv"
        num_train_models=1
        budget=${budget:-1}
        test_steps=25
        script="../main.py"
        ;;
    "mixture")
        config_prefix="mixture"
        attack="autodan"
        train_data="../../data/advbench/harmful_behaviors.csv"
        num_train_models=2
        budget=${budget:-1}
        test_steps=25
        script="../main.py"
        ;;
    "leak")
        config_prefix="transfer"
        attack="autodan"
        train_data="../../data/prompt_leaking/aws_prompts.csv"
        num_train_models=1
        budget=${budget:-1}
        test_steps=10
        script="../main.py"
        ;;
    "evaluate")
        config_prefix="transfer"
        attack="autodan"
        train_data="../../data/advbench/harmful_behaviors.csv"
        num_train_models=1
        budget=${budget:-1}
        test_steps=25
        script="../evaluate.py"
        n=25
        batch_size=64
        ;;
    *)
        echo "Unknown experiment type: $experiment_type"
        echo "Usage: $0 <experiment_type> <model> <n> [other_args...]"
        echo "Experiment types: individual, transfer, mixture, leak, evaluate"
        exit 1
        ;;
esac

# Set batch size based on experiment type
batch_size=${batch_size:-512}

# Common arguments
common_args=(
    --config="../configs/${config_prefix}_${model}.py"
    --config.train_data="$train_data"
    --config.result_prefix="../results/${experiment_type}_${model}_${n}_${insert_loc}_${w_tar_1}_${w_tar_2}_${ctrl_temp}"
    --config.progressive_goals=False
    --config.stop_on_success=True
    --config.num_train_models=$num_train_models
    --config.allow_non_ascii=True
    --config.n_train_data=$n
    --config.n_test_data=25
    --config.test_offset=25
    --config.n_steps=1000
    --config.test_steps=$test_steps
    --config.batch_size=$batch_size
    --config.topk=512
    --config.w_tar_1="$w_tar_1"
    --config.w_tar_2="$w_tar_2"
    --config.ctrl_temp="$ctrl_temp"
    --config.insert_loc="$insert_loc"
    --config.ctrl_suffix="$ctrl_suffix"
    --config.ctrl_prefix="$ctrl_prefix"
    --config.budget="$budget"
    --config.n_sample_goals="$n_sample_goals"
)

# Add attack type for main.py
if [ "$script" = "../main.py" ]; then
    common_args+=(--config.attack="$attack")
fi

# Add evaluation-specific args
if [ "$experiment_type" = "evaluate" ]; then
    common_args+=(
        --config.eval_model="$model"
        --config.logfile="$logfile"
    )
fi

# Add leak-specific result prefix
if [ "$experiment_type" = "leak" ]; then
    common_args[2]="--config.result_prefix=../results/transfer_${model}_autodan_${n}_progressive_ctr${w_tar_1}"
fi

# Run individual experiments with data offset loop
if [ "$experiment_type" = "individual" ]; then
    for data_offset in 0; do
        python -u "$script" "${common_args[@]}" --config.data_offset=$data_offset
    done
else
    # Run other experiment types once
    python -u "$script" "${common_args[@]}"
fi
