#!/bin/bash

# Parse arguments with defaults - REMOVE model override
ATTACK_TYPE=${1:-"individual"}
DATA_TYPE=${2:-"behaviors"}
N_STEPS=${3:-100}
N_TRAIN_DATA=${4:-1}
BATCH_SIZE=${5:-32}
LEARNING_RATE=${6:-0.01}
CONTROL_INIT=${7:-"! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"}

echo "🚀 Running Universal GCG Attack"
echo "================================"
echo "Using model from config file (Qwen/Qwen2.5-0.5B-Instruct)"
echo "Attack Type: $ATTACK_TYPE"
echo "Data Type: $DATA_TYPE"
echo "Steps: $N_STEPS"
echo "Training Data: $N_TRAIN_DATA"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "Control Init: $CONTROL_INIT"
echo "================================"

# Create results folder if it doesn't exist
if [ ! -d "../results" ]; then
    mkdir -p "../results"
    echo "Folder '../results' created."
else
    echo "Folder '../results' already exists."
fi

# Run attack with different offsets
for data_offset in 0 10; do
    echo "📊 Running experiment with data offset $data_offset"
    
    # FIXED: Don't override model settings - use config file defaults
    python3 ../main.py \
        --config=../configs/universal_config.py \
        --config.attack_type="$ATTACK_TYPE" \
        --config.data_type="$DATA_TYPE" \
        --config.n_steps=$N_STEPS \
        --config.n_train_data=$N_TRAIN_DATA \
        --config.batch_size=$BATCH_SIZE \
        --config.lr=$LEARNING_RATE \
        --config.control_init="$CONTROL_INIT" \
        --config.data_offset=$data_offset \
        --config.verbose=true
        
    if [ $? -eq 0 ]; then
        echo "Completed offset $data_offset"
    else
        echo "Failed offset $data_offset"
        break
    fi
done

echo "All experiments completed!"