#!/bin/bash

export WANDB_MODE=disabled

# Parse arguments with defaults
MODEL_NAME=${1:-"gpt2"}
ATTACK_TYPE=${2:-"individual"}
DATA_TYPE=${3:-"behaviors"}
DEVICE=${4:-"auto"}
N_STEPS=${5:-1000}
N_TRAIN_DATA=${6:-10}
BATCH_SIZE=${7:-512}
LEARNING_RATE=${8:-0.01}
CONTROL_INIT=${9:-"! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"}

echo "🚀 Running Universal GCG Attack"
echo "================================"
echo "Model: $MODEL_NAME"
echo "Attack Type: $ATTACK_TYPE"
echo "Data Type: $DATA_TYPE"
echo "Device: $DEVICE"
echo "Steps: $N_STEPS"
echo "Training Data: $N_TRAIN_DATA"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "Control Init: $CONTROL_INIT"
echo "================================"



# Create results folder if it doesn't exist
if [ ! -d "../results" ]; then
    mkdir "../results"
    echo "Folder '../results' created."
else
    echo "Folder '../results' already exists."
fi


# Run attack with different offsets
for data_offset in 0 10 20 30 40 50 60 70 80 90; do
    echo "📊 Running experiment with data offset $data_offset"
    
    python3 ../main.py \
        --config ../configs/universal_config.py \
        --config.model_name="$MODEL_NAME" \
        --config.device="$DEVICE" \
        --config.attack_type="$ATTACK_TYPE" \
        --config.data_type="$DATA_TYPE" \
        --config.n_steps=$N_STEPS \
        --config.n_train_data=$N_TRAIN_DATA \
        --config.batch_size=$BATCH_SIZE \
        --config.lr=$LEARNING_RATE \
        --config.control_init="$CONTROL_INIT" \
        --config.data_offset=$data_offset \
        --config.verbose=true
        
    echo "✅ Completed offset $data_offset"
done

echo "🎉 All experiments completed!"