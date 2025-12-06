echo "Running Universal GCG Attack"
echo "================================"

# Create results folder if it doesn't exist
if [ ! -d "../results" ]; then
    mkdir -p "../results"
    echo "Folder '../results' created."
else
    echo "Folder '../results' already exists."
fi

# Run attack with different data offsets
for data_offset in 0 1 2; do
    echo "📊 Running experiment with data offset $data_offset"
    echo "----------------------------------------------------"
    
    python3 ../main.py \
        --config=../configs/universal_config.py \
        --config.data_offset=$data_offset
    
    if [ $? -eq 0 ]; then
        echo "Completed data offset $data_offset successfully!"
    else
        echo "[ERROR] Failed at data offset $data_offset"
        echo "Stopping experiments due to failure."
        break
    fi
    
    echo ""
done

echo "All experiments completed!"