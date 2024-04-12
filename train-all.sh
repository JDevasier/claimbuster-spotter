#!/bin/bash

# Define the thresholds
thresholds=(0.01 0.05 0.1 0.3 0.5 0.7 0.9 0.95 0.99)

# Loop through each threshold
for threshold in "${thresholds[@]}"
do
    # Create the directory name based on the threshold
    model_dir="./output/bb_${threshold}/"
    
    echo "Running command for threshold: $threshold and imbalance ratio: $imbalance_ratio"
    python3 -m adv_transformer.train --cs_model_dir="$model_dir" --cs_adv_train=False --cs_gpu=0 --cs_train_steps=10 --cs_batch_size_adv=12 --cs_lambda=0.1 --cs_imbalance=True --cs_imbalance_ratio=$imbalance_ratio
done