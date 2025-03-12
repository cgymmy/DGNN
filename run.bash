#!/usr/bin/env bash

# 定义 partition 数组
# N_partitions=(0.06 0.05 0.04 0.03 0.02 0.01)
N_partitions=(0.08 0.05 0.03 0.02 0.01)
hidden_size=(10 20 30)

# 遍历数组中的每个 partition 值
for partition in "${N_partitions[@]}"; do
    python run.py --partition "$partition" 
done
# for hidden_size in "${hidden_size[@]}"; do
#     python run.py --hidden_size "$hidden_size"
# done