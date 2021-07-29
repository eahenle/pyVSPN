#!/usr/bin/bash
python3 main.py \
    --properties "properties.csv" \
    --node_encoding 20 \
    --graph_encoding 15 \
    --mpnn_steps 2 \
    --max_epochs 10000 \
    --stop_threshold 0.1 \
    --learning_rate 0.001 \
    --nb_reports 20 \
    "working_capacity_vacuum_swing [mmol/g]"
