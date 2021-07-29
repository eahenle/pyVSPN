#!/usr/bin/bash
python3 main.py \
    --properties "properties.csv" \
    --node_encoding 20 \
    --graph_encoding 15 \
    --mpnn_steps 4 \
    --max_epochs 5000 \
    --stop_threshold 0.1 \
    --learning_rate 0.002 \
    --nb_reports 20 \
    --l1_reg 0.1 \
    "working_capacity_vacuum_swing [mmol/g]"
