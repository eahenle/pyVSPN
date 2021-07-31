#!/usr/bin/bash
python3 main.py \
    --node_encoding 20 \
    --mpnn_steps 5 \
    --max_epochs 6000 \
    --stop_threshold 0.12 \
    --learning_rate 0.002 \
    --nb_reports 50 \
    --l1_reg 0.01 \
    --l2_reg 0.001 \
    "working_capacity_vacuum_swing [mmol/g]"
