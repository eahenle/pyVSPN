#!/usr/bin/bash

# runs MPNN training using specified flag args

python3 main.py                             \
    "working_capacity_vacuum_swing [mmol/g]"\
    --batch_size        2                   \
    --device            cpu                 \
    --learning_rate     0.002               \
    --l1_reg            0.01                \
    --l2_reg            0.001               \
    --max_epochs        5000                \
    --mpnn_steps        3                   \
    --nb_reports        25                  \
    --node_encoding     20                  \
    --stop_threshold    0.125               \
    --s2s_steps         2                   \
    --test_prop         0.2                 \
    --recache
