#!/usr/bin/bash

# runs MPNN training using specified flag args

python3 main.py                             \
    "working_capacity_vacuum_swing [mmol/g]"\
    --batch_size        512                 \
    --device            cpu                 \
    --learning_rate     0.005               \
    --l1_reg            0.01                \
    --l2_reg            0.01                \
    --max_epochs        2000                \
    --mpnn_steps        3                   \
    --nb_reports        2000                \
    --hidden_encoding   20                  \
    --stop_threshold    0.01                \
    --test_prop         0.1                 \
    --val_prop          0.001               \
    --mpnn_update       mean                \
    #--recache
