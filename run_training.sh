#!/usr/bin/bash

# runs MPNN training using specified flag args

python3 main.py                             \
    "working_capacity_vacuum_swing [mmol/g]"\
    --device            cpu                 \
    --learning_rate     0.01                \
    --max_epochs        200                 \
    --mpnn_steps        2                   \
    --test_prop         0.1                 \
    --val_prop          0.01                \
    --batch_size        128                 \
    --lr_decay_gamma    1                   \
    --model             "ParallelVSPN"      \
    --recache
