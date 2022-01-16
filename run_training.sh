#!/usr/bin/bash

# runs MPNN training using specified flag args

python3 main.py                             \
    "working_capacity_vacuum_swing [mmol/g]"\
    --device            cuda                \
    --learning_rate     0.01                \
    --max_epochs        20                  \
    --mpnn_steps        2                   \
    --test_prop         0.1                 \
    --val_prop          0.01                \
    --batch_size        256                 \
    --lr_decay_gamma    1                   \
    --model             "ParallelVSPN"      \
    --voro_embedding    3                   \
    --random_seed       42                  \
    --recache
