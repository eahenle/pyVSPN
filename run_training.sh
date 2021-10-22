#!/usr/bin/bash

# runs MPNN training using specified flag args

python3 main.py                             \
    "working_capacity_vacuum_swing [mmol/g]"\
    --device            cpu                 \
    --learning_rate     0.001               \
    --max_epochs        2000                \
    --mpnn_steps        5                   \
    --hidden_encoding   10                  \
    --test_prop         0.1                 \
    --val_prop          0.01                \
    --batch_size        1024                \
    --lr_decay_gamma    1                   \
    #--verbose                               \
    #--recache
