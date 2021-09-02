#!/usr/bin/bash

# runs MPNN training using specified flag args

python3 main.py                             \
    "working_capacity_vacuum_swing [mmol/g]"\
    --device            cpu                 \
    --learning_rate     0.1                 \
    --max_epochs        1000                \
    --mpnn_steps        1                   \
    --hidden_encoding   10                  \
    --test_prop         0.1                 \
    --val_prop          0.01                \
    --recache
