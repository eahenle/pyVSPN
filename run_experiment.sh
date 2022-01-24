#!/usr/bin/bash

# constant parameters
model="ParallelVSPN"
random_seed=42

CUDA=$1
voro_embedding=$2
voro_h=$3

# create experiment label string
exp_code="$voro_embedding-$voro_h"

# run experiment
python3 main.py                             \
    "working_capacity_vacuum_swing [mmol/g]"\
    --model             $model              \
    --random_seed       $random_seed        \
    --output_path       "./tuning/$exp_code"\
    --device            "cuda:$CUDA"        \
    --voro_embedding    $voro_embedding     \
    --voro_h            $voro_h             \
    --cache_path        "./cache/$CUDA"     \
    --recache
