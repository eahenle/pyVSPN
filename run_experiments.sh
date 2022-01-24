#!/usr/bin/bash

# constant parameters
model="ParallelVSPN"
max_epochs=2000
test_prop=0.1
val_prop=0.01
batch_size=128
random_seed=42
learning_rate=0.01
mpnn_steps=3
hidden_encoding=50

# loop over tunable parameters
for CUDA in 0 1 2 3 # use CUDA device as source for voro_embedding
do
    for voro_h in 10 20 30
    do
        echo "Running experiment w/ $mpnn_steps MPNN steps, voro embedding length $voro_embedding, hidden voro length $voro_h, hidden atom length $hidden_encoding."

        voro_embedding=$(( ($CUDA + 1) ** 2 ))

        # create experiment label string
        exp_code="$voro_embedding-$voro_h"

        # run experiment
        python3 main.py                             \
            "working_capacity_vacuum_swing [mmol/g]"\
            --device            "cuda:$CUDA"        \
            --learning_rate     $learning_rate      \
            --max_epochs        $max_epochs         \
            --mpnn_steps        $mpnn_steps         \
            --test_prop         $test_prop          \
            --val_prop          $val_prop           \
            --batch_size        $batch_size         \
            --model             $model              \
            --voro_embedding    $voro_embedding     \
            --random_seed       $random_seed        \
            --output_path       "./tuning/$exp_code"\
            --voro_h            $voro_h             \
            --hidden_encoding   $hidden_encoding    \
            --recache
    done
done

