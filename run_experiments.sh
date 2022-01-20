#!/usr/bin/bash

# constant parameters
model="ParallelVSPN"
max_epochs=20
test_prop=0.1
val_prop=0.01
batch_size=256
random_seed=42
learning_rate=0.01
lr_decay_gamma=1

# loop over tunable parameters
for mpnn_steps in 1 2 3 4 5 6
do
    for voro_embedding in 2 4 8 16
    do
        echo "Running experiment w/ $mpnn_steps MPNN steps, voro embedding length $voro_embedding."

        # create experiment label string
        str="$learning_rate-$mpnn_steps-$lr_decay_gamma-$voro_embedding"

        # run experiment
        python3 main.py                             \
            "working_capacity_vacuum_swing [mmol/g]"\
            --device            cuda                \
            --learning_rate     $learning_rate      \
            --max_epochs        $max_epochs         \
            --mpnn_steps        $mpnn_steps         \
            --test_prop         $test_prop          \
            --val_prop          $val_prop           \
            --batch_size        $batch_size         \
            --lr_decay_gamma    $lr_decay_gamma     \
            --model             $model              \
            --voro_embedding    $voro_embedding     \
            --random_seed       $random_seed        \
            --output_path       "./tuning/$str"     \
            --recache
    done
done

