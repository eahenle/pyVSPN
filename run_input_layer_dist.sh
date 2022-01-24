#!/usr/bin/bash

# runs MPNN training using specified flag args

python3 input_layer_dist.py "working_capacity_vacuum_swing [mmol/g]" --model ParallelVSPN
