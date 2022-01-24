#!/usr/bin/bash

./run_experiments.sh 0 10 > "tuning/exp_logs0.o" &
./run_experiments.sh 1 20 > "tuning/exp_logs1.o" &
./run_experiments.sh 2 30 > "tuning/exp_logs2.o" &
./run_experiments.sh 3 40 > "tuning/exp_logs3.o" &
