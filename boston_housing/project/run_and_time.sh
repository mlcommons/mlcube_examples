#!/bin/bash

set -e

: ${DATASET_FILE_PATH:=${1:-"./processed_dataset.csv"}}
: ${N_ESTIMATORS:=${2:-"100"}}

ARGS="--dataset_file_path=$DATASET_FILE_PATH"
ARGS+=" --n_estimators $N_ESTIMATORS"

# Execute command and time it
time python 03_train.py ${ARGS}