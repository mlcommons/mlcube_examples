#!/bin/bash

set -e

: ${data_path:=${1:-}}
: ${parameters_file:=${2:-}}
: ${output_path:=${2:-}}

ARGS="--data_path=$data_path"
ARGS+=" --parameters_file $parameters_file"
ARGS+=" --output_path $output_path"

# Execute command and time it
echo Processing data. This may take a while...
time python3 preprocess.py ${ARGS}