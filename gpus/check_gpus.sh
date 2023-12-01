#!/bin/bash

LOG_DIR=${LOG_DIR:-"/"}

# Handle MLCube parameters
while [ $# -gt 0 ]; do
    case "$1" in
    --log_dir=*)
        LOG_DIR="${1#*=}"
        ;;
    *) ;;
    esac
    shift
done

echo "CUDA_VISIBLE_DEVICES $CUDA_VISIBLE_DEVICES" |& tee "$LOG_DIR/gpus.log"
echo "NVIDIA_VISIBLE_DEVICES $NVIDIA_VISIBLE_DEVICES" |& tee "$LOG_DIR/gpus.log"
nvidia-smi |& tee -a "$LOG_DIR/gpus.log"
nvidia-smi --query-gpu=gpu_name,uuid --format=csv |& tee -a "$LOG_DIR/gpus.log"
