#!/bin/bash

DATA_DIR="${PLATFORM:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
MODEL_IN="${MODEL_IN:-}"
LOG_DIR="${LOG_DIR:-}"
METRICS="${METRICS:-}"
MODE="${MODE:-}"
if [ "$MODE" = "train" ]; then
    python3 -m src.train --devices 0 --width 48 --arch EquiUnet
else
    python3 -m src.inference --devices 0 --width 48 --arch EquiUnet
fi