#!/bin/bash

DATA_DIR="${PLATFORM:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
MODE="${MODE:-}"
if [ "$MODE" = "train" ]; then
then
    python3 -m src.train --devices 0 --width 48 --arch EquiUnet
else
    python3 -m src.inference --devices 0 --width 48 --arch EquiUnet
fi