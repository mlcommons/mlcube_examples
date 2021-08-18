#!/bin/bash

MODEL_FILE="model.pth"
CONFIG_FILE="config.json"
MODEL_DIR="${MODEL_DIR:-./checkpoints}"
MODEL_PATH="${MODEL_DIR}/${MODEL_FILE}"
CONFIG_PATH="${MODEL_DIR}/${CONFIG_FILE}"

if [ ! -d "$MODEL_DIR" ]
then
    mkdir $MODEL_DIR
    chmod go+rx $MODEL_DIR
#     python utils/download_librispeech.py utils/librispeech.csv $DATA_DIR -e ${DATA_ROOT_DIR}/
fi
curl  https://raw.githubusercontent.com/jfhealthcare/Chexpert/master/config/pre_train.pth --output ${MODEL_PATH}
curl  https://raw.githubusercontent.com/jfhealthcare/Chexpert/master/config/example.json --output ${CONFIG_PATH}

