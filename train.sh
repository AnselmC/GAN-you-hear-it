#!/bin/bash

INPUT='training_data_new/'
EPOCHS=50
BATCH_SIZE=8
SUBSET_SIZE=128
LINEAR='linear'
CONV='conv'
GENERATOR=$CONV
ENTROPY_SIZE_CONV=10
ENTROPY_SIZE_LINEAR=128
ENTROPY_SIZE=$(("$GENERATOR" = "$LINEAR" ? $ENTROPY_SIZE_LINEAR : $ENTROPY_SIZE_CONV))

python3 train.py --input $INPUT \
                 --epochs $EPOCHS \
                 --batch_size $BATCH_SIZE \
                 --subset_size $SUBSET_SIZE \
                 --entropy_size $ENTROPY_SIZE \
                 --generator $GENERATOR \
                 --visual
