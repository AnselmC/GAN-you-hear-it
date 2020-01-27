#!/bin/bash

INPUT='training_data_new/'
EPOCHS=50
BATCH_SIZE=32
SUBSET_SIZE=128
LINEAR='linear'
CONV='conv'
GENERATOR=$CONV
ENTROPY_SIZE_CONV=10
ENTROPY_SIZE_LINEAR=128
REG_STRENGTH=1000
DIS_LR=0.0002
GEN_LR=0.0001
ENTROPY_SIZE=$(($GENERATOR = $LINEAR ? $ENTROPY_SIZE_LINEAR : $ENTROPY_SIZE_CONV))

python3 train.py --input $INPUT \
                 --epochs $EPOCHS \
                 --batch_size $BATCH_SIZE \
                 --subset_size $SUBSET_SIZE \
                 --entropy_size $ENTROPY_SIZE \
                 --generator $GENERATOR \
                 --lr $DIS_LR $GEN_LR \
                 --reg_strength $REG_STRENGTH \
                 --visual
