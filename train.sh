#!/bin/bash
INPUT='data/training_data_single_beat/'
EPOCHS=100
BATCH_SIZE=64
SUBSET_SIZE=2000
LINEAR='linear'
CONV='conv'
GENERATOR=$CONV
ENTROPY_SIZE_CONV=128
ENTROPY_SIZE_LINEAR=128
REG_STRENGTH=10
DIS_LR=0.0001
GEN_LR=0.00015
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
