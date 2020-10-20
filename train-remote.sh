#!/bin/bash

INPUT='training_data_single_beat/'
EPOCHS=2001
BATCH_SIZE=128
SUBSET_SIZE=30000
LINEAR='linear'
CONV='conv'
GENERATOR=$CONV
ENTROPY_SIZE_CONV=128
ENTROPY_SIZE_LINEAR=128
GEN_LR=0.00001
DIS_LR=0.00002
REG_STRENGTH=1
ENTROPY_SIZE=$(("$GENERATOR" = "$LINEAR" ? $ENTROPY_SIZE_LINEAR : $ENTROPY_SIZE_CONV))

python3 train.py --input $INPUT \
                 --epochs $EPOCHS \
                 --batch_size $BATCH_SIZE \
                 --subset_size $SUBSET_SIZE \
                 --entropy_size $ENTROPY_SIZE \
                 --generator $GENERATOR \
		 --reg_strength $REG_STRENGTH \
		 --lr $DIS_LR $GEN_LR
