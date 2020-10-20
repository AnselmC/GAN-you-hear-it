#!/bin/bash

INPUT='training_data_single_beat/'
EPOCHS=1000
BATCH_SIZE=128
SUBSET_SIZE=5000
LINEAR='linear'
CONV='conv'
GENERATOR=$CONV
ENTROPY_SIZE_CONV=12
ENTROPY_SIZE_LINEAR=128
GEN_LR=0.000015
DIS_LR=0.00001
MODEL_GEN='results/models/500_128_12_4512_gen.model'
MODEL_DIS='results/models/500_128_12_4512_dis.model'
REG_STRENGTH=100
ENTROPY_SIZE=$(("$GENERATOR" = "$LINEAR" ? $ENTROPY_SIZE_LINEAR : $ENTROPY_SIZE_CONV))

python3 train.py --input $INPUT \
                 --epochs $EPOCHS \
                 --batch_size $BATCH_SIZE \
                 --subset_size $SUBSET_SIZE \
                 --entropy_size $ENTROPY_SIZE \
                 --generator $GENERATOR \
		 --reg_strength $REG_STRENGTH \
		 --lr $DIS_LR $GEN_LR \
		 --model $MODEL_DIS $MODEL_GEN
