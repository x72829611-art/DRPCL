#!/bin/bash
cd /home/student1/projects/DRPCL
source ~/miniconda3/etc/profile.d/conda.sh
# create and activate the conda environment


BETA=0.5
LR1=1e-6
LR2=8e-4
LR_Q=1e-6
BATCH_SIZE=128
REPLICATIONS=10

echo "========================================"
echo "DRPCL Experiment: TWINS"
echo "Time: $(date)"
echo "Config: Beta=$BETA, LR1=$LR1, LR2=$LR2, LR_Q=$LR_Q, Batch=$BATCH_SIZE"
echo "========================================"

python -u experiment/main.py \
    --datasets TWINS \
    --knob DRPCL_twins \
    --beta $BETA \
    --lr1 $LR1 \
    --lr2 $LR2 \
    --lr_q $LR_Q \
    --batch_size $BATCH_SIZE \
    --replications $REPLICATIONS
