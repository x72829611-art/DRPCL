#!/bin/bash
cd /home/student1/projects/DRPCL
source ~/miniconda3/etc/profile.d/conda.sh
# create and activate the conda environment

BETA=0.1
LR1=5e-6
LR2=5e-5
LR_Q=1e-6
BATCH_SIZE=128
REPLICATIONS=10

echo "========================================"
echo "DRPCL Experiment: JOBS"
echo "Time: $(date)"
echo "Config: Beta=$BETA, LR1=$LR1, LR2=$LR2, LR_Q=$LR_Q, Batch=$BATCH_SIZE"
echo "========================================"

python -u experiment/main.py \
    --datasets JOBS \
    --knob DRPCL_jobs \
    --beta $BETA \
    --lr1 $LR1 \
    --lr2 $LR2 \
    --lr_q $LR_Q \
    --batch_size $BATCH_SIZE \
    --replications $REPLICATIONS
