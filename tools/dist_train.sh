#!/usr/bin/env bash
PYTHON=${PYTHON:-"python"}

CFG=$1
GPUS=$2
WORK_DIR=$3
PY_ARGS=${@:5}
PORT=${PORT:-29500}

# echo ${CFG%.*}
# WORK_DIR=${WORK_DIR}$(echo ${CFG%.*} | sed -e "s/configs//g")/

echo $WORK_DIR
$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    tools/train.py $CFG --work_dir $WORK_DIR  --seed 0 --launcher pytorch ${PY_ARGS}
