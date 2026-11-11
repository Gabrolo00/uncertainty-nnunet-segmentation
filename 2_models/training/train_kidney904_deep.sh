#!/bin/bash
# Training dei 5 fold del deep ensemble del rene, distribuiti sulle due GPU.
# Uso: ./train_kidney904_deep.sh
export nnUNet_raw=/mnt/sda2/grolo/nnUNet/raw
export nnUNet_preprocessed=/mnt/sda2/grolo/nnUNet/preprocessed
export nnUNet_results=/mnt/sda2/grolo/nnUNet/results
export PYTHONPATH=/mnt/sda2/grolo/py_shims
cd /mnt/sda2/grolo   # log relativi in /mnt/sda2/grolo/logs
PY=/home/aiart/anaconda3/envs/nnunet_env/bin/python
train_fold() {  # fold gpu [extra]
  CUDA_VISIBLE_DEVICES=$2 $PY /home/grolo/thesis_pipeline/2_models/training/train_deep_ensemble.py --dataset 904 --configuration 3d_fullres --fold $1 --num_epochs 1000 $3 \
    >> logs/train_904_fold$1.log 2>&1
  echo "=== fold $1 finito (exit $?): $(date)" >> logs/train_904_fold$1.log
}
if [ "$1" = resume ]; then train_fold $2 $3 --continue_training; exit; fi
until grep -q "preprocessing finito (exit 0)" logs/preprocess_904.log; do
  grep -q "preprocessing finito" logs/preprocess_904.log && { echo "preprocessing fallito"; exit 1; }; sleep 60; done
(for f in 0 2 4; do train_fold $f 0; done) &
sleep 60
(for f in 1 3; do train_fold $f 1; done) &
wait
echo "=== TUTTI I FOLD FINITI: $(date)"
