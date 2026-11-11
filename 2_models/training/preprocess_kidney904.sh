#!/bin/bash
# Fingerprint, pianificazione e preprocessing nnU-Net del dataset del rene.
# Uso: ./preprocess_kidney904.sh
export nnUNet_raw=/mnt/sda2/grolo/nnUNet/raw
export nnUNet_preprocessed=/mnt/sda2/grolo/nnUNet/preprocessed
export nnUNet_results=/mnt/sda2/grolo/nnUNet/results
export PYTHONPATH=/mnt/sda2/grolo/py_shims
/home/aiart/anaconda3/envs/nnunet_env/bin/nnUNetv2_plan_and_preprocess -d 904 -c 3d_fullres --verify_dataset_integrity -np 8
echo "=== preprocessing finito (exit $?): $(date)"
