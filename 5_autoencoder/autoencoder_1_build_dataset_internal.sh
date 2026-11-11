#!/bin/bash
# Dataset dell'autoencoder, casi del test interno TotalSegmentator ed entropia float32.
# Uso: ./autoencoder_1_build_dataset_internal.sh
export AE_DATASET=904 NUMPY_MADVISE_HUGEPAGE=0
P=/home/aiart/anaconda3/envs/nnunet_env/bin/python
cd /mnt/sda2/grolo   # log relativi in /mnt/sda2/grolo/logs
nice -n 19 ionice -c3 $P /home/grolo/thesis_pipeline/5_autoencoder/autoencoder_build_dataset.py --skip_humanitas --workers 2 && \
nice -n 19 ionice -c3 $P /home/grolo/thesis_pipeline/5_autoencoder/autoencoder_entropy_float32.py
echo "=== DATASET 904 (TS) FINITO (exit $?): $(date) ==="
