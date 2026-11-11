#!/bin/bash
# Valutazione delle mappe residue e statistica.
# Uso: ./autoencoder_5_evaluate.sh
until grep -q "RESIDUI FINITI" /mnt/sda2/grolo/kidney/autoencoder/logs/resid.log; do sleep 60; done
export AE_DATASET=904 NUMPY_MADVISE_HUGEPAGE=0
P=/home/aiart/anaconda3/envs/nnunet_env/bin/python
nice -n 19 ionice -c3 $P /home/grolo/thesis_pipeline/5_autoencoder/autoencoder_eval.py 4 > /mnt/sda2/grolo/kidney/autoencoder/logs/eval_all.log 2>&1 && nice -n 19 $P /home/grolo/thesis_pipeline/5_autoencoder/autoencoder_stats.py > /mnt/sda2/grolo/kidney/autoencoder/logs/stats.log 2>&1
echo "=== VALUTAZIONE 904 FINITA (exit $?): $(date) ==="
