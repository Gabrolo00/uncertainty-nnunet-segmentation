#!/bin/bash
# Inferenza dell'autoencoder sui casi non visti: ricostruzione e residui sui canali di logit e CT.
# Uso: GPU=0 ./autoencoder_4_infer_residuals.sh
export AE_DATASET=904 NUMPY_MADVISE_HUGEPAGE=0 CUDA_VISIBLE_DEVICES=${GPU:-0}
P=/home/aiart/anaconda3/envs/nnunet_env/bin/python
cd /mnt/sda2/grolo   # log relativi in /mnt/sda2/grolo/logs
for v in "b entropy,logit" "c entropy,logit,ct"; do
  set -- $v
  for f in 0 1 2 3 4; do
    nice -n 19 $P /home/grolo/thesis_pipeline/5_autoencoder/autoencoder_train_infer.py infer_resid --variant $1 --channels $2 --test_fold $f --n_down 4 --latent 16 --iters 30000 || echo "ERRORE $1 fold $f"
  done
done
echo "=== RESIDUI FINITI: $(date) ==="
