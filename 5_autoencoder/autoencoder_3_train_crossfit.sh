#!/bin/bash
# Training dell'autoencoder per le tre varianti di ingresso, uno per gruppo di cross-fitting.
# Uso: ./autoencoder_3_train_crossfit.sh
export AE_DATASET=904 NUMPY_MADVISE_HUGEPAGE=0
P=/home/aiart/anaconda3/envs/nnunet_env/bin/python
L=/mnt/sda2/grolo/kidney/autoencoder/logs
cd /mnt/sda2/grolo   # log relativi in /mnt/sda2/grolo/logs
run() { # $1 gpu, poi triple "variante canali fold"
  local gpu=$1; shift
  while [ $# -gt 0 ]; do
    A="--variant $1 --channels $2 --test_fold $3 --n_down 4 --latent 16 --iters 30000"
    for cmd in train infer quickeval; do CUDA_VISIBLE_DEVICES=$gpu $P /home/grolo/thesis_pipeline/5_autoencoder/autoencoder_train_infer.py $cmd $A || { echo "ERRORE $1 fold $3 $cmd"; break; }; done
    echo "=== $1 fold $3 finito: $(date) ==="
    shift 3
  done
}
run 0 a entropy 0  a entropy 1  a entropy 2  a entropy 3  a entropy 4  b entropy,logit 0  b entropy,logit 1  b entropy,logit 2 > $L/crossfit_gpu0.log 2>&1 &
run 1 b entropy,logit 3  b entropy,logit 4  c entropy,logit,ct 0  c entropy,logit,ct 1  c entropy,logit,ct 2  c entropy,logit,ct 3  c entropy,logit,ct 4 > $L/crossfit_gpu1.log 2>&1 &
wait
echo "=== CROSS-FITTING 904 FINITO: $(date) ==="
