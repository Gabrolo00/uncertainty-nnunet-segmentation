#!/bin/bash
# Training dell'autoencoder (variante a) sui soli casi ben segmentati e su un sottoinsieme casuale di pari numero
# (controllo).
# Uso: ./autoencoder_6_train_wellsegmented.sh
export AE_DATASET=904 NUMPY_MADVISE_HUGEPAGE=0
P=/home/aiart/anaconda3/envs/nnunet_env/bin/python
L=/mnt/sda2/grolo/kidney/autoencoder/logs
LIST=/mnt/sda2/grolo/kidney/autoencoder/dataset/good_cases_far2_lt0.5ml.csv
cd /mnt/sda2/grolo   # log relativi in /mnt/sda2/grolo/logs
run() { # $1 gpu, $2 modo
  for t in 0 1 2 3 4; do
    A="--variant a --channels entropy --test_fold $t --n_down 4 --latent 16 --iters 30000 --train_list $LIST --train_mode $2"
    for cmd in train infer quickeval; do CUDA_VISIBLE_DEVICES=$1 $P /home/grolo/thesis_pipeline/5_autoencoder/autoencoder_train_infer.py $cmd $A || { echo "ERRORE $2 fold $t $cmd"; break; }; done
    echo "=== $2 fold $t finito: $(date) ==="
  done
}
run 0 list > $L/goodcases_list_gpu0.log 2>&1 &
run 1 random_like_list > $L/goodcases_rand_gpu1.log 2>&1 &
wait
echo "=== AE CASI BUONI 904 FINITO: $(date) ==="
