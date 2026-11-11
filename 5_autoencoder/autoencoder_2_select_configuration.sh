#!/bin/bash
# Scelta della configurazione dell'autoencoder (profondita' e dimensione del collo di bottiglia) su un gruppo di selezione.
# Uso: GPU=0 ./autoencoder_2_select_configuration.sh
export AE_DATASET=904 NUMPY_MADVISE_HUGEPAGE=0
export CUDA_VISIBLE_DEVICES=${GPU:-0}
P=/home/aiart/anaconda3/envs/nnunet_env/bin/python
D=/mnt/sda2/grolo/kidney/autoencoder/dataset
cd /mnt/sda2/grolo   # log relativi in /mnt/sda2/grolo/logs
grep -q humanitas $D/folds.csv || { echo "folds.csv senza Humanitas: dataset non completo"; exit 1; }
for cfg in "4 16" "4 64" "3 4" "3 16" "3 64"; do
  set -- $cfg
  A="--variant a --channels entropy --test_fold 0 --inner_fold 1 --n_down $1 --latent $2 --iters 30000"
  for cmd in train infer quickeval; do $P /home/grolo/thesis_pipeline/5_autoencoder/autoencoder_train_infer.py $cmd $A || { echo "ERRORE d$1 lat$2 $cmd"; break; }; done
  echo "=== config d$1 lat$2 finita: $(date) ==="
done
echo "=== SELEZIONE 904 FINITA: $(date) ==="
