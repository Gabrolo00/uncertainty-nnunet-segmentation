#!/bin/bash
# Combinazioni e analisi per zona per gli autoencoder addestrati sui casi ben segmentati e sul controllo.
# Uso: ./autoencoder_8_combinations_and_zones_wellsegmented.sh
export AE_DATASET=904 AE_COMBINE_SET=good NUMPY_MADVISE_HUGEPAGE=0
source /home/grolo/venv_explainability/bin/activate
cd /mnt/sda2/grolo   # log relativi in /mnt/sda2/grolo/logs
for s in "/home/grolo/thesis_pipeline/5_autoencoder/autoencoder_combine.py select 8" "/home/grolo/thesis_pipeline/5_autoencoder/autoencoder_combine.py eval 8" "/home/grolo/thesis_pipeline/5_autoencoder/autoencoder_zones.py 8"; do
  echo "=== $s: $(date) ==="
  nice python3 -u $s > "logs/goodcases_$(echo $s | tr -d ' .').log" 2>&1 || { echo "ERRORE $s"; exit 1; }
done
echo "=== COMBINAZIONI E ZONE CASI BUONI FINITE: $(date) ==="
