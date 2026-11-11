#!/bin/bash
# Analisi per zona con budget di allarmi esatto (1% della regione di interesse) per tutti gli autoencoder.
# Uso: ./autoencoder_9_zones_exact_budget.sh
export AE_DATASET=904 NUMPY_MADVISE_HUGEPAGE=0
source /home/grolo/venv_explainability/bin/activate
cd /mnt/sda2/grolo   # log relativi in /mnt/sda2/grolo/logs
nice python3 -u /home/grolo/thesis_pipeline/5_autoencoder/autoencoder_zones.py 8 > logs/regions_exact_abc.log 2>&1 &
AE_COMBINE_SET=good nice python3 -u /home/grolo/thesis_pipeline/5_autoencoder/autoencoder_zones.py 8 > logs/regions_exact_good.log 2>&1 &
wait
echo "=== ZONE BUDGET ESATTO FINITE: $(date) ==="
