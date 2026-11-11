#!/bin/bash
# Valutazione e statistica includendo gli autoencoder addestrati sui casi ben segmentati e sul controllo.
# Uso: ./autoencoder_7_evaluate_wellsegmented.sh
export AE_DATASET=904 NUMPY_MADVISE_HUGEPAGE=0
source /home/grolo/venv_explainability/bin/activate
cd /mnt/sda2/grolo   # log relativi in /mnt/sda2/grolo/logs
echo "=== valutazione: $(date) ==="
nice python3 -u /home/grolo/thesis_pipeline/5_autoencoder/autoencoder_eval.py 6 > logs/autoencoder_eval_wellsegmented.log 2>&1 || { echo "ERRORE valutazione"; exit 1; }
echo "=== statistica: $(date) ==="
python3 -u /home/grolo/thesis_pipeline/5_autoencoder/autoencoder_stats.py > logs/autoencoder_stats_wellsegmented.log 2>&1 || { echo "ERRORE statistica"; exit 1; }
echo "=== VALUTAZIONE CASI BUONI FINITA: $(date) ==="
