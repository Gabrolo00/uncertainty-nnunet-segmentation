#!/bin/bash
# Scomposizione aleatoria/epistemica per il deep ensemble del rene: soglie, valutazione e classifica di entropia
# attesa ed entropia totale insieme alle 14 misure.
# Uso: ./decomposition_kidney904.sh
set -e
source /home/grolo/venv_explainability/bin/activate
export NUMPY_MADVISE_HUGEPAGE=0
cd /home/grolo
RAW=/mnt/sda2/grolo/nnUNet/raw/Dataset904_KidneyTSMechanical
TEST=/mnt/sda2/grolo/kidney/predictions/deep
O=/mnt/sda2/grolo/kidney/deep
L=$O/logs
MAPS=$O/kidney_uncertainty_maps
ALL=$O/kidney_threshold_calibration_decomposition.csv
EV=$O/kidney_eval_threshold_decomposition
R=$O/ranking_stats_with_decomposition
for x in $ALL $EV $R $O/kidney_threshold_calibration_expected_entropy.csv $O/kidney_threshold_calibration_total_entropy.csv; do
  [ ! -e $x ] || { echo "$x esiste gia': non sovrascrivo"; exit 1; }
done
for u in expected_entropy total_entropy; do
  n=$(ls $MAPS | grep -cE "^s[0-9]+_$u\.nii\.gz$"); [ "$n" -eq 97 ] || { echo "$n mappe $u invece di 97"; exit 1; }
done

echo "=== [E] expected_entropy e total_entropy $(date) ==="
export RAW TEST MAPS O L
run_e() {
  local U=$1 CSV=$O/kidney_threshold_calibration_$1.csv rc=0
  for a in 1 2 3; do
    rm -f $CSV
    python -u /home/grolo/thesis_pipeline/3_uncertainty/thresholds/Calibrate_threshold_loo_v2.py --prediction_dir $TEST --gt_dir $RAW/labelsTs \
      --ct_dir $RAW/imagesTs --uncertainty_dir $MAPS --uncertainty_type $U \
      --output_csv $CSV --qa_dir $O/kidney_qa_$U \
      --roi_mode buffer_15mm --cache_case_scores >> $L/blockE_threshold_$U.log 2>&1
    rc=$?; echo "=== $U finito (exit $rc, tentativo $a): $(date) ===" >> $L/blockE_threshold_$U.log
    [ $rc -ne 137 ] && return $rc
    sleep 300
  done
  return $rc
}
export -f run_e
printf 'expected_entropy\ntotal_entropy\n' | xargs -P 2 -I{} bash -c 'run_e {}'
python - <<PY
import pandas as pd
ds = [pd.read_csv(f"$O/kidney_threshold_calibration_{u}.csv") for u in ("expected_entropy", "total_entropy")]
for d in ds: assert (d.case_id != "SUMMARY").sum() == 97 and (d.case_id == "SUMMARY").sum() == 1
pd.concat(ds, ignore_index=True).to_csv("$ALL", index=False)
PY

echo "=== [F] $(date) ==="
python -u /home/grolo/thesis_pipeline/3_uncertainty/evaluation/evaluate_threshold.py --dataset_id 904 --pred_dir $TEST --gt_dir $RAW/labelsTs --unc_dir $MAPS \
  --thresh_csv $ALL --output_dir $EV --unc_types expected_entropy,total_entropy > $L/blockF_decomposition.log 2>&1

echo "=== [statistica] 14 + 2 misure $(date) ==="
mkdir -p $R
python - <<PY
import pandas as pd
a = pd.read_csv("$O/kidney_eval_threshold_ALL/metrics_per_case_4.csv"); b = pd.read_csv("$EV/metrics_per_case_4.csv")
a = a[~a.uncertainty_type.str.startswith("combined_")]
assert a.uncertainty_type.nunique() == 14 and b.uncertainty_type.nunique() == 2 and set(a.case_id) == set(b.case_id)
pd.concat([a, b], ignore_index=True).to_csv("$R/metrics_per_case_16.csv", index=False)
PY
python /home/grolo/thesis_pipeline/4_analysis/uncertainty_ranking_stats.py --per_case_csv $R/metrics_per_case_16.csv --out_dir $R > $R/ranking_stats.log 2>&1
echo "=== SCOMPOSIZIONE 904 COMPLETA $(date) ==="
