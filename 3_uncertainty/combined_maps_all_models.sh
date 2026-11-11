#!/bin/bash
# Mappe combinate per tutti i modelli: creazione, valutazione alla soglia 1 e classifica statistica con le 14 misure.
# Uso: ./combined_maps_all_models.sh
set -e
source /home/grolo/venv_explainability/bin/activate
export NUMPY_MADVISE_HUGEPAGE=0
cd /home/grolo
TS=/mnt/sda2/grolo
NR=/mnt/sda2/grolo/nnUNet/raw
NP=/mnt/sda2/grolo
H=/mnt/sda2/grolo/humanitas_source
COMB="combined_thr_switch_entropy_std,combined_thr_max_entropy_std,combined_thr_switch_DEU_MI"

retry_oom() {
  local a rc
  for a in 1 2 3; do
    "$@" && return 0
    rc=$?; echo "=== exit $rc (tentativo $a): $(date) ==="
    [ $rc -ne 137 ] && return $rc
    sleep 300
  done
  return $rc
}

run() {  # nome O MAPS TEST GT TAU_CSV PER_CASE_14 OUT_EVAL OUT_RANK [casi_da_tenere.txt]
  local NAME=$1 O=$2 MAPS=$3 TEST=$4 GT=$5 TAU=$6 A14=$7 EV=$8 R=$9 KEEP=${10:-}
  # I modelli gia' completi si saltano
  if [ -f $R/step1_stats_all.csv ]; then echo "=== [$NAME] gia' completo, saltato $(date) ==="; return 0; fi
  echo "=== [$NAME] mappe $(date) ==="
  retry_oom python -u /home/grolo/thesis_pipeline/3_uncertainty/combined/make_threshold_scaled_combined_maps.py --maps_dir $MAPS --pred_dir $TEST --tau_csv $TAU \
    --out_dir $O/combined_thr_maps --link_into_maps > $O/combined_thr_maps.log 2>&1
  echo "=== [$NAME] valutazione $(date) ==="
  [ -z "$(ls -A $EV 2>/dev/null)" ] || { echo "$EV contiene gia' file: non sovrascrivo"; exit 1; }
  retry_oom python -u /home/grolo/thesis_pipeline/3_uncertainty/evaluation/evaluate_threshold.py --dataset_id $NAME --pred_dir $TEST --gt_dir $GT --unc_dir $MAPS \
    --thresh_csv $O/combined_thr_maps/thresholds_for_evaluate.csv --output_dir $EV --unc_types $COMB > $EV.log 2>&1
  mkdir -p $R
  python - <<EOF
import pandas as pd
a = pd.read_csv("$A14"); a = a[~a.uncertainty_type.str.startswith("combined_")]
b = pd.read_csv("$EV/metrics_per_case_4.csv")
if "$KEEP":
    keep = set(open("$KEEP").read().split()); a = a[a.case_id.isin(keep)]; b = b[b.case_id.isin(keep)]
assert a.uncertainty_type.nunique() == 14 and b.uncertainty_type.nunique() == 3
assert set(a.case_id) == set(b.case_id)
pd.concat([a, b], ignore_index=True).to_csv("$R/metrics_per_case_17.csv", index=False)
print("$NAME: 14 + 3 misure, casi", a.case_id.nunique())
EOF
  python /home/grolo/thesis_pipeline/4_analysis/uncertainty_ranking_stats.py --per_case_csv $R/metrics_per_case_17.csv --out_dir $R > $R/ranking_stats.log 2>&1
  echo "=== [$NAME] completo $(date) ==="
}

run kidney904_deep $TS/kidney/deep $TS/kidney/deep/kidney_uncertainty_maps $NP/kidney/predictions/deep \
  $NR/Dataset904_KidneyTSMechanical/labelsTs $TS/kidney/deep/kidney_threshold_calibration_ALL.csv \
  $TS/kidney/deep/kidney_eval_threshold_ALL/metrics_per_case_4.csv $TS/kidney/deep/kidney_eval_threshold_combined_thr \
  $TS/kidney/deep/ranking_stats_with_combined
run trachea_deep $TS/trachea/deep $TS/trachea/deep/trachea_uncertainty_maps \
  $NP/trachea/predictions/deep $NR/Dataset903_TracheaTS/labelsTs \
  $TS/trachea/deep/trachea_threshold_calibration_ALL.csv \
  $TS/trachea/deep/trachea_eval_threshold_ALL/metrics_per_case_4.csv \
  $TS/trachea/deep/trachea_eval_threshold_combined_thr $TS/trachea/deep/ranking_stats_with_combined
run kidney904_checkpoint $TS/kidney/checkpoint $TS/kidney/checkpoint/kidney_cyclr_uncertainty_maps \
  $NP/kidney/predictions/checkpoint $NR/Dataset904_KidneyTSMechanical/labelsTs \
  $TS/kidney/checkpoint/kidney_cyclr_threshold_calibration_ALL.csv \
  $TS/kidney/checkpoint/kidney_cyclr_eval_threshold_ALL/metrics_per_case_4.csv \
  $TS/kidney/checkpoint/kidney_cyclr_eval_threshold_combined_thr $TS/kidney/checkpoint/ranking_stats_with_combined

# Test esterno: soglie del test interno (tau_from_TS_deep904.csv), solo contorni clinici nella statistica
W=/mnt/sda2/grolo/kidney/external
python -c "
import pandas as pd
c = pd.read_csv('$H/cohort.csv', dtype=str).fillna('')
open('$W/eval/clinical_cases.txt', 'w').write('\n'.join(c[(c.excluded == '') & (c.gt_source == 'Rene (clinico)')].case))"
run humanitas904 $W $W/maps_deep $W/pred_deep $W/cropped/labelsTs $W/eval/tau_from_TS_deep904.csv \
  $W/eval/evaluate_threshold_deep904/metrics_per_case_4.csv $W/eval/evaluate_threshold_deep904_combined_thr \
  $W/eval/ranking_stats_with_combined $W/eval/clinical_cases.txt

# trachea, checkpoint ensemble
run trachea_checkpoint $TS/trachea/checkpoint $TS/trachea/checkpoint/trachea_cyclr_uncertainty_maps \
  $NP/trachea/predictions/checkpoint $NR/Dataset903_TracheaTS/labelsTs \
  $TS/trachea/checkpoint/trachea_cyclr_threshold_calibration_ALL.csv \
  $TS/trachea/checkpoint/trachea_cyclr_eval_threshold_ALL/metrics_per_case_4.csv \
  $TS/trachea/checkpoint/trachea_cyclr_eval_threshold_combined_thr $TS/trachea/checkpoint/ranking_stats_with_combined
echo "=== MAPPE COMBINATE CON SOGLIA PROPRIA: TUTTO COMPLETO $(date) ==="
