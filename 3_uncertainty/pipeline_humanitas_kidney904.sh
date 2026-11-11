#!/bin/bash
# Mappe di incertezza, DEU/EBCU, soglie trasferite e valutazione del deep ensemble del rene sul test esterno, con
# T*, tabella E[U | d] e soglie del test interno.
# Uso: ./pipeline_humanitas_kidney904.sh
set -e
B=/mnt/sda2/grolo/humanitas_source
W=/mnt/sda2/grolo/kidney/external
TS=/mnt/sda2/grolo
VENV=/home/grolo/venv_explainability/bin/python
MEDCAM=/home/aiart/anaconda3/envs/nnunet_medcam/bin/python
export NUMPY_MADVISE_HUGEPAGE=0  # evita il rallentamento delle huge pages di numpy sulle CT grandi
L=$W/logs_eval
mkdir -p $L $W/eval
cd /home/grolo

n=$(ls $W/pred_deep/*_logits.pt 2>/dev/null | wc -l); s=$(ls $W/pred_deep/*.nii.gz 2>/dev/null | wc -l)
[ "$n" -eq 110 ] && [ "$s" -eq 110 ] || { echo "pred_deep: $s segmentazioni, $n logit (attesi 110)"; exit 1; }
[ -f $TS/kidney/deep/kidney_threshold_calibration_ALL.csv ] || { echo "manca il CSV delle tau* del 904"; exit 1; }
for d in maps_deep deu_deep ebcu_deep; do
  [ -z "$(ls -A $W/$d 2>/dev/null)" ] || { echo "$W/$d contiene gia' file: non sovrascrivo"; exit 1; }
done

T=$($VENV -c "import pandas as pd; print(f'{pd.read_csv(\"$TS/kidney/deep/loo_T_summary.csv\").T_star.median():.6f}')")
BASE=$TS/kidney/deep/baseline_EU
echo "=== T* del 904 da TotalSegmentator: $T ($(date))"

count_maps() { ls "$1" | grep -cE "^hum[^_]+_$2\.nii\.gz$"; }

echo "=== [D] mappe di incertezza ($(date))"
$VENV -u /home/grolo/thesis_pipeline/3_uncertainty/maps/total_uncertainty_measures.py --input_dir $W/pred_deep --output_dir $W/maps_deep --T $T \
  > $L/blockD.log 2>&1
echo "=== [D2] DEU ed EBCU con la baseline del 904 ($(date))"
$VENV -u /home/grolo/thesis_pipeline/3_uncertainty/distance_aware/compute_DEU.py --baseline_npz $BASE/baseline_EU_given_d_mm_inside_outside.npz \
  --uncertainty_dir $W/maps_deep --pred_dir $W/pred_deep --out_dir $W/deu_deep --uncertainty_type entropy \
  > $L/blockD2_DEU.log 2>&1
$VENV -u /home/grolo/thesis_pipeline/3_uncertainty/distance_aware/make_ebcu_maps.py --baseline_npz $BASE/baseline_EU_given_d_mm_inside_outside.npz \
  --entropy_dir $W/maps_deep --pred_dir $W/pred_deep --out_dir $W/ebcu_deep --uncertainty_type entropy \
  > $L/blockD2_EBCU.log 2>&1
for f in $W/deu_deep/*_DEU_entropy.nii.gz $W/ebcu_deep/*_ebcu_entropy.nii.gz; do ln -sf "$f" "$W/maps_deep/$(basename $f)"; done
for u in entropy DEU_entropy ebcu_entropy variance mutual_information; do
  n=$(count_maps $W/maps_deep $u); [ "$n" -eq 110 ] || { echo "maps_deep: $n mappe $u (attese 110)"; exit 1; }
done

echo "=== soglie tau* del 904 da TotalSegmentator ($(date))"
$MEDCAM /home/grolo/thesis_pipeline/4_analysis/humanitas_eval.py --config 904 --stage thresholds > $L/thresholds.log 2>&1

echo "=== [F] evaluate_threshold ($(date))"
$VENV -u /home/grolo/thesis_pipeline/3_uncertainty/evaluation/evaluate_threshold.py --dataset_id HUM904 --pred_dir $W/pred_deep --gt_dir $W/cropped/labelsTs \
  --unc_dir $W/maps_deep --thresh_csv $W/eval/tau_from_TS_deep904.csv --output_dir $W/eval/evaluate_threshold_deep904 \
  > $L/blockF.log 2>&1

for st in segmentation calibration threshold_free; do
  echo "=== $st ($(date))"
  $MEDCAM /home/grolo/thesis_pipeline/4_analysis/humanitas_eval.py --config 904 --stage $st > $L/eval_$st.log 2>&1
done
echo "=== VALUTAZIONE COMPLETA $(date)"
