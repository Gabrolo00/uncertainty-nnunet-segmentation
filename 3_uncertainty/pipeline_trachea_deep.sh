#!/bin/bash
# Pipeline di incertezza del deep ensemble della trachea: temperature scaling, mappe, DEU/EBCU, soglie, valutazione
# e classifica statistica.
# Uso: ./pipeline_trachea_deep.sh
set -e
cd /mnt/sda2/grolo   # log relativi in /mnt/sda2/grolo/logs
until grep -q "CODA DOMENICA COMPLETA" logs/queue_sunday_master.log 2>/dev/null; do sleep 300; done
grep -q "collect OOF trachea (exit 0)" logs/queue_sunday_master.log || { echo "collect OOF trachea non riuscito"; exit 1; }
grep -q "test trachea finito (exit 0)" logs/queue_sunday_gpu1.log || { echo "inferenza test trachea non riuscita"; exit 1; }

source /home/grolo/venv_explainability/bin/activate
export NUMPY_MADVISE_HUGEPAGE=0  # evita il rallentamento delle huge pages di numpy sulle CT grandi
cd /home/grolo
RAW=/mnt/sda2/grolo/nnUNet/raw/Dataset903_TracheaTS
P=/mnt/sda2/grolo/trachea/predictions
OOF=$P/oof_logits_clean_scratch
OOFRAW=$P/oof_train_logits_scratch
TEST=$P/deep
O=/mnt/sda2/grolo/trachea/deep
L=$O/logs
MAPS=$O/trachea_uncertainty_maps
[ -z "$(ls -A $O 2>/dev/null)" ] || { echo "$O contiene gia' file: non sovrascrivo"; exit 1; }
mkdir -p $L

n=$(ls $OOF/*_logits.pt | wc -l); [ "$n" -eq 236 ] || { echo "OOF: $n logit invece di 236"; exit 1; }
n=$(ls $OOFRAW/*.nii.gz | wc -l); [ "$n" -eq 236 ] || { echo "OOF: $n predizioni invece di 236"; exit 1; }
n=$(ls $TEST/*_logits.pt | wc -l); [ "$n" -eq 60 ] || { echo "test: $n logit invece di 60"; exit 1; }
n=$(ls $TEST/fold_logits/*_logits.pt | wc -l); [ "$n" -eq 300 ] || { echo "test: $n fold_logits invece di 300"; exit 1; }

echo "=== [C] T* LOO sui 236 OOF, 4 processi $(date) ==="
for i in 0 1 2 3; do
  python -u /home/grolo/thesis_pipeline/3_uncertainty/temperature/loo_temperature_scaling.py --logits_dir $OOF --gt_dir $RAW/labelsTr --num_shards 4 --shard_index $i \
    --out_summary_csv $O/loo_T_summary_shard$i.csv --out_grid_csv $O/loo_T_grid_shard$i.csv > $L/blockC_shard$i.log 2>&1 &
done
wait
# I casi con ROI vuota (reference e predizione vuote) sono saltati dal temperature scaling: righe = 236 - saltati
T=$(python - <<EOF
import pandas as pd, sys, re
s = pd.concat([pd.read_csv(f"$O/loo_T_summary_shard{i}.csv") for i in range(4)]).sort_values("heldout_case")
g = pd.concat([pd.read_csv(f"$O/loo_T_grid_shard{i}.csv") for i in range(4)]).sort_values(["heldout_case", "split", "stage", "T"])
skipped = set()
for i in range(4):
    m = re.search(r"Skipping \d+ case\(s\) with empty ROI.*?: \[(.*?)\]", open(f"$L/blockC_shard{i}.log").read())
    if m: skipped |= set(x.strip(" '") for x in m.group(1).split(","))
if s.heldout_case.duplicated().any() or len(s) + len(skipped) != 236:
    sys.exit(f"temperature scaling: {len(s)} righe + {len(skipped)} saltati != 236")
s.to_csv("$O/loo_T_summary.csv", index=False); g.to_csv("$O/loo_T_grid.csv", index=False)
t = float(s.T_star.median())
if not 0.5 < t < 10: sys.exit(f"T* = {t} fuori dall'intervallo plausibile")
print(f"{t:.6f}")
EOF
)
echo "=== T* trachea scratch = $T  ($(date)) ==="

echo "=== [D] mappe di incertezza sul test $(date) ==="
python -u /home/grolo/thesis_pipeline/3_uncertainty/maps/total_uncertainty_measures.py --input_dir $TEST --output_dir $MAPS --T $T > $L/blockD_uncertainty_maps.log 2>&1
n=$(ls $MAPS | grep -cE "^s[0-9]+_entropy\.nii\.gz$"); [ "$n" -eq 60 ] || { echo "mappe: $n mappe entropy invece di 60"; exit 1; }

echo "=== [D2.1] entropy sui 236 OOF $(date) ==="
python -u /home/grolo/thesis_pipeline/3_uncertainty/maps/total_uncertainty_measures.py --input_dir $OOFRAW --output_dir $O/trachea_uncertainty_maps_TRAIN \
  --T $T --measures entropy > $L/blockD2_train_entropy.log 2>&1
n=$(ls $O/trachea_uncertainty_maps_TRAIN | grep -cE "^s[0-9]+_entropy\.nii\.gz$"); [ "$n" -eq 236 ] || { echo "D2.1: $n mappe invece di 236"; exit 1; }
echo "=== [D2.2] baseline E[U|d] $(date) ==="
python -u /home/grolo/thesis_pipeline/3_uncertainty/distance_aware/build_baseline_EU_given_d.py --entropy_dir $O/trachea_uncertainty_maps_TRAIN \
  --pred_dir $OOFRAW --out_dir $O/baseline_EU --uncertainty_type entropy --max_mm 60 > $L/blockD2_baseline.log 2>&1
echo "=== [D2.3] DEU $(date) ==="
python -u /home/grolo/thesis_pipeline/3_uncertainty/distance_aware/compute_DEU.py --baseline_npz $O/baseline_EU/baseline_EU_given_d_mm_inside_outside.npz \
  --uncertainty_dir $MAPS --pred_dir $TEST --out_dir $O/trachea_deu_maps --uncertainty_type entropy > $L/blockD2_compute_DEU.log 2>&1
echo "=== [D2.4] EBCU $(date) ==="
python -u /home/grolo/thesis_pipeline/3_uncertainty/distance_aware/make_ebcu_maps.py --baseline_npz $O/baseline_EU/baseline_EU_given_d_mm_inside_outside.npz \
  --entropy_dir $MAPS --pred_dir $TEST --out_dir $O/trachea_ebcu_maps --uncertainty_type entropy > $L/blockD2_make_ebcu.log 2>&1
for f in $O/trachea_deu_maps/*_DEU_entropy.nii.gz $O/trachea_ebcu_maps/*_ebcu_entropy.nii.gz; do ln -sf "$f" "$MAPS/$(basename $f)"; done
for u in DEU_entropy ebcu_entropy; do
  n=$(ls $MAPS | grep -cE "^s[0-9]+_${u}\.nii\.gz$"); [ "$n" -eq 60 ] || { echo "D2: $n mappe $u invece di 60"; exit 1; }
done

echo "=== [E] threshold calibration LOO, 14 misure, 2 in parallelo $(date) ==="
export RAW TEST MAPS O L
run_e() {
  # Salta le misure gia' complete (CSV con riga SUMMARY); se il kernel uccide la run per OOM (exit 137)
  # la ripete, fino a 3 tentativi (lo script scrive il CSV solo alla fine, quindi una run uccisa non lascia righe)
  local U=$1 CSV=$O/trachea_threshold_calibration_$1.csv rc=0
  if [ -f $CSV ] && grep -q ",SUMMARY," $CSV; then echo "=== $U gia' completo, saltato: $(date) ===" >> $L/blockE_threshold_${U}.log; return 0; fi
  for a in 1 2 3; do
    rm -f $CSV
    python -u /home/grolo/thesis_pipeline/3_uncertainty/thresholds/Calibrate_threshold_loo_v2.py --prediction_dir $TEST --gt_dir $RAW/labelsTs \
      --ct_dir $RAW/imagesTs --uncertainty_dir $MAPS --uncertainty_type $U \
      --output_csv $CSV --qa_dir $O/trachea_qa_${U} \
      --roi_mode buffer_15mm --cache_case_scores >> $L/blockE_threshold_${U}.log 2>&1
    rc=$?
    echo "=== $U finito (exit $rc, tentativo $a): $(date) ===" >> $L/blockE_threshold_${U}.log
    [ $rc -ne 137 ] && return $rc
    sleep 300
  done
  return $rc
}
export -f run_e
printf '%s\n' entropy DEU_entropy ebcu_entropy mutual_information epkl std variance \
  sample_mean_uncertainty confidence_map boundary_disagreement max_fold_deviation \
  anti_confidence_fn anti_confidence_fp anti_confidence_total | xargs -P 2 -I{} bash -c 'run_e {}'

python - <<EOF
import pandas as pd
M = ['entropy','DEU_entropy','ebcu_entropy','variance','std','mutual_information','epkl',
     'sample_mean_uncertainty','confidence_map','boundary_disagreement','max_fold_deviation',
     'anti_confidence_fn','anti_confidence_fp','anti_confidence_total']
ds = [pd.read_csv(f"$O/trachea_threshold_calibration_{u}.csv") for u in M]
for u, d in zip(M, ds):
    assert (d.case_id != 'SUMMARY').sum() == 60 and (d.case_id == 'SUMMARY').sum() == 1, u
pd.concat(ds, ignore_index=True).to_csv("$O/trachea_threshold_calibration_ALL.csv", index=False)
print("ALL csv ok")
EOF

echo "=== [F] evaluate_threshold $(date) ==="
python -u /home/grolo/thesis_pipeline/3_uncertainty/evaluation/evaluate_threshold.py --dataset_id 903 --pred_dir $TEST --gt_dir $RAW/labelsTs \
  --unc_dir $MAPS --thresh_csv $O/trachea_threshold_calibration_ALL.csv \
  --output_dir $O/trachea_eval_threshold_ALL > $L/blockF_evaluate_threshold.log 2>&1

echo "=== [stat] $(date) ==="
python /home/grolo/thesis_pipeline/4_analysis/uncertainty_ranking_stats.py \
  --per_case_csv $O/trachea_eval_threshold_ALL/metrics_per_case_4.csv --out_dir $O/ranking_stats \
  > $L/ranking_stats.log 2>&1
echo "=== PIPELINE COMPLETA: $(date) ==="
