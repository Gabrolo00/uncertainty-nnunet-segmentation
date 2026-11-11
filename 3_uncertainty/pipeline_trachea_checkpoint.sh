#!/bin/bash
# Pipeline di incertezza del checkpoint ensemble della trachea: temperature scaling, mappe, DEU/EBCU, soglie,
# valutazione, classifica statistica e confronto con il deep ensemble.
# Uso: ./pipeline_trachea_checkpoint.sh
set -e
cd /mnt/sda2/grolo   # log relativi in /mnt/sda2/grolo/logs
until grep -q "CODA GPU CHECKPOINT TRACHEA COMPLETA" logs/trachea_cyclr_master.log 2>/dev/null; do sleep 300; done
grep -q "inferenza test finita (exit 0)" logs/trachea_cyclr_master.log || { echo "inferenza test non riuscita"; exit 1; }
grep -q "inferenza validation finita (exit 0)" logs/trachea_cyclr_master.log || { echo "inferenza validation non riuscita"; exit 1; }

source /home/grolo/venv_explainability/bin/activate
export NUMPY_MADVISE_HUGEPAGE=0
cd /home/grolo
RAW=/mnt/sda2/grolo/nnUNet/raw/Dataset903_TracheaTS
P=/mnt/sda2/grolo/trachea/predictions
TEST=$P/checkpoint
VAL=$P/val_fold0_cyclr_scratch_T1
O=/mnt/sda2/grolo/trachea/checkpoint
L=$O/logs
MAPS=$O/trachea_cyclr_uncertainty_maps
[ -z "$(ls -A $O 2>/dev/null)" ] || { echo "$O contiene gia' file: non sovrascrivo"; exit 1; }
mkdir -p $L

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
cnt() { ls "$1" | grep -cE "^s[0-9]+_$2\.nii\.gz$"; }

n=$(ls $VAL/*_logits.pt | wc -l); [ "$n" -eq 48 ] || { echo "val: $n logit invece di 48"; exit 1; }
n=$(ls $TEST/*_logits.pt | wc -l); [ "$n" -eq 60 ] || { echo "test: $n logit invece di 60"; exit 1; }
n=$(ls $TEST/fold_logits/*_logits.pt | wc -l); [ "$n" -eq 360 ] || { echo "test: $n logit di snapshot invece di 360"; exit 1; }

echo "=== [C] T* LOO sui 48 casi di validation, 2 processi $(date) ==="
for i in 0 1; do
  python -u /home/grolo/thesis_pipeline/3_uncertainty/temperature/loo_temperature_scaling.py --logits_dir $VAL --gt_dir $RAW/labelsTr --num_shards 2 --shard_index $i \
    --out_summary_csv $O/loo_T_summary_shard$i.csv --out_grid_csv $O/loo_T_grid_shard$i.csv > $L/blockC_shard$i.log 2>&1 &
done
wait
T=$(python - <<EOF
import pandas as pd, sys, re
s = pd.concat([pd.read_csv(f"$O/loo_T_summary_shard{i}.csv") for i in range(2)]).sort_values("heldout_case")
g = pd.concat([pd.read_csv(f"$O/loo_T_grid_shard{i}.csv") for i in range(2)]).sort_values(["heldout_case", "split", "stage", "T"])
skipped = set()
for i in range(2):
    m = re.search(r"Skipping \d+ case\(s\) with empty ROI.*?: \[(.*?)\]", open(f"$L/blockC_shard{i}.log").read())
    if m: skipped |= set(x.strip(" '") for x in m.group(1).split(","))
if s.heldout_case.duplicated().any() or len(s) + len(skipped) != 48:
    sys.exit(f"temperature scaling: {len(s)} righe + {len(skipped)} saltati != 48")
s.to_csv("$O/loo_T_summary.csv", index=False); g.to_csv("$O/loo_T_grid.csv", index=False)
t = float(s.T_star.median())
if not 0.5 < t < 10: sys.exit(f"T* = {t} fuori dall'intervallo plausibile")
print(f"{t:.6f}")
EOF
)
echo "=== T* checkpoint trachea = $T  ($(date)) ==="

echo "=== [D] mappe sul test $(date) ==="
retry_oom python -u /home/grolo/thesis_pipeline/3_uncertainty/maps/total_uncertainty_measures.py --input_dir $TEST --output_dir $MAPS --T $T > $L/blockD_uncertainty_maps.log 2>&1
for u in entropy mutual_information variance std epkl confidence_map sample_mean_uncertainty boundary_disagreement \
         max_fold_deviation anti_confidence_fn anti_confidence_fp anti_confidence_total; do
  n=$(cnt $MAPS $u); [ "$n" -eq 60 ] || { echo "mappe: $n mappe $u invece di 60"; exit 1; }
done

echo "=== [D2] baseline dai 48 casi di validation, DEU, EBCU $(date) ==="
retry_oom python -u /home/grolo/thesis_pipeline/3_uncertainty/maps/total_uncertainty_measures.py --input_dir $VAL --output_dir $O/trachea_cyclr_uncertainty_maps_VAL \
  --T $T --measures entropy > $L/blockD2_val_entropy.log 2>&1
n=$(cnt $O/trachea_cyclr_uncertainty_maps_VAL entropy); [ "$n" -eq 48 ] || { echo "D2.1: $n mappe invece di 48"; exit 1; }
retry_oom python -u /home/grolo/thesis_pipeline/3_uncertainty/distance_aware/build_baseline_EU_given_d.py --entropy_dir $O/trachea_cyclr_uncertainty_maps_VAL \
  --pred_dir $VAL --out_dir $O/baseline_EU --uncertainty_type entropy --max_mm 60 > $L/blockD2_baseline.log 2>&1
retry_oom python -u /home/grolo/thesis_pipeline/3_uncertainty/distance_aware/compute_DEU.py --baseline_npz $O/baseline_EU/baseline_EU_given_d_mm_inside_outside.npz \
  --uncertainty_dir $MAPS --pred_dir $TEST --out_dir $O/trachea_cyclr_deu_maps --uncertainty_type entropy > $L/blockD2_compute_DEU.log 2>&1
retry_oom python -u /home/grolo/thesis_pipeline/3_uncertainty/distance_aware/make_ebcu_maps.py --baseline_npz $O/baseline_EU/baseline_EU_given_d_mm_inside_outside.npz \
  --entropy_dir $MAPS --pred_dir $TEST --out_dir $O/trachea_cyclr_ebcu_maps --uncertainty_type entropy > $L/blockD2_make_ebcu.log 2>&1
for f in $O/trachea_cyclr_deu_maps/*_DEU_entropy.nii.gz $O/trachea_cyclr_ebcu_maps/*_ebcu_entropy.nii.gz; do ln -sf "$f" "$MAPS/$(basename $f)"; done
for u in DEU_entropy ebcu_entropy; do n=$(cnt $MAPS $u); [ "$n" -eq 60 ] || { echo "D2: $n mappe $u invece di 60"; exit 1; }; done


# Su una sola riga: la variabile entra in una stringa Python piu' sotto
M14="entropy DEU_entropy ebcu_entropy mutual_information epkl std variance sample_mean_uncertainty confidence_map boundary_disagreement max_fold_deviation anti_confidence_fn anti_confidence_fp anti_confidence_total"
for u in $M14; do n=$(cnt $MAPS $u); [ "$n" -eq 60 ] || { echo "$n mappe $u invece di 60"; exit 1; }; done

echo "=== [E] 17 misure, 2 in parallelo $(date) ==="
export RAW TEST MAPS O L
run_e() {
  local U=$1 CSV=$O/trachea_cyclr_threshold_calibration_$1.csv rc=0
  if [ -f $CSV ] && grep -q ",SUMMARY," $CSV; then return 0; fi
  for a in 1 2 3; do
    rm -f $CSV
    python -u /home/grolo/thesis_pipeline/3_uncertainty/thresholds/Calibrate_threshold_loo_v2.py --prediction_dir $TEST --gt_dir $RAW/labelsTs --ct_dir $RAW/imagesTs \
      --uncertainty_dir $MAPS --uncertainty_type $U --output_csv $CSV --qa_dir $O/trachea_cyclr_qa_${U} \
      --roi_mode buffer_15mm --cache_case_scores >> $L/blockE_threshold_${U}.log 2>&1
    rc=$?; echo "=== $U finito (exit $rc, tentativo $a): $(date) ===" >> $L/blockE_threshold_${U}.log
    [ $rc -ne 137 ] && return $rc
    sleep 300
  done
  return $rc
}
export -f run_e
echo $M14 | tr ' ' '\n' | xargs -P 2 -I{} bash -c 'run_e {}'
python - <<EOF
import pandas as pd
M = "$M14".split()
ds = [pd.read_csv(f"$O/trachea_cyclr_threshold_calibration_{u}.csv") for u in M]
for u, d in zip(M, ds): assert (d.case_id != "SUMMARY").sum() == 60 and (d.case_id == "SUMMARY").sum() == 1, u
pd.concat(ds, ignore_index=True).to_csv("$O/trachea_cyclr_threshold_calibration_ALL.csv", index=False)
print("ALL csv ok (17 misure)")
EOF

echo "=== [F] $(date) ==="
retry_oom python -u /home/grolo/thesis_pipeline/3_uncertainty/evaluation/evaluate_threshold.py --dataset_id 903cyclr --pred_dir $TEST --gt_dir $RAW/labelsTs --unc_dir $MAPS \
  --thresh_csv $O/trachea_cyclr_threshold_calibration_ALL.csv --output_dir $O/trachea_cyclr_eval_threshold_ALL \
  > $L/blockF_evaluate_threshold.log 2>&1

echo "=== [statistica] 14 misure e 17 misure $(date) ==="
mkdir -p $O/ranking_stats
python - <<EOF
import pandas as pd
d = pd.read_csv("$O/trachea_cyclr_eval_threshold_ALL/metrics_per_case_4.csv")
d.to_csv("$O/ranking_stats/metrics_per_case_14.csv", index=False)
EOF
python /home/grolo/thesis_pipeline/4_analysis/uncertainty_ranking_stats.py --per_case_csv $O/ranking_stats/metrics_per_case_14.csv --out_dir $O/ranking_stats > $L/ranking_stats.log 2>&1

echo "=== [confronto deep vs checkpoint, trachea] $(date) ==="
cd /mnt/sda2/grolo   # log relativi in /mnt/sda2/grolo/logs
python /home/grolo/thesis_pipeline/4_analysis/compare_deep_vs_checkpoint.py --config trachea > logs/compare_deep_vs_checkpoint_trachea.log 2>&1
python /home/grolo/thesis_pipeline/4_analysis/table_maps_deep_vs_checkpoint.py --config trachea > logs/table_maps_deep_vs_checkpoint_trachea.log 2>&1
echo "=== PIPELINE CHECKPOINT TRACHEA COMPLETA: $(date) ==="
