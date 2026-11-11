#!/bin/bash
# Pipeline di incertezza del checkpoint ensemble del rene: temperature scaling, mappe, DEU/EBCU, soglie, valutazione
# e classifica statistica.
# Uso: ./pipeline_kidney904_checkpoint.sh   (RESUME_D=1, RESUME_D2=1, RESUME_E=1 ripartono da mappe, DEU/EBCU o soglie)
set -e
source /home/grolo/venv_explainability/bin/activate
export NUMPY_MADVISE_HUGEPAGE=0
cd /home/grolo
RAW=/mnt/sda2/grolo/nnUNet/raw/Dataset904_KidneyTSMechanical
P=/mnt/sda2/grolo/kidney/predictions
TEST=$P/checkpoint
VAL=$P/val_fold0_cyclr_scratch_T1
O=/mnt/sda2/grolo/kidney/checkpoint
L=$O/logs
MAPS=$O/kidney_cyclr_uncertainty_maps
mkdir -p $L

# Ripete un comando se il kernel lo uccide per memoria esaurita (exit 137), fino a 3 tentativi (i passi riscrivono
# gli stessi file, quindi ripeterli e' sicuro).
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

n=$(ls $VAL/*_logits.pt | wc -l); [ "$n" -eq 47 ] || { echo "val: $n logit invece di 47"; exit 1; }
n=$(ls $TEST/*_logits.pt | wc -l); [ "$n" -eq 97 ] || { echo "test: $n logit invece di 97"; exit 1; }
n=$(ls $TEST/fold_logits/*_logits.pt | wc -l); [ "$n" -eq 582 ] || { echo "test: $n logit di snapshot invece di 582"; exit 1; }

# RESUME_D=1: riparte dalle mappe (T* riletto dai CSV del temperature scaling)
if [ "$RESUME_D" != 1 ]; then
echo "=== [C] T* LOO sui 47 casi di validation, 2 processi $(date) ==="
for i in 0 1; do
  python -u /home/grolo/thesis_pipeline/3_uncertainty/temperature/loo_temperature_scaling.py --logits_dir $VAL --gt_dir $RAW/labelsTr --num_shards 2 --shard_index $i \
    --out_summary_csv $O/loo_T_summary_shard$i.csv --out_grid_csv $O/loo_T_grid_shard$i.csv > $L/blockC_shard$i.log 2>&1 &
done
wait
fi
T=$(python - <<EOF
import pandas as pd, sys
s = pd.concat([pd.read_csv(f"$O/loo_T_summary_shard{i}.csv") for i in range(2)]).sort_values("heldout_case")
g = pd.concat([pd.read_csv(f"$O/loo_T_grid_shard{i}.csv") for i in range(2)]).sort_values(["heldout_case", "split", "stage", "T"])
if len(s) != 47 or s.heldout_case.duplicated().any(): sys.exit(f"temperature scaling: {len(s)} righe invece di 47")
s.to_csv("$O/loo_T_summary.csv", index=False); g.to_csv("$O/loo_T_grid.csv", index=False)
t = float(s.T_star.median())
if not 0.5 < t < 10: sys.exit(f"T* = {t} fuori dall'intervallo plausibile")
print(f"{t:.6f}")
EOF
)
echo "=== T* checkpoint 904 = $T  ($(date)) ==="

# RESUME_D2=1: salta anche le mappe e riparte da DEU/EBCU (controllo del numero di mappe per misura)
if [ "$RESUME_D2" != 1 ]; then
echo "=== [D] mappe di incertezza sul test (tutte le misure) $(date) ==="
retry_oom python -u /home/grolo/thesis_pipeline/3_uncertainty/maps/total_uncertainty_measures.py --input_dir $TEST --output_dir $MAPS --T $T > $L/blockD_uncertainty_maps.log 2>&1
fi
for u in entropy mutual_information variance std epkl confidence_map sample_mean_uncertainty boundary_disagreement \
         max_fold_deviation anti_confidence_fn anti_confidence_fp anti_confidence_total; do
  n=$(ls $MAPS | grep -cE "^s[0-9]+_${u}\.nii\.gz$"); [ "$n" -eq 97 ] || { echo "mappe: $n mappe $u invece di 97"; exit 1; }
done

# RESUME_E=1: salta anche DEU/EBCU e riparte dalle soglie
if [ "$RESUME_E" != 1 ]; then
echo "=== [D2.1] entropy sui 47 casi di validation $(date) ==="
retry_oom python -u /home/grolo/thesis_pipeline/3_uncertainty/maps/total_uncertainty_measures.py --input_dir $VAL --output_dir $O/kidney_cyclr_uncertainty_maps_VAL \
  --T $T --measures entropy > $L/blockD2_val_entropy.log 2>&1
n=$(ls $O/kidney_cyclr_uncertainty_maps_VAL | grep -cE "^s[0-9]+_entropy\.nii\.gz$"); [ "$n" -eq 47 ] || { echo "D2.1: $n mappe invece di 47"; exit 1; }
echo "=== [D2.2] baseline E[U|d] $(date) ==="
retry_oom python -u /home/grolo/thesis_pipeline/3_uncertainty/distance_aware/build_baseline_EU_given_d.py --entropy_dir $O/kidney_cyclr_uncertainty_maps_VAL \
  --pred_dir $VAL --out_dir $O/baseline_EU --uncertainty_type entropy --max_mm 60 > $L/blockD2_baseline.log 2>&1
echo "=== [D2.3] DEU $(date) ==="
retry_oom python -u /home/grolo/thesis_pipeline/3_uncertainty/distance_aware/compute_DEU.py --baseline_npz $O/baseline_EU/baseline_EU_given_d_mm_inside_outside.npz \
  --uncertainty_dir $MAPS --pred_dir $TEST --out_dir $O/kidney_cyclr_deu_maps --uncertainty_type entropy \
  > $L/blockD2_compute_DEU.log 2>&1
echo "=== [D2.4] EBCU $(date) ==="
retry_oom python -u /home/grolo/thesis_pipeline/3_uncertainty/distance_aware/make_ebcu_maps.py --baseline_npz $O/baseline_EU/baseline_EU_given_d_mm_inside_outside.npz \
  --entropy_dir $MAPS --pred_dir $TEST --out_dir $O/kidney_cyclr_ebcu_maps --uncertainty_type entropy \
  > $L/blockD2_make_ebcu.log 2>&1
for f in $O/kidney_cyclr_deu_maps/*_DEU_entropy.nii.gz $O/kidney_cyclr_ebcu_maps/*_ebcu_entropy.nii.gz; do
  ln -sf "$f" "$MAPS/$(basename $f)"
done
fi
for u in DEU_entropy ebcu_entropy; do
  n=$(ls $MAPS | grep -cE "^s[0-9]+_${u}\.nii\.gz$"); [ "$n" -eq 97 ] || { echo "D2: $n mappe $u invece di 97"; exit 1; }
done
echo "=== C, D, D2 COMPLETI: $(date) ==="

# Soglie: 14 misure, 2 in parallelo (~12 GB ciascuna), con --cache_case_scores
echo "=== [E] threshold calibration LOO, 14 misure, 2 in parallelo $(date) ==="
export RAW TEST MAPS O L
run_e() {
  # Salta le misure gia' complete (CSV con riga SUMMARY); se il kernel uccide la run per OOM (exit 137)
  # la ripete, fino a 3 tentativi (lo script scrive il CSV solo alla fine, quindi una run uccisa non lascia righe)
  local U=$1 CSV=$O/kidney_cyclr_threshold_calibration_$1.csv rc=0
  if [ -f $CSV ] && grep -q ",SUMMARY," $CSV; then echo "=== $U gia' completo, saltato: $(date) ===" >> $L/blockE_threshold_${U}.log; return 0; fi
  for a in 1 2 3; do
    rm -f $CSV
    python -u /home/grolo/thesis_pipeline/3_uncertainty/thresholds/Calibrate_threshold_loo_v2.py --prediction_dir $TEST --gt_dir $RAW/labelsTs \
      --ct_dir $RAW/imagesTs --uncertainty_dir $MAPS --uncertainty_type $U \
      --output_csv $CSV --qa_dir $O/kidney_cyclr_qa_${U} \
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
ds = [pd.read_csv(f"$O/kidney_cyclr_threshold_calibration_{u}.csv") for u in M]
for u, d in zip(M, ds):
    assert (d.case_id != 'SUMMARY').sum() == 97 and (d.case_id == 'SUMMARY').sum() == 1, u
pd.concat(ds, ignore_index=True).to_csv("$O/kidney_cyclr_threshold_calibration_ALL.csv", index=False)
print("ALL csv ok")
EOF

echo "=== [F] evaluate_threshold $(date) ==="
python -u /home/grolo/thesis_pipeline/3_uncertainty/evaluation/evaluate_threshold.py --dataset_id 904 --pred_dir $TEST --gt_dir $RAW/labelsTs \
  --unc_dir $MAPS --thresh_csv $O/kidney_cyclr_threshold_calibration_ALL.csv \
  --output_dir $O/kidney_cyclr_eval_threshold_ALL > $L/blockF_evaluate_threshold.log 2>&1

echo "=== [stat] $(date) ==="
python /home/grolo/thesis_pipeline/4_analysis/uncertainty_ranking_stats.py \
  --per_case_csv $O/kidney_cyclr_eval_threshold_ALL/metrics_per_case_4.csv --out_dir $O/ranking_stats \
  > $L/ranking_stats.log 2>&1
echo "=== PIPELINE COMPLETA: $(date) ==="
