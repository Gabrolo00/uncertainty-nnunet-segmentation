#!/bin/bash
# Pipeline di incertezza del deep ensemble del rene: temperature scaling, mappe, DEU/EBCU, soglie, valutazione e
# classifica statistica.
# Uso: ./pipeline_kidney904_deep.sh   (RESUME_D2=1, RESUME_E=1 ripartono da DEU/EBCU o dalle soglie)
set -e
cd /mnt/sda2/grolo   # log relativi in /mnt/sda2/grolo/logs
until grep -q "OOF COMPLETO (exit 0)" logs/oof_904_master.log && grep -q "deep 904 inferenza test finito (exit 0)" logs/queue_904_gpu0.log; do
  grep -q "FALLITO\|fallito" logs/oof_904_master.log && { echo "OOF fallito"; exit 1; }
  grep -q "deep 904 inferenza test finito (exit [1-9]" logs/queue_904_gpu0.log && { echo "inferenza test fallita"; exit 1; }
  sleep 300; done
source /home/grolo/venv_explainability/bin/activate
# Niente huge pages per gli array numpy: con la RAM frammentata il kernel passa gran parte del tempo a compattare la
# memoria. Cambia solo l'allocazione della memoria, non i risultati.
export NUMPY_MADVISE_HUGEPAGE=0
cd /home/grolo
RAW=/mnt/sda2/grolo/nnUNet/raw/Dataset904_KidneyTSMechanical
P=/mnt/sda2/grolo/kidney/predictions
OOF=$P/oof_logits_clean
TEST=$P/deep
O=/mnt/sda2/grolo/kidney/deep
L=$O/logs
MAPS=$O/kidney_uncertainty_maps
mkdir -p $L

n=$(ls $OOF/*_logits.pt | wc -l); [ "$n" -eq 234 ] || { echo "OOF: $n logit invece di 234"; exit 1; }
n=$(ls $TEST/*_logits.pt | wc -l); [ "$n" -eq 97 ] || { echo "test: $n logit invece di 97"; exit 1; }
n=$(ls $TEST/fold_logits/*_logits.pt | wc -l); [ "$n" -eq 485 ] || { echo "test: $n fold_logits invece di 485"; exit 1; }

# RESUME_D2=1: riparte da DEU/EBCU (T* riletto dai CSV del temperature scaling)
if [ "$RESUME_D2" != 1 ]; then
echo "=== [C] T* LOO sui 234 OOF, 4 processi $(date) ==="
for i in 0 1 2 3; do
  python -u /home/grolo/thesis_pipeline/3_uncertainty/temperature/loo_temperature_scaling.py --logits_dir $OOF --gt_dir $RAW/labelsTr --num_shards 4 --shard_index $i \
    --out_summary_csv $O/loo_T_summary_shard$i.csv --out_grid_csv $O/loo_T_grid_shard$i.csv > $L/blockC_shard$i.log 2>&1 &
done
wait  # set -e non vede gli exit code dei processi in background: li controlla il merge
fi
T=$(python - <<EOF
import pandas as pd, sys
s = pd.concat([pd.read_csv(f"$O/loo_T_summary_shard{i}.csv") for i in range(4)]).sort_values("heldout_case")
g = pd.concat([pd.read_csv(f"$O/loo_T_grid_shard{i}.csv") for i in range(4)]).sort_values(["heldout_case", "split", "stage", "T"])
if len(s) != 234 or s.heldout_case.duplicated().any(): sys.exit(f"temperature scaling: {len(s)} righe invece di 234")
s.to_csv("$O/loo_T_summary.csv", index=False); g.to_csv("$O/loo_T_grid.csv", index=False)
t = float(s.T_star.median())
if not 0.5 < t < 10: sys.exit(f"T* = {t} fuori dall'intervallo plausibile")
print(f"{t:.6f}")
EOF
)
echo "=== T* deep 904 = $T  ($(date)) ==="

if [ "$RESUME_D2" != 1 ]; then
echo "=== [D] mappe di incertezza sul test (tutte le misure) $(date) ==="
python -u /home/grolo/thesis_pipeline/3_uncertainty/maps/total_uncertainty_measures.py --input_dir $TEST --output_dir $MAPS --T $T > $L/blockD_uncertainty_maps.log 2>&1
fi
n=$(ls $MAPS/*_entropy.nii.gz | wc -l); [ "$n" -ge 97 ] || { echo "mappe: solo $n mappe entropy"; exit 1; }

# RESUME_E=1 (con RESUME_D2=1): riparte dalle soglie
if [ "$RESUME_E" != 1 ]; then
echo "=== [D2.1] entropy sui 234 OOF $(date) ==="
# Input = oof_train_logits (logit + predizione .nii.gz): total_uncertainty_measures.py salta i casi senza .nii.gz.
python -u /home/grolo/thesis_pipeline/3_uncertainty/maps/total_uncertainty_measures.py --input_dir $P/oof_train_logits --output_dir $O/kidney_uncertainty_maps_TRAIN \
  --T $T --measures entropy > $L/blockD2_train_entropy.log 2>&1
n=$(ls $O/kidney_uncertainty_maps_TRAIN/*_entropy.nii.gz | wc -l); [ "$n" -eq 234 ] || { echo "D2.1: $n mappe entropy OOF invece di 234"; exit 1; }
echo "=== [D2.2] baseline E[U|d] $(date) ==="
python -u /home/grolo/thesis_pipeline/3_uncertainty/distance_aware/build_baseline_EU_given_d.py --entropy_dir $O/kidney_uncertainty_maps_TRAIN \
  --pred_dir $P/oof_train_logits --out_dir $O/baseline_EU --uncertainty_type entropy --max_mm 60 \
  > $L/blockD2_baseline.log 2>&1
echo "=== [D2.3] DEU $(date) ==="
python -u /home/grolo/thesis_pipeline/3_uncertainty/distance_aware/compute_DEU.py --baseline_npz $O/baseline_EU/baseline_EU_given_d_mm_inside_outside.npz \
  --uncertainty_dir $MAPS --pred_dir $TEST --out_dir $O/kidney_deu_maps --uncertainty_type entropy \
  > $L/blockD2_compute_DEU.log 2>&1
echo "=== [D2.4] EBCU $(date) ==="
python -u /home/grolo/thesis_pipeline/3_uncertainty/distance_aware/make_ebcu_maps.py --baseline_npz $O/baseline_EU/baseline_EU_given_d_mm_inside_outside.npz \
  --entropy_dir $MAPS --pred_dir $TEST --out_dir $O/kidney_ebcu_maps --uncertainty_type entropy \
  > $L/blockD2_make_ebcu.log 2>&1
for f in $O/kidney_deu_maps/*_DEU_entropy.nii.gz $O/kidney_ebcu_maps/*_ebcu_entropy.nii.gz; do
  ln -sf "$f" "$MAPS/$(basename $f)"
done
fi
for u in DEU_entropy ebcu_entropy; do
  n=$(ls $MAPS/*_${u}.nii.gz | wc -l); [ "$n" -eq 97 ] || { echo "D2: $n mappe $u invece di 97"; exit 1; }
done

echo "=== [E] threshold calibration LOO, 14 misure, 2 in parallelo $(date) ==="
export RAW TEST MAPS O L
run_e() {
  local U=$1
  python -u /home/grolo/thesis_pipeline/3_uncertainty/thresholds/Calibrate_threshold_loo_v2.py --prediction_dir $TEST --gt_dir $RAW/labelsTs \
    --ct_dir $RAW/imagesTs --uncertainty_dir $MAPS --uncertainty_type $U \
    --output_csv $O/kidney_threshold_calibration_${U}.csv --qa_dir $O/kidney_qa_${U} \
    --roi_mode buffer_15mm --cache_case_scores > $L/blockE_threshold_${U}.log 2>&1
  echo "=== $U finito (exit $?): $(date) ===" >> $L/blockE_threshold_${U}.log
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
ds = [pd.read_csv(f"$O/kidney_threshold_calibration_{u}.csv") for u in M]
for u, d in zip(M, ds):
    assert (d.case_id != 'SUMMARY').sum() == 97 and (d.case_id == 'SUMMARY').sum() == 1, u
pd.concat(ds, ignore_index=True).to_csv("$O/kidney_threshold_calibration_ALL.csv", index=False)
print("ALL csv ok")
EOF

echo "=== [F] evaluate_threshold $(date) ==="
python -u /home/grolo/thesis_pipeline/3_uncertainty/evaluation/evaluate_threshold.py --dataset_id 904 --pred_dir $TEST --gt_dir $RAW/labelsTs \
  --unc_dir $MAPS --thresh_csv $O/kidney_threshold_calibration_ALL.csv \
  --output_dir $O/kidney_eval_threshold_ALL > $L/blockF_evaluate_threshold.log 2>&1

echo "=== [stat] $(date) ==="
python /home/grolo/thesis_pipeline/4_analysis/uncertainty_ranking_stats.py \
  --per_case_csv $O/kidney_eval_threshold_ALL/metrics_per_case_4.csv --out_dir $O/ranking_stats \
  > $L/ranking_stats.log 2>&1
echo "=== PIPELINE COMPLETA: $(date) ==="
