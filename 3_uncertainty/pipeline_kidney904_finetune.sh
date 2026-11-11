#!/bin/bash
# Pipeline del deep ensemble fine-tuned del rene: temperature scaling sui casi out-of-fold, mappe e DEU/EBCU sul
# test esterno, confronto con il modello from scratch.
# Uso: ./pipeline_kidney904_finetune.sh
set -e
cd /mnt/sda2/grolo   # log relativi in /mnt/sda2/grolo/logs
until grep -q "CODA GPU FINE-TUNING 904 COMPLETA" logs/ft904_after_master.log 2>/dev/null; do sleep 300; done
grep -q "collect OOF (exit 0)" logs/ft904_after_master.log || { echo "collect OOF non riuscito"; exit 1; }
grep -q "Humanitas ensemble fine-tuned finito (exit 0)" logs/ft904_after_gpu0.log || { echo "inferenza Humanitas non riuscita"; exit 1; }

source /home/grolo/venv_explainability/bin/activate
export NUMPY_MADVISE_HUGEPAGE=0
cd /home/grolo
RAW=/mnt/sda2/grolo/nnUNet/raw/Dataset904_KidneyTSMechanical
P=/mnt/sda2/grolo/kidney/predictions
OOF=$P/oof_logits_clean_finetune291
OOFRAW=$P/oof_train_logits_finetune291
O=/mnt/sda2/grolo/kidney/finetune
H=/mnt/sda2/grolo/humanitas_source
W=/mnt/sda2/grolo/kidney/external
MEDCAM=/home/aiart/anaconda3/envs/nnunet_medcam/bin/python
L=$O/logs
for d in $O $W/maps_finetune $W/deu_finetune $W/ebcu_finetune $W/eval_scratch_vs_finetune; do
  [ -z "$(ls -A $d 2>/dev/null)" ] || { echo "$d contiene gia' file: non sovrascrivo"; exit 1; }
done
mkdir -p $L $W/eval_scratch_vs_finetune

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
count() { ls "$1" | grep -cE "^$2_$3\.nii\.gz$"; }

n=$(ls $OOF/*_logits.pt | wc -l); [ "$n" -eq 234 ] || { echo "OOF: $n logit invece di 234"; exit 1; }
n=$(ls $OOFRAW/*.nii.gz | wc -l); [ "$n" -eq 234 ] || { echo "OOF: $n predizioni invece di 234"; exit 1; }
n=$(ls $W/pred_finetune/*_logits.pt | wc -l); [ "$n" -eq 110 ] || { echo "Humanitas: $n logit invece di 110"; exit 1; }

echo "=== [C] T* LOO sui 234 OOF del fine-tuned, 4 processi $(date) ==="
for i in 0 1 2 3; do
  python -u /home/grolo/thesis_pipeline/3_uncertainty/temperature/loo_temperature_scaling.py --logits_dir $OOF --gt_dir $RAW/labelsTr --num_shards 4 --shard_index $i \
    --out_summary_csv $O/loo_T_summary_shard$i.csv --out_grid_csv $O/loo_T_grid_shard$i.csv > $L/blockC_shard$i.log 2>&1 &
done
wait
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
echo "=== T* fine-tuned 904 = $T  ($(date)) ==="

echo "=== [D2] entropy OOF + baseline E[U|d] del fine-tuned $(date) ==="
retry_oom python -u /home/grolo/thesis_pipeline/3_uncertainty/maps/total_uncertainty_measures.py --input_dir $OOFRAW --output_dir $O/kidney_uncertainty_maps_TRAIN \
  --T $T --measures entropy > $L/blockD2_train_entropy.log 2>&1
n=$(count $O/kidney_uncertainty_maps_TRAIN "s[0-9]+" entropy); [ "$n" -eq 234 ] || { echo "D2: $n mappe OOF invece di 234"; exit 1; }
retry_oom python -u /home/grolo/thesis_pipeline/3_uncertainty/distance_aware/build_baseline_EU_given_d.py --entropy_dir $O/kidney_uncertainty_maps_TRAIN \
  --pred_dir $OOFRAW --out_dir $O/baseline_EU --uncertainty_type entropy --max_mm 60 > $L/blockD2_baseline.log 2>&1

echo "=== [Humanitas D/D2] mappe del fine-tuned $(date) ==="
retry_oom python -u /home/grolo/thesis_pipeline/3_uncertainty/maps/total_uncertainty_measures.py --input_dir $W/pred_finetune --output_dir $W/maps_finetune --T $T \
  > $L/humanitas_blockD.log 2>&1
retry_oom python -u /home/grolo/thesis_pipeline/3_uncertainty/distance_aware/compute_DEU.py --baseline_npz $O/baseline_EU/baseline_EU_given_d_mm_inside_outside.npz \
  --uncertainty_dir $W/maps_finetune --pred_dir $W/pred_finetune --out_dir $W/deu_finetune --uncertainty_type entropy \
  > $L/humanitas_DEU.log 2>&1
retry_oom python -u /home/grolo/thesis_pipeline/3_uncertainty/distance_aware/make_ebcu_maps.py --baseline_npz $O/baseline_EU/baseline_EU_given_d_mm_inside_outside.npz \
  --entropy_dir $W/maps_finetune --pred_dir $W/pred_finetune --out_dir $W/ebcu_finetune --uncertainty_type entropy \
  > $L/humanitas_EBCU.log 2>&1
for f in $W/deu_finetune/*_DEU_entropy.nii.gz $W/ebcu_finetune/*_ebcu_entropy.nii.gz; do ln -sf "$f" "$W/maps_finetune/$(basename $f)"; done
for u in entropy variance mutual_information DEU_entropy ebcu_entropy; do
  n=$(count $W/maps_finetune "hum[^_]+" $u); [ "$n" -eq 110 ] || { echo "maps_finetune: $n mappe $u invece di 110"; exit 1; }
done

echo "=== [controllo] rene predetto dal fine-tuned fuori dal ritaglio $(date) ==="
$MEDCAM - > $W/eval_scratch_vs_finetune/pass1_finetune_outside_crop.log 2>&1 <<EOF
import numpy as np, pandas as pd, SimpleITK as sitk
r = pd.read_csv("$W/crop_report.csv"); r = r[r.status == "ok"]
rows = []
for _, x in r.iterrows():
    img = sitk.ReadImage(f"$W/pass1_finetune/{x.case}.nii.gz"); p = sitk.GetArrayFromImage(img) > 0
    v = np.prod(img.GetSpacing()) / 1000
    out = p.copy(); out[int(x.z0):int(x.z1) + 1] = False
    rows.append(dict(case=x.case, pred_ml_full=p.sum() * v, pred_ml_outside_crop=out.sum() * v))
d = pd.DataFrame(rows); d.to_csv("$W/eval_scratch_vs_finetune/pass1_finetune_outside_crop.csv", index=False)
print(len(d), "casi | mL fuori dal ritaglio: max", round(d.pred_ml_outside_crop.max(), 2), "| casi > 0:", int((d.pred_ml_outside_crop > 0).sum()))
EOF
cat $W/eval_scratch_vs_finetune/pass1_finetune_outside_crop.log

echo "=== [confronto] from-scratch vs fine-tuned su Humanitas $(date) ==="
for st in segmentation calibration threshold_free; do
  retry_oom $MEDCAM /home/grolo/thesis_pipeline/4_analysis/humanitas_eval.py --config 904ft --stage $st > $L/humanitas_eval_$st.log 2>&1
done
retry_oom python /home/grolo/thesis_pipeline/4_analysis/humanitas_member_diversity.py --config 904ft > $L/humanitas_member_diversity.log 2>&1
echo "=== PIPELINE FINE-TUNING 904 COMPLETA: $(date) ==="
