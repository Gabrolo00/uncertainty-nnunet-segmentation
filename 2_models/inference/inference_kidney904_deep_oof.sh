#!/bin/bash
# Inferenza out-of-fold del deep ensemble del rene (ogni fold sui propri casi di validazione) e raccolta dei logit
# per il temperature scaling.
# Uso: ./inference_kidney904_deep_oof.sh
export nnUNet_raw=/mnt/sda2/grolo/nnUNet/raw
export nnUNet_preprocessed=/mnt/sda2/grolo/nnUNet/preprocessed
export nnUNet_results=/mnt/sda2/grolo/nnUNet/results
export PYTHONPATH=/mnt/sda2/grolo/py_shims
export CUDA_VISIBLE_DEVICES=1
cd /mnt/sda2/grolo   # log relativi in /mnt/sda2/grolo/logs
PY=/home/aiart/anaconda3/envs/nnunet_env/bin/python
P=/mnt/sda2/grolo/kidney/predictions
OUT=$P/oof_train_logits
mkdir -p $OUT
run_fold() {
  echo "=== fold $1 start: $(date)"
  $PY /home/grolo/thesis_pipeline/2_models/inference/run_inference.py -i oof_input_904_fold$1 -o $OUT -d 904 -c 3d_fullres -f $1 -chk checkpoint_final.pth \
    -T 1.0 --save_logits -npp 2 -nps 2 > logs/oof_904_fold$1.log 2>&1 || { echo "fold $1 FALLITO"; exit 1; }
  echo "=== fold $1 done: $(date)"
}
for f in 0 1 2 3; do run_fold $f; done
until grep -q "fold 4 finito (exit 0)" logs/train_904_fold4.log 2>/dev/null; do
  grep -q "fold 4 finito" logs/train_904_fold4.log 2>/dev/null && { echo "training fold 4 fallito"; exit 1; }; sleep 120; done
run_fold 4
echo "=== collect: $(date)"
$PY /home/grolo/thesis_pipeline/3_uncertainty/temperature/collect_oof_logits.py \
  --splits_json $nnUNet_preprocessed/Dataset904_KidneyTSMechanical/splits_final.json \
  --logits_dir $OUT/fold_logits --output_dir $P/oof_logits_clean --copy > logs/oof_904_collect.log 2>&1
echo "=== OOF COMPLETO (exit $?): $(date)"
