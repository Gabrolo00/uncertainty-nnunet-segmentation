#!/bin/bash
# Inferenza del deep ensemble fine-tuned del rene: out-of-fold sui casi di training e test esterno Humanitas sullo
# stesso ritaglio del modello from scratch.
# Uso: ./inference_kidney904_finetune.sh
export nnUNet_raw=/mnt/sda2/grolo/nnUNet/raw
export nnUNet_preprocessed=/mnt/sda2/grolo/nnUNet/preprocessed
export nnUNet_results=/mnt/sda2/grolo/nnUNet/results
export PYTHONPATH=/mnt/sda2/grolo/py_shims
export NUMPY_MADVISE_HUGEPAGE=0  # evita il rallentamento delle huge pages di numpy sulle CT grandi
cd /mnt/sda2/grolo   # log relativi in /mnt/sda2/grolo/logs
PY=/home/aiart/anaconda3/envs/nnunet_env/bin/python
CFG=3d_fullres__finetune_pretrained291
M=$nnUNet_results/Dataset904_KidneyTSMechanical/nnUNetTrainer__nnUNetPlans__$CFG
P=/mnt/sda2/grolo/kidney/predictions
OOF=$P/oof_train_logits_finetune291
H=/mnt/sda2/grolo/humanitas_source
W=/mnt/sda2/grolo/kidney/external

wait_fold() {  # aspetta la fine del training del fold $1 del fine-tuning
  until grep -q "fold $1 finito (exit 0)" logs/train_904_ft_fold$1.log 2>/dev/null; do
    grep -qE "fold $1 finito \(exit [1-9]" logs/train_904_ft_fold$1.log 2>/dev/null && { echo "training fold $1 fallito"; return 1; }
    sleep 300; done
  [ -f $M/fold_$1/checkpoint_final.pth ] || { echo "manca $M/fold_$1/checkpoint_final.pth"; return 1; }
}
empty() { [ -z "$(ls -A "$1" 2>/dev/null)" ] || { echo "$1 contiene gia' file: non sovrascrivo"; return 1; }; }

oof_fold() {  # fold gpu
  [ -z "$(ls $OOF/fold_logits 2>/dev/null | grep "_fold_$1_")" ] || { echo "OOF fold $1 gia' presente"; return 1; }
  echo "=== OOF fold $1 start: $(date)"
  CUDA_VISIBLE_DEVICES=$2 $PY /home/grolo/thesis_pipeline/2_models/inference/run_inference.py -i oof_input_904_fold$1 -o $OOF -d 904 -c $CFG -f $1 \
    -chk checkpoint_final.pth -T 1.0 --save_logits -npp 2 -nps 2 > logs/ft904_oof_fold$1.log 2>&1
  local rc=$?; echo "=== OOF fold $1 finito (exit $rc): $(date)"; return $rc
}

gpu0() {
  # Tutto sulla GPU 0
  wait_fold 4 || return 1
  for f in 0 1 2 3; do wait_fold $f || return 1; done
  for f in 0 1 2 3 4; do oof_fold $f 0; done
  empty $W/pred_finetune || return 1
  echo "=== Humanitas ensemble fine-tuned start: $(date)"
  CUDA_VISIBLE_DEVICES=0 $PY /home/grolo/thesis_pipeline/2_models/inference/run_inference.py -i $W/cropped/images -o $W/pred_finetune -d 904 -c $CFG -f 0 1 2 3 4 \
    --save_logits -npp 1 -nps 1 > $W/logs_infer_finetune.log 2>&1
  echo "=== Humanitas ensemble fine-tuned finito (exit $?): $(date)"
  if empty $W/pass1_finetune; then
    echo "=== passaggio veloce Humanitas (controllo FP fuori ritaglio) start: $(date)"
    CUDA_VISIBLE_DEVICES=0 $PY /home/grolo/thesis_pipeline/2_models/inference/run_inference.py -i $H/images -o $W/pass1_finetune -d 904 -c $CFG -f 0 --disable_tta \
      -npp 1 -nps 1 > $W/logs_infer_pass1_finetune.log 2>&1
    echo "=== passaggio veloce finito (exit $?): $(date)"
  fi
}

gpu0 > logs/ft904_after_gpu0.log 2>&1

n=$(ls $OOF/fold_logits/*_logits.pt 2>/dev/null | wc -l)
if [ "$n" -eq 234 ] && [ -z "$(ls -A $P/oof_logits_clean_finetune291 2>/dev/null)" ]; then
  $PY /home/grolo/thesis_pipeline/3_uncertainty/temperature/collect_oof_logits.py \
    --splits_json $nnUNet_preprocessed/Dataset904_KidneyTSMechanical/splits_final.json \
    --logits_dir $OOF/fold_logits --output_dir $P/oof_logits_clean_finetune291 --copy > logs/ft904_oof_collect.log 2>&1
  echo "=== collect OOF (exit $?): $(date)"
else
  echo "=== collect NON eseguito: $n fold_logits OOF (attesi 234) o cartella di destinazione non vuota"
fi
echo "=== CODA GPU FINE-TUNING 904 COMPLETA: $(date)"
