#!/bin/bash
# Inferenza del deep ensemble della trachea: out-of-fold sui casi di training e test (5 fold, TTA, logit).
# Uso: ./inference_trachea_deep.sh
export nnUNet_raw=/mnt/sda2/grolo/nnUNet/raw
export nnUNet_preprocessed=/mnt/sda2/grolo/nnUNet/preprocessed
export nnUNet_results=/mnt/sda2/grolo/nnUNet/results
export PYTHONPATH=/mnt/sda2/grolo/py_shims
# Huge pages di numpy disattivate: con le CT grandi preprocessing ed export di nnU-Net sono molto lenti.
export NUMPY_MADVISE_HUGEPAGE=0
cd /mnt/sda2/grolo   # log relativi in /mnt/sda2/grolo/logs
PY=/home/aiart/anaconda3/envs/nnunet_env/bin/python
P=/mnt/sda2/grolo/trachea/predictions
OOF=$P/oof_train_logits_scratch
TEST=$P/deep
M=$nnUNet_results/Dataset903_TracheaTS/nnUNetTrainer__nnUNetPlans__3d_fullres

wait_fold() {  # aspetta la fine del training del fold $1 della trachea
  until grep -q "trachea fold $1 finito (exit 0)" logs/queue_904_gpu0.log logs/queue_904_gpu1.log 2>/dev/null; do
    grep -qE "trachea fold $1 finito \(exit [1-9]" logs/queue_904_gpu0.log logs/queue_904_gpu1.log && { echo "training trachea fold $1 fallito"; return 1; }
    sleep 300; done
  [ -f $M/fold_$1/checkpoint_final.pth ] || { echo "manca $M/fold_$1/checkpoint_final.pth"; return 1; }
}

oof_fold() {  # fold gpu: i casi di validazione del fold (da splits_final.json)
  [ -z "$(ls $OOF/fold_logits 2>/dev/null | grep "_fold_$1_")" ] || { echo "OOF fold $1 gia' presente: non sovrascrivo"; return 1; }
  echo "=== OOF trachea fold $1 start: $(date)"
  CUDA_VISIBLE_DEVICES=$2 $PY /home/grolo/thesis_pipeline/2_models/inference/run_inference.py -i oof_input_fold$1 -o $OOF -d 903 -c 3d_fullres -f $1 \
    -chk checkpoint_final.pth -T 1.0 --save_logits -npp 2 -nps 2 > logs/trachea_scratch_oof_fold$1.log 2>&1
  local rc=$?; echo "=== OOF trachea fold $1 finito (exit $rc): $(date)"; return $rc
}

gpu0() {
  for f in 0 1 2 4; do wait_fold $f && oof_fold $f 0; done
  echo "=== GPU0 OOF TRACHEA COMPLETI: $(date)"
}

gpu1() {
  wait_fold 3 || return 1
  oof_fold 3 1
  for f in 0 1 2 4; do wait_fold $f || return 1; done
  [ -z "$(ls -A $TEST 2>/dev/null)" ] || { echo "$TEST contiene gia' file: non sovrascrivo"; return 1; }
  echo "=== test trachea start: $(date)"
  CUDA_VISIBLE_DEVICES=1 $PY /home/grolo/thesis_pipeline/2_models/inference/run_inference.py -i $nnUNet_raw/Dataset903_TracheaTS/imagesTs -o $TEST -d 903 \
    -c 3d_fullres -f 0 1 2 3 4 -chk checkpoint_final.pth -T 1.0 --save_logits -npp 2 -nps 2 \
    > logs/trachea_scratch_test.log 2>&1
  echo "=== test trachea finito (exit $?): $(date)"
}

gpu0 > logs/queue_sunday_gpu0.log 2>&1 &
gpu1 > logs/queue_sunday_gpu1.log 2>&1 &
wait

n=$(ls $OOF/fold_logits/*_logits.pt 2>/dev/null | wc -l)
if [ "$n" -eq 236 ] && [ -z "$(ls -A $P/oof_logits_clean_scratch 2>/dev/null)" ]; then
  $PY /home/grolo/thesis_pipeline/3_uncertainty/temperature/collect_oof_logits.py \
    --splits_json $nnUNet_preprocessed/Dataset903_TracheaTS/splits_final.json \
    --logits_dir $OOF/fold_logits --output_dir $P/oof_logits_clean_scratch --copy > logs/trachea_scratch_oof_collect.log 2>&1
  echo "=== collect OOF trachea (exit $?): $(date)"
else
  echo "=== collect NON eseguito: $n fold_logits OOF (attesi 236) o oof_logits_clean_scratch non vuota"
fi
echo "=== CODA DOMENICA COMPLETA: $(date)"
