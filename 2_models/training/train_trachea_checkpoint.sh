#!/bin/bash
# Training del checkpoint ensemble della trachea e sua inferenza (validazione del fold 0 e test).
# Uso: GPU=1 ./train_trachea_checkpoint.sh   (RESUME=1 riprende il training)
export nnUNet_raw=/mnt/sda2/grolo/nnUNet/raw
export nnUNet_preprocessed=/mnt/sda2/grolo/nnUNet/preprocessed
export nnUNet_results=/mnt/sda2/grolo/nnUNet/results
export PYTHONPATH=/mnt/sda2/grolo/py_shims
export NUMPY_MADVISE_HUGEPAGE=0  # evita il rallentamento delle huge pages di numpy sulle CT grandi
export CUDA_VISIBLE_DEVICES=${GPU:-1}
cd /mnt/sda2/grolo   # log relativi in /mnt/sda2/grolo/logs
PY=/home/aiart/anaconda3/envs/nnunet_env/bin/python
M=$nnUNet_results/Dataset903_TracheaTS/nnUNetTrainer__nnUNetPlans__3d_fullres__cyclr_scratch
P=/mnt/sda2/grolo/trachea/predictions
empty() { [ -z "$(ls -A "$1" 2>/dev/null)" ] || { echo "$1 contiene gia' file: non sovrascrivo"; exit 1; }; }

if [ "$RESUME" = 1 ]; then EXTRA=--resume; else empty $M; EXTRA=; fi
echo "=== training checkpoint trachea start: $(date)"
$PY /home/grolo/thesis_pipeline/2_models/training/train_checkpoint_ensemble.py --dataset 903 --fold 0 --output_folder_suffix __cyclr_scratch $EXTRA \
  >> logs/trachea_cyclr_train.log 2>&1
rc=$?; echo "=== training checkpoint trachea finito (exit $rc): $(date)"; [ $rc -eq 0 ] || exit 1
n=$(ls $M/fold_0/checkpoint_cycle*_ep*.pth | wc -l); echo "snapshot salvati: $n"

empty $P/val_fold0_cyclr_scratch_T1
echo "=== inferenza validation fold 0 (48 casi) start: $(date)"
$PY /home/grolo/thesis_pipeline/2_models/inference/run_checkpoint_ensemble_inference.py --model_folder $M -i oof_input_fold0 -o $P/val_fold0_cyclr_scratch_T1 \
  > logs/trachea_cyclr_infer_val.log 2>&1
echo "=== inferenza validation finita (exit $?): $(date)"
empty $P/checkpoint
echo "=== inferenza test (60 casi) start: $(date)"
$PY /home/grolo/thesis_pipeline/2_models/inference/run_checkpoint_ensemble_inference.py --model_folder $M -i $nnUNet_raw/Dataset903_TracheaTS/imagesTs \
  -o $P/checkpoint > logs/trachea_cyclr_infer_test.log 2>&1
echo "=== inferenza test finita (exit $?): $(date)"
echo "=== CODA GPU CHECKPOINT TRACHEA COMPLETA: $(date)"
