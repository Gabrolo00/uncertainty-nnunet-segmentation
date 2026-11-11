#!/bin/bash
# Training del checkpoint ensemble del rene, sua inferenza (validazione e test) e training del deep ensemble della
# trachea, distribuiti sulle due GPU.
# Uso: ./train_kidney904_checkpoint_and_trachea_deep.sh
export nnUNet_raw=/mnt/sda2/grolo/nnUNet/raw
export nnUNet_preprocessed=/mnt/sda2/grolo/nnUNet/preprocessed
export nnUNet_results=/mnt/sda2/grolo/nnUNet/results
export PYTHONPATH=/mnt/sda2/grolo/py_shims
cd /mnt/sda2/grolo   # log relativi in /mnt/sda2/grolo/logs
PY=/home/aiart/anaconda3/envs/nnunet_env/bin/python
P=/mnt/sda2/grolo/kidney/predictions
M=$nnUNet_results/Dataset904_KidneyTSMechanical/nnUNetTrainer__nnUNetPlans__3d_fullres__cyclr_scratch

step() {  # nome log comando...
  local name=$1 log=$2; shift 2
  echo "=== $name start: $(date)"
  "$@" > $log 2>&1; local rc=$?
  echo "=== $name finito (exit $rc): $(date)"
  return $rc
}

trachea_fold() {  # fold gpu
  CUDA_VISIBLE_DEVICES=$2 step "trachea fold $1" logs/train_903_scratch_fold$1.log \
    $PY /home/grolo/thesis_pipeline/2_models/training/train_deep_ensemble.py --dataset 903 --configuration 3d_fullres --fold $1 --num_epochs 1000
}

gpu1() {
  CUDA_VISIBLE_DEVICES=1 step "cyclr 904 training" logs/cyclr_904_train.log \
    $PY /home/grolo/thesis_pipeline/2_models/training/train_checkpoint_ensemble.py --dataset 904 --fold 0 --output_folder_suffix __cyclr_scratch &&
  CUDA_VISIBLE_DEVICES=1 step "cyclr 904 inferenza val fold 0" logs/cyclr_904_infer_val.log \
    $PY /home/grolo/thesis_pipeline/2_models/inference/run_checkpoint_ensemble_inference.py --model_folder $M -i oof_input_904_fold0 -o $P/val_fold0_cyclr_scratch_T1 &&
  CUDA_VISIBLE_DEVICES=1 step "cyclr 904 inferenza test" logs/cyclr_904_infer_test.log \
    $PY /home/grolo/thesis_pipeline/2_models/inference/run_checkpoint_ensemble_inference.py --model_folder $M -i $nnUNet_raw/Dataset904_KidneyTSMechanical/imagesTs \
    -o $P/checkpoint
  for f in 1 3; do trachea_fold $f 1; done
}

gpu0() {
  until grep -q "fold 4 finito (exit 0)" logs/train_904_fold4.log 2>/dev/null; do
    grep -q "fold 4 finito" logs/train_904_fold4.log 2>/dev/null && { echo "training fold 4 fallito: GPU 0 ferma"; return 1; }
    sleep 120; done
  CUDA_VISIBLE_DEVICES=0 step "deep 904 inferenza test" logs/deep_904_infer_test.log \
    $PY /home/grolo/thesis_pipeline/2_models/inference/run_inference.py -i $nnUNet_raw/Dataset904_KidneyTSMechanical/imagesTs -o $P/deep \
    -d 904 -c 3d_fullres -f 0 1 2 3 4 -chk checkpoint_final.pth -T 1.0 --save_logits -npp 2 -nps 2 --verbose
  for f in 0 2 4; do trachea_fold $f 0; done
}

gpu1 > logs/queue_904_gpu1.log 2>&1 &
gpu0 > logs/queue_904_gpu0.log 2>&1 &
wait
echo "=== CODA GPU COMPLETA: $(date)"
