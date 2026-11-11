#!/bin/bash
# Training dei 5 fold del deep ensemble del rene a partire dai pesi pretrained di TotalSegmentator.
# Uso: ./train_kidney904_finetune.sh
export nnUNet_raw=/mnt/sda2/grolo/nnUNet/raw
export nnUNet_preprocessed=/mnt/sda2/grolo/nnUNet/preprocessed
export nnUNet_results=/mnt/sda2/grolo/nnUNet/results
export PYTHONPATH=/mnt/sda2/grolo/py_shims
export NUMPY_MADVISE_HUGEPAGE=0
cd /mnt/sda2/grolo   # log relativi in /mnt/sda2/grolo/logs
PY=/home/aiart/anaconda3/envs/nnunet_env/bin/python
W291=/home/grolo/.totalsegmentator/nnunet/results/Dataset291_TotalSegmentator_part1_organs_1559subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth
OUT=$nnUNet_results/Dataset904_KidneyTSMechanical/nnUNetTrainer__nnUNetPlans__3d_fullres__finetune_pretrained291

echo "=== avvio fine-tuning: $(date)"

train_fold() {  # fold gpu
  [ -f $OUT/fold_$1/checkpoint_final.pth ] && { echo "fold $1 gia' completo: salto"; return 0; }
  [ -f $OUT/fold_$1/checkpoint_latest.pth ] && { echo "fold $1 ha gia' un checkpoint: non riparto da zero (usare resume)"; return 1; }
  CUDA_VISIBLE_DEVICES=$2 $PY /home/grolo/thesis_pipeline/2_models/training/train_deep_ensemble_finetune.py --dataset 904 --configuration 3d_fullres --fold $1 \
    --num_epochs 1000 --pretrained_weights $W291 --output_folder_suffix __finetune_pretrained291 \
    >> logs/train_904_ft_fold$1.log 2>&1
  echo "=== fold $1 finito (exit $?): $(date)" >> logs/train_904_ft_fold$1.log
}
(for f in 0 2 4; do train_fold $f 0; done) &
sleep 60
(for f in 1 3; do train_fold $f 1; done) &
wait
echo "=== FINE-TUNING 904: TUTTI I FOLD FINITI: $(date)"
