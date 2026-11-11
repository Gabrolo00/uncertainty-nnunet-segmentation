#!/bin/bash
# Inferenza del deep ensemble del rene sul test esterno Humanitas: passaggio veloce sulla CT intera,
# ritaglio attorno ai reni, ensemble completo (5 fold, TTA, logit) sul ritaglio.
# Uso: STAGES="1 2 3" GPU=0 ./inference_humanitas_kidney904.sh   (RESUME=1 riprende uno stage interrotto)
set -e
export nnUNet_raw=/mnt/sda2/grolo/nnUNet/raw
export nnUNet_preprocessed=/mnt/sda2/grolo/nnUNet/preprocessed
export nnUNet_results=/mnt/sda2/grolo/nnUNet/results
export PYTHONPATH=/mnt/sda2/grolo/py_shims
export CUDA_VISIBLE_DEVICES=${GPU:-0}
# Senza questa variabile preprocessing ed export di nnU-Net sono molto lenti sulle CT whole body
# (compattazione della memoria per le huge pages di numpy)
export NUMPY_MADVISE_HUGEPAGE=0
B=/mnt/sda2/grolo/humanitas_source
W=/mnt/sda2/grolo/kidney/external
PY=/home/aiart/anaconda3/envs/nnunet_env/bin/python
MEDCAM=/home/aiart/anaconda3/envs/nnunet_medcam/bin/python
mkdir -p $W
cd /mnt/sda2/grolo   # log relativi in /mnt/sda2/grolo/logs

empty_or_exit() { [ "$RESUME" = 1 ] && return 0; [ -z "$(ls -A "$1" 2>/dev/null)" ] || { echo "$1 contiene gia' file: non sovrascrivo"; exit 1; }; }
CONT=""; [ "$RESUME" = 1 ] && CONT="--continue_prediction"

for s in ${STAGES:-1 2 3}; do
  case $s in
    1) empty_or_exit $W/pass1
       echo "=== [1] passaggio veloce 904 $(date)"
       $PY /home/grolo/thesis_pipeline/2_models/inference/run_inference.py -i $B/images -o $W/pass1 -d 904 -c 3d_fullres -f 0 --disable_tta -npp 1 -nps 1 $CONT \
         >> $W/logs_infer_pass1.log 2>&1 ;;
    2) echo "=== [2] ritaglio 904 $(date)"
       $MEDCAM /home/grolo/thesis_pipeline/1_data/crop_humanitas_kidney.py --work_dir $W > $W/logs_crop.log 2>&1 ;;
    3) empty_or_exit $W/pred_deep
       echo "=== [3] deep ensemble 904 $(date)"
       $PY /home/grolo/thesis_pipeline/2_models/inference/run_inference.py -i $W/cropped/images -o $W/pred_deep -d 904 -c 3d_fullres -f 0 1 2 3 4 \
         --save_logits -npp 1 -nps 1 $CONT >> $W/logs_infer_deep.log 2>&1 ;;
  esac
done
echo "=== FINITO $(date)"
