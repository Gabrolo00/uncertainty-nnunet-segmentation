"""
Inferenza del deep ensemble (stessi argomenti di nnUNetv2_predict, piu' il salvataggio dei logit).

Uso: python run_inference.py -i <imagesTs> -o <out> -d <dataset> -c 3d_fullres -f 0 1 2 3 4 --save_logits
"""
import sys

sys.path[:0] = ['/home/grolo/thesis_pipeline/2_models/inference']  # cartelle dei moduli del progetto importati qui
from predict_from_raw_data import predict_entry_point

if __name__ == '__main__':
    predict_entry_point()
