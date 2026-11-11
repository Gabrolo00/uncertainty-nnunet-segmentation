"""
Salva per ogni caso l'entropia in float32 (entropy32.npy), usata per calcolare DEU ed EBCU di riferimento
senza l'arrotondamento a 16 bit.

Uso: AE_DATASET=904 python autoencoder_entropy_float32.py
"""
import json, os, sys
import numpy as np, torch
from multiprocessing import Pool
from scipy.special import softmax
sys.path[:0] = ['/home/grolo/thesis_pipeline/5_autoencoder']  # cartelle dei moduli del progetto importati qui
from autoencoder_build_dataset import src_paths, ent, T_STAR
from autoencoder_config import DATASET_DIR
D = DATASET_DIR

def run(c):
    m = json.load(open(f"{D}/{c}/meta.json")); z0, z1, y0, y1, x0, x1 = m["crop"]
    ml = torch.load(src_paths(m["source"], c)["logit"], weights_only=False)[:, z0:z1, y0:y1, x0:x1].float().numpy()
    e = ent(softmax(ml / T_STAR, axis=0)).astype(np.float32)
    np.save(f"{D}/{c}/entropy32.npy", e)
    return float(np.abs(e - np.load(f"{D}/{c}/entropy.npy").astype(np.float32)).max())

if __name__ == "__main__":
    cases = sorted(c for c in os.listdir(D) if os.path.isdir(f"{D}/{c}"))
    with Pool(6) as p: r = p.map(run, cases)
    print(len(r), "casi; max |entropy32 - entropy fp16| =", round(max(r), 5))
