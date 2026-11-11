"""
Ricampiona CT e reference Humanitas a 1.5 mm con la stessa funzione e gli stessi parametri di nnU-Net.

Uso: AE_DATASET=904 python autoencoder_resample_humanitas.py
"""
import json, os, sys
import numpy as np, nibabel as nib
from multiprocessing import Pool
from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape

from autoencoder_config import C, DATASET_DIR
D = DATASET_DIR


def run(c):
    m = json.load(open(f"{D}/{c}/meta.json"))
    z0, z1, y0, y1, x0, x1 = m["crop"]
    sp_zyx = list(reversed(m["native_spacing"]))
    grid = m["grid_zyx"]
    ct = np.asarray(nib.load(f"{C['HUM_CT']}/{c}_0000.nii.gz").dataobj, dtype=np.float32).transpose(2, 1, 0)
    gt = (np.asarray(nib.load(f"{C['HUM_GT']}/{c}.nii.gz").dataobj) > 0.5).transpose(2, 1, 0).astype(np.float32)
    ct_r = resample_data_or_seg_to_shape(ct[None], grid, sp_zyx, [1.5, 1.5, 1.5], is_seg=False, order=3, order_z=0,
                                         force_separate_z=None)[0]
    gt_r = resample_data_or_seg_to_shape(gt[None], grid, sp_zyx, [1.5, 1.5, 1.5], is_seg=True, order=1, order_z=0,
                                         force_separate_z=None)[0]
    old_gt = np.load(f"{D}/{c}/gt.npy").astype(bool)
    new_gt = gt_r[z0:z1, y0:y1, x0:x1] > 0.5
    np.save(f"{D}/{c}/ct.npy", np.clip(ct_r[z0:z1, y0:y1, x0:x1], -200, 300).astype(np.float16))
    np.save(f"{D}/{c}/gt.npy", new_gt.astype(np.uint8))
    m["ct_gt_resampling"] = "nnunet resample_data_or_seg_to_shape (ct order3/order_z0, gt is_seg order1/order_z0)"
    json.dump(m, open(f"{D}/{c}/meta.json", "w"), indent=1)
    dice = 2 * (old_gt & new_gt).sum() / max(old_gt.sum() + new_gt.sum(), 1)
    return c, dice


if __name__ == "__main__":
    cases = sorted(c for c in os.listdir(D) if c.startswith("hum"))
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 6) as p:
        res = p.map(run, cases)
    d = np.array([r[1] for r in res])
    print(f"{len(res)} casi Humanitas riscritti; Dice GT vecchia (zoom) contro nuova (nnU-Net): min {d.min():.4f}, mediana {np.median(d):.4f}")
