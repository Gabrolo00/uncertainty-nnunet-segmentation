"""
Costruisce il dataset dell'autoencoder sulla griglia della rete (1.5 mm): per ogni caso di test interno ed
esterno entropia, logit, CT, predizione, reference e distanza dal bordo, piu' i gruppi di cross-fitting.

Uso: AE_DATASET=904 python autoencoder_build_dataset.py [--skip_humanitas] [--workers N]
"""
import argparse
import json
import os
from multiprocessing import Pool

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import distance_transform_edt, label, zoom
from scipy.special import softmax

from autoencoder_config import C, DS, DATASET_DIR   # AE_DATASET sceglie il dataset (default 904)
T_STAR = C["T_STAR"]
LN2 = np.log(2.0)
TS_PRED = C["TS_PRED"]
TS_RAW = C["TS_RAW"]
K = "/mnt/sda2/grolo/humanitas_source"


def src_paths(source, c):
    if source == "totalseg":
        return dict(logit=f"{TS_PRED}/{c}_logits.pt", folds=[f"{TS_PRED}/fold_logits/{c}_fold_{f}_logits.pt" for f in range(5)],
                    ct=f"{TS_RAW}/imagesTs/{c}_0000.nii.gz", gt=f"{TS_RAW}/labelsTs/{c}.nii.gz")
    return dict(logit=f"{C['HUM_PRED']}/{c}_logits.pt",
                folds=[f"{C['HUM_PRED']}/fold_logits/{c}_fold_{f}_logits.pt" for f in range(5)],
                ct=f"{C['HUM_CT']}/{c}_0000.nii.gz", gt=f"{C['HUM_GT']}/{c}.nii.gz")


def ent(p):
    return -(p * np.log(np.clip(p, 1e-12, 1))).sum(0) / LN2


def to_grid(nat_xyz, grid_zyx, order):
    """Volume nativo (X, Y, Z) -> griglia della rete (Z, Y, X) con la shape dei logit."""
    v = np.transpose(nat_xyz, (2, 1, 0)).astype(np.float32)
    f = [g / s for g, s in zip(grid_zyx, v.shape)]
    out = zoom(v, f, order=order)
    assert out.shape == tuple(grid_zyx), (out.shape, grid_zyx)
    return out


def process(job):
    source, c, out_dir = job
    pth = src_paths(source, c)
    od = os.path.join(out_dir, c)
    if os.path.exists(os.path.join(od, "meta.json")):
        return c, "gia' fatto"
    ml = torch.load(pth["logit"], weights_only=False).float().numpy()          # (2, Z, Y, X)
    grid = ml.shape[1:]
    p_mean = softmax(ml / T_STAR, axis=0)
    entropy = ent(p_mean)
    pred = (ml[1] > ml[0])
    logit = np.clip((ml[1] - ml[0]) / T_STAR, -20, 20)
    del ml, p_mean
    sum_p = np.zeros((2, *grid), np.float32)
    sum_h = np.zeros(grid, np.float32)
    for fp in pth["folds"]:
        fl = torch.load(fp, weights_only=False).float().numpy()
        assert fl.shape[1:] == grid, (fp, fl.shape, grid)
        pf = softmax(fl / T_STAR, axis=0)
        sum_p += pf
        sum_h += ent(pf)
        del fl, pf
    mi = np.maximum(ent(sum_p / 5) - sum_h / 5, 0)
    del sum_p, sum_h

    ct_img = nib.load(pth["ct"])
    ct = to_grid(np.asarray(ct_img.dataobj), grid, order=1)
    gt = to_grid(np.asarray(nib.load(pth["gt"]).dataobj) > 0.5, grid, order=0) > 0.5

    # ritaglio senza GT
    lab, n = label(pred)
    sizes = np.bincount(lab.ravel())
    keep = np.isin(lab, [k for k in range(1, n + 1) if sizes[k] * 1.5 ** 3 / 1000 >= 5])
    if keep.any():
        zz = np.where(keep.any((1, 2)))[0]
        z0, z1 = max(zz.min() - 27, 0), min(zz.max() + 27 + 1, grid[0])   # 27 voxel = 40 mm
    else:
        z0, z1 = 0, grid[0]
    body = (ct[z0:z1] > -500).any(0)
    yy, xx = np.where(body.any(1))[0], np.where(body.any(0))[0]
    y0, y1, x0, x1 = yy.min(), yy.max() + 1, xx.min(), xx.max() + 1
    sl = (slice(z0, z1), slice(y0, y1), slice(x0, x1))

    pc = pred[sl]
    dist = (distance_transform_edt(pc, sampling=1.5) - distance_transform_edt(~pc, sampling=1.5)) if pc.any() \
        else np.full(pc.shape, -30.0)
    os.makedirs(od, exist_ok=True)
    for name, arr, dt in [("entropy", entropy[sl], np.float16), ("mi", mi[sl], np.float16),
                          ("logit", logit[sl], np.float16), ("dist", np.clip(dist, -30, 30), np.float16),
                          ("ct", np.clip(ct[sl], -200, 300), np.float16),
                          ("pred", pc, np.uint8), ("gt", gt[sl], np.uint8)]:
        np.save(os.path.join(od, f"{name}.npy"), arr.astype(dt))
    meta = dict(source=source, case=c, grid_zyx=list(map(int, grid)), crop=[int(z0), int(z1), int(y0), int(y1), int(x0), int(x1)],
                native_shape_xyz=list(map(int, ct_img.shape)), native_spacing=[float(s) for s in ct_img.header.get_zooms()[:3]],
                native_affine=ct_img.affine.tolist(), T=T_STAR, crop_shape=list(map(int, pc.shape)),
                pred_ml=float(pred.sum() * 1.5 ** 3 / 1000), gt_ml_in_crop=float(gt[sl].sum() * 1.5 ** 3 / 1000),
                gt_ml_total=float(gt.sum() * 1.5 ** 3 / 1000))
    json.dump(meta, open(os.path.join(od, "meta.json"), "w"), indent=1)
    return c, f"ok {pc.shape}, GT fuori dal ritaglio {meta['gt_ml_total'] - meta['gt_ml_in_crop']:.1f} mL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=DATASET_DIR)
    ap.add_argument("--skip_humanitas", action="store_true", help="solo la parte TotalSegmentator")
    ap.add_argument("--cases", default="", help="solo questi casi (test)")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    ts = sorted(f[:-7] for f in os.listdir(f"{TS_RAW}/labelsTs") if f.endswith(".nii.gz"))
    co = pd.read_csv(f"{K}/cohort.csv")
    co = co[co.excluded != True]
    hum = [] if a.skip_humanitas else [c for c in co["case"] if os.path.exists(f"{C['HUM_PRED']}/{c}_logits.pt")]
    print(f"[config] dataset {DS}: T*={T_STAR}, TS={TS_PRED}, Humanitas={len(hum)} casi, out={a.out_dir}", flush=True)
    jobs = [("totalseg", c, a.out_dir) for c in ts] + [("humanitas", c, a.out_dir) for c in hum]
    if a.cases:
        jobs = [j for j in jobs if j[1] in a.cases.split(",")]
    with Pool(a.workers) as p:
        for c, msg in p.imap_unordered(process, jobs):
            print(c, msg, flush=True)
    if not a.cases:
        # 5 gruppi per il cross-fitting, stratificati per sorgente (un caso = un paziente)
        rng = np.random.default_rng(42)
        rows = []
        gsrc = dict(zip(co["case"], co["gt_source"]))
        for src, cases in [("totalseg", ts), ("humanitas", hum)]:
            perm = rng.permutation(cases)
            rows += [dict(case=c, source=src, fold=i % 5, gt_source=gsrc.get(c, "TotalSegmentator"))
                     for i, c in enumerate(perm)]
        pd.DataFrame(rows).to_csv(os.path.join(a.out_dir, "folds.csv"), index=False)
        print(pd.DataFrame(rows).groupby(["fold", "source"]).size().unstack())


if __name__ == "__main__":
    main()
