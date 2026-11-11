#!/usr/bin/env python3
"""
Metriche per caso del deep ensemble del rene sul test interno ed esterno: Dice, volumi, falsi positivi e negativi,
stima del seno renale, percentili dell'incertezza per spessore di fetta, livello medio e AUROC/AUPRC delle mappe.

Uso: python domain_shift_stats.py
"""
import os
from multiprocessing import Pool

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, label
from skimage.morphology import convex_hull_image
from sklearn.metrics import average_precision_score, roc_auc_score

K = "/mnt/sda2/grolo/humanitas_source"
TS = "/mnt/sda2/grolo"
RAW = "/mnt/sda2/grolo/nnUNet/raw/Dataset904_KidneyTSMechanical"
PRED = "/mnt/sda2/grolo/kidney/predictions/deep"
OUT = f"{TS}/comparisons/domain_shift"
MAPS = ["entropy", "variance", "mutual_information", "anti_confidence_total", "DEU_entropy", "ebcu_entropy"]


def paths(source, c):
    if source == "external":
        return (f"/mnt/sda2/grolo/kidney/external/cropped/images/{c}_0000.nii.gz", f"/mnt/sda2/grolo/kidney/external/cropped/labelsTs/{c}.nii.gz",
                f"/mnt/sda2/grolo/kidney/external/pred_deep/{c}.nii.gz", f"/mnt/sda2/grolo/kidney/external/maps_deep")
    return (f"{RAW}/imagesTs/{c}_0000.nii.gz", f"{RAW}/labelsTs/{c}.nii.gz", f"{PRED}/{c}.nii.gz",
            f"{TS}/kidney/deep/kidney_uncertainty_maps")


def sdt(m, sp):
    return distance_transform_edt(m, sampling=sp) - distance_transform_edt(~m, sampling=sp)


def load(p):
    return np.asarray(nib.load(p).dataobj, dtype=np.float32)


def process(job):
    source, c = job
    pct, pgt, ppr, mdir = paths(source, c)
    img = nib.load(pgt)
    sp = np.array(img.header.get_zooms()[:3], float)
    gt, pr, ct = load(pgt) > 0.5, load(ppr) > 0.5, load(pct)
    v = np.prod(sp) / 1000.0
    roi = sdt(gt | pr, sp) >= -15.0
    ad = np.abs(sdt(pr, sp)) if pr.any() else np.full(pr.shape, np.inf)
    fp, fn = pr & ~gt & roi, ~pr & gt & roi
    err = fp | fn
    r = dict(source=source, case=c, slice_mm=float(sp[2]), gt_ml=gt.sum() * v, pred_ml=pr.sum() * v,
             fp_ml=fp.sum() * v, fn_ml=fn.sum() * v,
             dice=2 * (pr & gt).sum() / max(pr.sum() + gt.sum(), 1))
    hull = np.zeros_like(pr)
    lab, n = label(pr)
    for k in range(1, n + 1):
        comp = lab == k
        if comp.sum() * v < 5:
            continue
        idx = np.argwhere(comp)
        bb = tuple(slice(a, b + 1) for a, b in zip(idx.min(0), idx.max(0)))
        hull[bb] |= convex_hull_image(comp[bb])
    fnf = fn & (ad > 2.0)
    r["fn_far2_ml"] = fnf.sum() * v
    r["fn_far2_in_hull_fat_ml"] = (fnf & hull & (ct < -20)).sum() * v
    ok = roi & (ad > 4.5) & ~err
    nb = roi & (ad > 2.0)
    for m in MAPS:
        u = load(f"{mdir}/{c}_{m}.nii.gz")
        if u.shape != gt.shape:
            raise ValueError(f"{source} {c} {m}: {u.shape} != {gt.shape}")
        r[f"mean_{m}_roi"] = float(u[roi].mean())
        if m in ("entropy", "mutual_information"):
            x = u[ok]
            for q in (50, 90, 99):
                r[f"{m}_ok_p{q}"] = float(np.percentile(x, q)) if x.size else np.nan
        if source == "internal":
            for name, reg in (("roi15mm", roi), ("roi15mm_no_border2mm", nb)):
                e, s = err[reg], u[reg]
                ok_ = 0 < e.sum() < e.size
                r[f"auroc_{m}_{name}"] = roc_auc_score(e, s) if ok_ else np.nan
                r[f"auprc_{m}_{name}"] = average_precision_score(e, s) if ok_ else np.nan
    print(source, c, "ok", flush=True)
    return r


def main():
    os.makedirs(OUT, exist_ok=True)
    co = pd.read_csv(f"{K}/cohort.csv")
    ext = co[(co.gt_source == "Rene (clinico)") & co.excluded.isna()]["case"].tolist()
    internal = sorted(f[:-7] for f in os.listdir(f"{RAW}/labelsTs") if f.endswith(".nii.gz"))
    jobs = [("internal", c) for c in internal] + [("external", c) for c in ext]
    with Pool(4) as p:
        rows = p.map(process, jobs, chunksize=1)
    pd.DataFrame(rows).to_csv(f"{OUT}/per_case.csv", index=False)
    print("FATTO", len(rows))


if __name__ == "__main__":
    main()
