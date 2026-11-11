#!/usr/bin/env python3
"""
Dice, diversita' dei membri (Dice tra coppie, varianza e MI medie) e calibrazione per caso dei modelli di rene e trachea.

Uso: python ensemble_dice_diversity_calibration.py [--only_trachea_checkpoint]
"""
import glob
import itertools
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import distance_transform_edt
from scipy.stats import wilcoxon

sys.path[:0] = ['/home/grolo/thesis_pipeline/3_uncertainty/temperature']  # cartelle dei moduli del progetto importati qui
import loo_temperature_scaling as lts  # noqa: E402

TS = "/mnt/sda2/grolo"
PR = "/mnt/sda2/grolo"
RAW = "/mnt/sda2/grolo/nnUNet/raw"
OUT = Path(f"{TS}/comparisons/ensemble_dice_diversity")

MODELS = {
    "kidney_deep": dict(gt=f"{RAW}/Dataset904_KidneyTSMechanical/labelsTs",
                        pred=f"{PR}/kidney/predictions/deep",
                        maps=f"{TS}/kidney/deep/kidney_uncertainty_maps", tstar=f"{TS}/kidney/deep/loo_T_summary.csv"),
    "kidney_checkpoint": dict(gt=f"{RAW}/Dataset904_KidneyTSMechanical/labelsTs",
                              pred=f"{PR}/kidney/predictions/checkpoint",
                              maps=f"{TS}/kidney/checkpoint/kidney_cyclr_uncertainty_maps",
                              tstar=f"{TS}/kidney/checkpoint/loo_T_summary.csv"),
    "trachea_checkpoint": dict(gt=f"{RAW}/Dataset903_TracheaTS/labelsTs",
                               pred=f"{PR}/trachea/predictions/checkpoint",
                               maps=f"{TS}/trachea/checkpoint/trachea_cyclr_uncertainty_maps",
                               tstar=f"{TS}/trachea/checkpoint/loo_T_summary.csv"),
    "trachea_deep": dict(gt=f"{RAW}/Dataset903_TracheaTS/labelsTs",
                         pred=f"{PR}/trachea/predictions/deep",
                         maps=f"{TS}/trachea/deep/trachea_uncertainty_maps",
                         tstar=f"{TS}/trachea/deep/loo_T_summary.csv"),
}


def dice(a, b):
    s = a.sum() + b.sum()
    return np.nan if s == 0 else 2.0 * np.logical_and(a, b).sum() / s


def roi15(gt, pred, spacing):
    union = gt | pred
    if not union.any():
        return None
    return distance_transform_edt(~union, sampling=spacing) <= 15.0


def load_members(pred_dir, case):
    files = sorted(glob.glob(f"{pred_dir}/fold_logits/{case}_fold_*_logits.pt"))
    out = []
    for f in files:
        lg = torch.load(f, map_location="cpu", weights_only=True)
        lg = lg.numpy() if hasattr(lg, "numpy") else np.asarray(lg)
        out.append(lg.astype(np.float32).argmax(0) == 1)
    return out


def main(only=None, out_name="per_case.csv"):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, cfg in MODELS.items():
        if only and name not in only:
            continue
        T = float(pd.read_csv(cfg["tstar"]).T_star.median())
        cases = sorted(p.name.replace(".nii.gz", "") for p in Path(cfg["gt"]).glob("*.nii.gz"))
        for i, c in enumerate(cases, 1):
            gt_img = nib.load(f"{cfg['gt']}/{c}.nii.gz")
            gt = gt_img.get_fdata() > 0.5
            sp = gt_img.header.get_zooms()[:3]
            pred = nib.load(f"{cfg['pred']}/{c}.nii.gz").get_fdata() > 0.5
            r = dict(model=name, case=c, T_star=T, gt_empty=bool(not gt.any()), pred_empty=bool(not pred.any()),
                     dice=dice(gt, pred) if (gt.any() or pred.any()) else np.nan)
            roi = roi15(gt, pred, sp)
            if roi is not None:
                members = load_members(cfg["pred"], c)
                r["n_members"] = len(members)
                r["pairwise_dice"] = float(np.nanmean([dice(a, b) for a, b in itertools.combinations(members, 2)]))
                for meas in ["variance", "mutual_information"]:
                    u = nib.load(f"{cfg['maps']}/{c}_{meas}.nii.gz").get_fdata()
                    if u.shape != roi.shape:
                        raise ValueError(f"{name} {c} {meas}: {u.shape} != {roi.shape}")
                    r[f"mean_{meas}_roi"] = float(u[roi].mean())
                if name.startswith("trachea"):
                    gtl, spl = lts.load_gt(Path(f"{cfg['gt']}/{c}.nii.gz"))
                    gtl = (gtl > 0).astype(np.int64)
                    lg = lts.align_logits_to_gt_transpose_then_resample(
                        lts.load_logits_pt(Path(f"{cfg['pred']}/{c}_logits.pt")), gtl.shape)
                    rm = lts.build_roi_mask(gtl, np.argmax(lg, 0), spl, 15.0)
                    if rm.any():
                        X = lts.logits_to_flat(lg)[rm.reshape(-1)]
                        y = gtl.reshape(-1)[rm.reshape(-1)]
                        for tag, t in [("T1", 1.0), ("Tstar", T)]:
                            p = lts.softmax_T_flat(X, t)
                            r[f"nll_{tag}"] = lts.nll_from_logits_flat(X, y, t)
                            r[f"ece_{tag}"] = lts.ece_multiclass_confidence(p, y)
                            r[f"brier_{tag}"] = lts.binary_brier_from_probs(p, y)
            rows.append(r)
            print(f"[{name} {i}/{len(cases)}] {c} dice={r['dice']}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / out_name, index=False)
    if only:  # run parziale (checkpoint della trachea), riepilogo a parte con trachea_summary()
        return

    L = []
    def ms(x):
        x = pd.Series(x).dropna()
        return f"{x.mean():.4g} ± {x.std():.3g} (mediana {x.median():.4g}, n={len(x)})"
    for name in MODELS:
        d = df[df.model == name]
        L.append(f"== {name}: T* = {d.T_star.iloc[0]:.4f}")
        for c in ["dice", "pairwise_dice", "mean_variance_roi", "mean_mutual_information_roi"]:
            L.append(f"  {c}: {ms(d[c])}")
        if name.startswith("trachea"):
            neg = d[d.gt_empty]
            L.append(f"  negativi: {len(neg)}, con maschera predetta: {(~neg.pred_empty).sum()}")
            L.append(f"  positivi con predizione vuota: {(d[~d.gt_empty].pred_empty).sum()}")
            for k in ["nll", "ece", "brier"]:
                a, b = d[f"{k}_T1"].dropna(), d[f"{k}_Tstar"].dropna()
                L.append(f"  {k}: T1 {ms(a)} -> T* {ms(b)} ({100 * (b.mean() / a.mean() - 1):.1f}%)")
    s = df[df.model == "kidney_deep"].set_index("case")
    k = df[df.model == "kidney_checkpoint"].set_index("case")
    s, k = s.align(k, join="inner", axis=0)
    L.append("== rene deep contro checkpoint (Wilcoxon appaiato)")
    for c in ["dice", "pairwise_dice", "mean_variance_roi", "mean_mutual_information_roi"]:
        x = pd.concat([s[c], k[c]], axis=1).dropna()
        x.columns = ["d", "k"]
        L.append(f"  {c}: deep {x.d.mean():.5g} | checkpoint {x.k.mean():.5g} | p = {wilcoxon(x.d, x.k).pvalue:.3g}"
                 f" | deep > checkpoint in {(x.d > x.k).sum()}/{len(x)}")
    (OUT / "summary.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def trachea_summary():
    """Deep contro checkpoint della trachea (Wilcoxon appaiato), unendo i due CSV per caso."""
    d = pd.read_csv(OUT / "per_case.csv")
    k = pd.read_csv(OUT / "per_case_trachea_checkpoint.csv")
    d = d[d.model == "trachea_deep"].set_index("case")
    k = k.set_index("case")
    L = [f"== trachea checkpoint: T* = {k.T_star.iloc[0]:.4f}",
         f"  negativi con maschera predetta: {(~k[k.gt_empty].pred_empty).sum()}/{k.gt_empty.sum()}, positivi vuoti: {k[~k.gt_empty].pred_empty.sum()}"]
    for c in ["dice", "pairwise_dice", "mean_variance_roi", "mean_mutual_information_roi",
              "nll_T1", "nll_Tstar", "ece_T1", "ece_Tstar", "brier_T1", "brier_Tstar"]:
        x = pd.concat([d[c], k[c]], axis=1).dropna()
        x.columns = ["d", "k"]
        L.append(f"  {c}: deep {x.d.mean():.5g} ± {x.d.std():.3g} | checkpoint {x.k.mean():.5g} ± {x.k.std():.3g}"
                 f" (mediane {x.d.median():.4g} / {x.k.median():.4g}) | p = {wilcoxon(x.d, x.k).pvalue:.3g}"
                 f" | deep > checkpoint in {(x.d > x.k).sum()}/{len(x)}")
    (OUT / "summary_trachea_checkpoint.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only_trachea_checkpoint", action="store_true")
    if ap.parse_args().only_trachea_checkpoint:
        main(only={"trachea_checkpoint"}, out_name="per_case_trachea_checkpoint.csv")
        trachea_summary()
    else:
        main()
