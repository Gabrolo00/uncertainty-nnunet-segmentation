#!/usr/bin/env python3
"""
Diversita' dei membri sul test esterno (Dice tra coppie, varianza e MI medie nella regione di interesse).

Uso: python humanitas_member_diversity.py --config 904ft
"""
import itertools
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import distance_transform_edt
from scipy.stats import wilcoxon

B = Path("/mnt/sda2/grolo/humanitas_source")
# --config "904ft": deep ensemble from scratch contro fine-tuned sullo stesso ritaglio del test esterno.
CONFIGS = {
    "904ft": dict(work=Path("/mnt/sda2/grolo/kidney/external"), out=Path("/mnt/sda2/grolo/kidney/external") / "eval_scratch_vs_finetune",
                  models={"scratch": ("pred_deep", "maps_deep"), "finetune": ("pred_finetune", "maps_finetune")}),
}
W = CONFIGS["904ft"]["work"]
OUT = CONFIGS["904ft"]["out"]
MODELS = CONFIGS["904ft"]["models"]


def dice(a, b):
    s = a.sum() + b.sum()
    return np.nan if s == 0 else 2.0 * np.logical_and(a, b).sum() / s


def roi15(gt, pred, spacing):
    union = gt | pred
    dist_out = distance_transform_edt(~union, sampling=spacing)
    return dist_out <= 15.0


def main():
    cohort = pd.read_csv(B / "cohort.csv")
    cases = cohort[(cohort.gt_source == "Rene (clinico)") & cohort.excluded.isna()]["case"].tolist()
    rows = []
    for i, case in enumerate(cases, 1):
        gt_img = nib.load(W / "cropped" / "labelsTs" / f"{case}.nii.gz")
        gt = gt_img.get_fdata() > 0.5
        spacing = gt_img.header.get_zooms()[:3]
        for model, (pdir, mdir) in MODELS.items():
            pred = nib.load(W / pdir / f"{case}.nii.gz").get_fdata() > 0.5
            members = []
            for k in range(5):
                lg = torch.load(W / pdir / "fold_logits" / f"{case}_fold_{k}_logits.pt", map_location="cpu", weights_only=True)
                lg = lg.numpy() if hasattr(lg, "numpy") else np.asarray(lg)
                members.append(lg.astype(np.float32).argmax(0) == 1)
                del lg
            pw = [dice(a, b) for a, b in itertools.combinations(members, 2)]
            roi = roi15(gt, pred, spacing)
            row = dict(case=case, model=model, pairwise_dice=float(np.nanmean(pw)),
                       member_fg_vox=float(np.mean([m.sum() for m in members])))
            for meas in ["variance", "mutual_information"]:
                u = nib.load(W / mdir / f"{case}_{meas}.nii.gz").get_fdata()
                if u.shape != roi.shape:
                    raise ValueError(f"{case} {model} {meas}: shape {u.shape} != {roi.shape}")
                row[f"mean_{meas}_roi"] = float(u[roi].mean())
            rows.append(row)
        print(f"[{i}/{len(cases)}] {case} ok", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "member_diversity_per_case.csv", index=False)

    s = df[df.model == "scratch"].set_index("case")
    f = df[df.model == "finetune"].set_index("case")
    s, f = s.align(f, join="inner", axis=0)
    lines = [f"Casi: {len(s)} (contorni clinici)"]
    for c in ["pairwise_dice", "mean_variance_roi", "mean_mutual_information_roi"]:
        p = wilcoxon(s[c], f[c]).pvalue
        lines.append(f"{c}: scratch {s[c].mean():.5g} ± {s[c].std():.3g} (mediana {s[c].median():.5g}) | "
                     f"finetune {f[c].mean():.5g} ± {f[c].std():.3g} (mediana {f[c].median():.5g}) | "
                     f"Wilcoxon p = {p:.3g} | scratch > finetune in {(s[c] > f[c]).sum()}/{len(s)}")
    (OUT / "member_diversity_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="904ft", choices=list(CONFIGS))
    cfg = CONFIGS[ap.parse_args().config]
    W, OUT, MODELS = cfg["work"], cfg["out"], cfg["models"]
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[member_diversity] work {W} -> {OUT}", flush=True)
    main()
