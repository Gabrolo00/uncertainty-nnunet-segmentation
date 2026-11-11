"""
Controlli sulle mappe: voxel in cui l'EBCU non e' definita (in profondita' nel rene) ed effetto del salvataggio
delle mappe a 8 bit sulle metriche.

Uso: AE_DATASET=904 python check_8bit_and_nan.py
"""
import json
import os
import sys
from multiprocessing import Pool

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path[:0] = ['/home/grolo/thesis_pipeline/3_uncertainty/distance_aware', '/home/grolo/thesis_pipeline/5_autoencoder']  # cartelle dei moduli del progetto importati qui
import autoencoder_eval as E  # noqa: E402
from compute_DEU import signed_distance_to_boundary_mm  # noqa: E402
from make_ebcu_maps import ebcu_from_baseline_percentiles  # noqa: E402

D = E.D
OUT = f"{E.OUT}/checks_8bit_nan"
OFFICIAL = "/mnt/sda2/grolo/kidney/deep/kidney_uncertainty_maps"
PAIRS = {"entropy": "entropy", "mi": "mutual_information", "DEU": "DEU_entropy", "EBCU": "ebcu_entropy"}


def process(job):
    c, fold, source = job
    m = json.load(open(f"{D}/{c}/meta.json"))
    z0, z1, y0, y1, x0, x1 = m["crop"]
    pr = np.load(f"{D}/{c}/pred.npy").astype(bool)
    gt = np.load(f"{D}/{c}/gt.npy").astype(bool)
    ad = np.abs(np.load(f"{D}/{c}/dist.npy").astype(np.float32))
    roi = (distance_transform_edt(~(pr | gt), sampling=1.5) <= 15) & ~E.sinus_fn(c, pr, gt)
    err, ok = (pr ^ gt) & roi, ~(pr ^ gt) & roi
    u = np.load(f"{D}/{c}/entropy32.npy")
    d_in, d_out = signed_distance_to_boundary_mm(pr.astype(np.uint8), E.SP)
    raw = ebcu_from_baseline_percentiles(u, d_in, d_out, E.BASE["inside_p90"], E.BASE["inside_p99"],
                                         E.BASE["outside_p90"], E.BASE["outside_p99"])
    nan = ~np.isfinite(raw)
    nan_row = dict(case=c, source=source, nan_roi=int((nan & roi).sum()), roi=int(roi.sum()), err_far2=int((err & (ad > 2)).sum()),
                   nan_err=int((nan & err).sum()), nan_err_far2=int((nan & err & (ad > 2)).sum()),
                   nan_u_p99=float(np.percentile(u[nan & roi], 99)) if (nan & roi).any() else np.nan,
                   min_depth=float(ad[nan & roi].min()) if (nan & roi).any() else np.nan)
    rows, align = [], []
    if source == "totalseg":
        refs = E.reference_maps(c, pr)
        views = {"roi15": (err, ok)}
        for w in E.WIDTHS:
            views[f"far{w:g}"] = (err & (ad > w), ok & (ad > w))
            views[f"farVSall{w:g}"] = (err & (ad > w), ok)
        for k, off in PAIRS.items():
            o = np.asarray(nib.load(f"{OFFICIAL}/{c}_{off}.nii.gz").get_fdata(), np.float32).transpose(2, 1, 0)[z0:z1, y0:y1, x0:x1]
            assert o.shape == pr.shape, (c, off, o.shape, pr.shape)
            o = np.nan_to_num(o)
            f = np.nan_to_num(refs[k])
            align.append(dict(case=c, map=k, corr=float(np.corrcoef(o[roi], f[roi])[0, 1]), levels_off=int(np.unique(o[roi]).size),
                              levels_float=int(np.unique(f[roi]).size), frac_ties_off=1 - np.unique(o[roi]).size / roi.sum()))
            for vn, (pos, neg) in views.items():
                mk = pos | neg
                y = pos[mk]
                if y.sum() < 10 or (~y).sum() < 10:
                    continue
                prev = float(y.mean())
                for nm, mp in [("8bit", o), ("float", f)]:
                    ap = average_precision_score(y, mp[mk])
                    rows.append(dict(case=c, map=k, view=vn, kind=nm, auprc_norm=(ap - prev) / (1 - prev), auroc=roc_auc_score(y, mp[mk])))
    return nan_row, rows, align


def main():
    os.makedirs(OUT, exist_ok=True)
    f = pd.read_csv(f"{D}/folds.csv")
    f = f[f.gt_source != "Kidney_L/R (RayStation auto)"]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 6) as p:
        res = p.map(process, list(zip(f.case, f.fold, f.source)))
    nan = pd.DataFrame([r[0] for r in res])
    d = pd.DataFrame([x for r in res for x in r[1]])
    al = pd.DataFrame([x for r in res for x in r[2]])
    for n, t in [("nan_ebcu", nan), ("metrics", d), ("alignment", al)]:
        t.to_csv(f"{OUT}/{n}_per_case.csv", index=False)
    L = ["### NaN EBCU per sorgente"]
    g = nan.groupby("source")
    L.append(pd.DataFrame(dict(casi=g.size(), casi_con_nan=g.nan_roi.apply(lambda x: (x > 0).sum()), frac_roi=g.nan_roi.sum() / g.roi.sum(),
                               voxel_err_nan=g.nan_err.sum(), voxel_err_far2=g.err_far2.sum(), err_far2_nan=g.nan_err_far2.sum(),
                               frac_err_far2=g.nan_err_far2.sum() / g.err_far2.sum(), prof_min_mm=g.min_depth.min())).T.to_string())
    L += ["\n### Allineamento ufficiale/float (mediana)", al.groupby("map")[["corr", "levels_off", "levels_float"]].median().round(4).to_string()]
    for met in ["auprc_norm", "auroc"]:
        L.append(f"\n### {met}, mediana per caso, 8 bit contro virgola mobile")
        rows = []
        for (k, v), x in d.groupby(["map", "view"]):
            w = x.pivot(index="case", columns="kind", values=met).dropna()
            dif = w["8bit"] - w["float"]
            try:
                p_ = wilcoxon(w["8bit"], w["float"]).pvalue
            except ValueError:
                p_ = 1.0
            rows.append(dict(map=k, view=v, n=len(w), bit8=w["8bit"].median(), float_=w["float"].median(), med_diff=dif.median(),
                             p90_abs_diff=dif.abs().quantile(.9), bit8_better=int((dif > 0).sum()), float_better=int((dif < 0).sum()), p=p_))
        t = pd.DataFrame(rows)
        L.append(t.round(4).to_string(index=False))
        rk = t.pivot(index="map", columns="view", values="bit8").rank(ascending=False)
        rf = t.pivot(index="map", columns="view", values="float_").rank(ascending=False)
        L += ["classifica 8 bit:", rk.to_string(), "classifica virgola mobile:", rf.to_string()]
    txt = "\n".join(L)
    open(f"{OUT}/summary.txt", "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
