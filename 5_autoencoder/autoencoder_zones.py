"""
Analisi per zona: errori divisi per lato del bordo predetto e fascia di distanza; per ogni zona AUPRC normalizzata,
errori coperti con un budget dell'1% della regione di interesse e posizione dei falsi allarmi.

Uso: AE_DATASET=904 python autoencoder_zones.py [workers]
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import average_precision_score

sys.path[:0] = ['/home/grolo/thesis_pipeline/5_autoencoder']  # cartelle dei moduli del progetto importati qui
import autoencoder_eval as E  # noqa: E402
from autoencoder_combine import SET, combos  # noqa: E402

OUT = f"{E.OUT}/regions_good" if SET == "good" else f"{E.OUT}/regions"
AE_RUNS = ["a", "a_good", "a_rand"] if SET == "good" else ["a", "c"]
BANDS = {"0-2": (0, 2), "2-4.5": (2, 4.5), "4.5-10": (4.5, 10), ">10": (10, 1e9)}
TOP = 0.01
CHOSEN = json.load(open(f"{E.OUT}/{'combine_good' if SET == 'good' else 'combine'}/chosen_params.json"))


def process(job):
    c, fold, source = job
    D = E.D
    pr = np.load(f"{D}/{c}/pred.npy").astype(bool)
    gt = np.load(f"{D}/{c}/gt.npy").astype(bool)
    d = np.load(f"{D}/{c}/dist.npy").astype(np.float32)
    ad = np.abs(d)
    roi = (distance_transform_edt(~(pr | gt), sampling=1.5) <= 15) & ~E.sinus_fn(c, pr, gt)
    err, ok = (pr ^ gt) & roi, ~(pr ^ gt) & roi
    refs = E.reference_maps(c, pr)
    u = refs["entropy"]
    maps = dict(refs)
    for v in AE_RUNS:
        md = f"{E.R}/{E.VARIANTS[v]}/fold{fold}/maps/{c}"
        rec = np.load(f"{md}/recon.npy").astype(np.float32)
        rl = np.load(f"{md}/resid_logit.npy").astype(np.float32) if os.path.exists(f"{md}/resid_logit.npy") else None
        maps[f"{v}:excess"] = np.maximum(u - rec, 0)
        maps[f"{v}:excess+missing"] = np.abs(u - rec)
        maps.update({f"{v}:{k}": m for k, m in combos(u, rec, rl, CHOSEN[v]).items()})
    rows, fa = [], []
    sides = {"dentro (FP)": d > 0, "fuori (FN)": d <= 0}
    for k, mp in maps.items():
        # Budget ESATTO: K = 1% dei voxel della ROI. I voxel sopra la soglia contano 1; quelli con valore
        # uguale alla soglia (pareggi: EBCU satura a 1, mappe con molti zeri) contano la frazione che serve per
        # arrivare a K, cioe' il valore atteso di uno spareggio casuale (con ">=" l'EBCU supererebbe molto il budget).
        s_roi = np.nan_to_num(mp[roi], nan=0.0)
        K = TOP * s_roi.size
        thr = np.sort(s_roi)[::-1][max(int(np.ceil(K)) - 1, 0)]
        above, at = s_roi > thr, s_roi == thr
        wv = above.astype(np.float64) + at * (max(K - above.sum(), 0) / max(at.sum(), 1))
        top = np.zeros(d.shape, np.float64)   # peso di allarme per voxel (somma = K)
        top[roi] = wv
        frac_top = float(wv.sum() / s_roi.size)
        n_top_ok = float((top * ok).sum())
        for bn, (lo, hi) in BANDS.items():
            bm = (ad > lo) & (ad <= hi) if lo > 0 else ad <= hi
            fa.append(dict(case=c, source=source, map=k, band=bn,
                           frac_false_alarms=float((top * (ok & bm)).sum()) / max(n_top_ok, 1e-9), n_top_ok=n_top_ok,
                           frac_top=frac_top))
            for sn, sm in sides.items():
                pos = err & sm & bm
                npos = int(pos.sum())
                r = dict(case=c, source=source, map=k, side=sn, band=bn, n_pos=npos)
                if npos >= 10:
                    m = pos | ok
                    y = pos[m]
                    prev = float(y.mean())
                    ap_ = average_precision_score(y, np.nan_to_num(mp[m], nan=0.0))
                    r.update(auprc_norm=(ap_ - prev) / (1 - prev), recall_top1=float((top * pos).sum() / npos), frac_top=frac_top)
                rows.append(r)
    return rows, fa


def main():
    os.makedirs(OUT, exist_ok=True)
    f = pd.read_csv(f"{E.D}/folds.csv")
    f = f[f.gt_source != "Kidney_L/R (RayStation auto)"]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as p:
        res = p.map(process, list(zip(f.case, f.fold, f.source)))
    z = pd.DataFrame([r for x in res for r in x[0]])
    fa = pd.DataFrame([r for x in res for r in x[1]])
    z.to_csv(f"{OUT}/zones_per_case.csv", index=False)
    fa.to_csv(f"{OUT}/false_alarms_per_case.csv", index=False)
    lines = []
    for src in ["totalseg", "humanitas"]:
        zz = z[(z.source == src) & z.auprc_norm.notna()]
        n = zz.drop_duplicates(["case", "side", "band"]).groupby(["side", "band"]).case.size()
        lines += [f"\n######## {src} — casi con >= 10 voxel di errore per zona:\n{n.unstack().to_string()}"]
        for met in ["auprc_norm", "recall_top1"]:
            t = zz.pivot_table(index="map", columns=["side", "band"], values=met, aggfunc="median")
            lines += [f"\n{met} (mediana per caso)\n{t.round(3).to_string()}"]
        vol = z[z.source == src].groupby(["side", "band"]).n_pos.sum() * 1.5 ** 3 / 1000
        lines += [f"\nvolume totale di errori per zona (mL):\n{vol.unstack().round(1).to_string()}"]
        t = fa[fa.source == src].pivot_table(index="map", columns="band", values="frac_false_alarms", aggfunc="mean")
        lines += [f"\nfalsi allarmi nell'1% piu' alto: frazione per fascia (media sui casi)\n"
                  f"{t[list(BANDS)].round(3).to_string()}"]
    txt = "\n".join(lines)
    open(f"{OUT}/summary.txt", "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
