"""
Valuta le mappe residue dell'autoencoder e le misure di riferimento (entropia, MI, DEU, EBCU) nelle viste
standard, no-border e target; ogni caso e' valutato dall'autoencoder che non l'ha visto in training.

Uso: AE_DATASET=904 python autoencoder_eval.py [workers] [casi separati da virgola]
"""
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path[:0] = ['/home/grolo/thesis_pipeline/3_uncertainty/distance_aware', '/home/grolo/thesis_pipeline/5_autoencoder']  # cartelle dei moduli del progetto importati qui
from compute_DEU import compute_deu_map, signed_distance_to_boundary_mm  # noqa: E402
from make_ebcu_maps import ebcu_from_baseline_percentiles  # noqa: E402
from autoencoder_train_infer import sinus_fn  # noqa: E402

from autoencoder_config import C, DATASET_DIR, RUNS_DIR, EVAL_DIR   # AE_DATASET sceglie il dataset (default 904)
D = DATASET_DIR
R = RUNS_DIR
OUT = EVAL_DIR
# dict(): lettura immediata. NpzFile legge in modo pigro e i processi del Pool condividerebbero lo stesso file aperto
# (BadZipFile).
BASE = dict(np.load(C["BASELINE"]))   # tabella E[U|d, lato] del modello valutato
_T = C["AE_TAG"]   # configurazione scelta con la selezione annidata, per dataset (autoencoder_config.py)
VARIANTS = {"a": f"a_{_T}", "b": f"b_{_T}", "c": f"c_{_T}",
            "a_half": f"a_{_T}_frac0.5",   # curva di apprendimento (metà dei casi di training)
            "a_good": f"a_{_T}_good",      # training solo sui casi ben segmentati
            "a_rand": f"a_{_T}_rand"}      # controllo, casi casuali di pari numero
WIDTHS = [2.0, 4.5]
SP = (1.5, 1.5, 1.5)


def reference_maps(c, pr):
    u = np.load(f"{D}/{c}/entropy32.npy")          # float32: l'EBCU amplifica l'arrotondamento fp16
    deu = compute_deu_map(u, pr.astype(np.uint8), SP, BASE["mm"], BASE["inside_mean"], BASE["outside_mean"])
    d_in, d_out = signed_distance_to_boundary_mm(pr.astype(np.uint8), SP)
    ebcu = ebcu_from_baseline_percentiles(u, d_in, d_out, BASE["inside_p90"], BASE["inside_p99"],
                                          BASE["outside_p90"], BASE["outside_p99"])
    # La tabella E[U|d] non ha valori oltre 21 mm dentro il rene (nessun voxel cosi' profondo nel training TS):
    # li' l'EBCU originale diventa NaN. Stessa regola della DEU (che li' vale 0 per costruzione): niente allarme.
    ebcu = np.nan_to_num(ebcu, nan=0.0)
    return {"entropy": u, "mi": np.load(f"{D}/{c}/mi.npy").astype(np.float32), "DEU": deu, "EBCU": ebcu}


def process(job):
    c, fold, source = job
    pr = np.load(f"{D}/{c}/pred.npy").astype(bool)
    gt = np.load(f"{D}/{c}/gt.npy").astype(bool)
    ad = np.abs(np.load(f"{D}/{c}/dist.npy").astype(np.float32))
    roi = (distance_transform_edt(~(pr | gt), sampling=1.5) <= 15) & ~sinus_fn(c, pr, gt)
    err = (pr ^ gt) & roi
    ok = ~(pr ^ gt) & roi
    maps = reference_maps(c, pr)
    for v, tag in VARIANTS.items():
        md = f"{R}/{tag}/fold{fold}/maps/{c}"
        # solo gruppi COMPLETATI (quickeval_summary.csv e' scritto dopo l'inferenza): evita file in scrittura
        if os.path.exists(f"{R}/{tag}/fold{fold}/quickeval_summary.csv") and os.path.exists(f"{md}/excess.npy"):
            ex = np.load(f"{md}/excess.npy").astype(np.float32)
            ms = np.load(f"{md}/missing.npy").astype(np.float32)
            maps.update({f"{v}:excess": ex, f"{v}:missing": ms, f"{v}:excess+missing": ex + ms})
            for ch in ["logit", "ct"]:   # residui sui canali extra (infer_resid), letti solo dentro la ROI
                if os.path.exists(f"{md}/resid_{ch}.npy"):
                    maps[f"{v}:resid_{ch}"] = np.load(f"{md}/resid_{ch}.npy").astype(np.float32)
    views = {"roi15": (err, ok)}
    for w in WIDTHS:
        far = ad > w
        views[f"far{w:g}"] = (err & far, ok & far)
        views[f"farVSall{w:g}"] = (err & far, ok)
    rows = []
    for vn, (pos, neg) in views.items():
        m = pos | neg
        y = pos[m]
        if y.sum() < 10 or (~y).sum() < 10:
            continue
        prev = float(y.mean())   # AUPRC di una mappa casuale = frazione di positivi nella vista (per caso)
        for k, mp in maps.items():
            s = mp[m]
            if not np.isfinite(s).all():
                print(f"[ATTENZIONE] {c} {vn} {k}: {int((~np.isfinite(s)).sum())} valori non finiti -> 0", flush=True)
                s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
            ap_ = average_precision_score(y, s)
            rows.append(dict(case=c, source=source, fold=fold, view=vn, map=k, n_pos=int(y.sum()), prev=prev,
                             auroc=roc_auc_score(y, s), auprc=ap_, auprc_over_random=ap_ / prev,
                             auprc_norm=(ap_ - prev) / (1 - prev)))
    return rows


def summarize(d):
    lines = []
    for excl in [False, True]:
        x = d[~d.case.isin(["s0543", "s0545"])] if excl else d
        lines.append(f"\n######## {'senza s0543/s0545' if excl else 'tutti i casi'}")
        for met in ["auprc", "auprc_norm", "auprc_over_random", "auroc"]:
            t = x.pivot_table(index="map", columns=["source", "view"], values=met, aggfunc="median").round(3)
            lines.append(f"\n{met.upper()} (mediana per caso)\n{t.to_string()}")
    n = d.drop_duplicates(["case", "view"]).groupby(["source", "view"]).case.nunique()
    lines.append(f"\ncasi valutati per vista:\n{n.to_string()}")
    return "\n".join(lines)


def main():
    os.makedirs(OUT, exist_ok=True)
    f = pd.read_csv(f"{D}/folds.csv")
    f = f[f.gt_source != "Kidney_L/R (RayStation auto)"]       # i 3 contorni automatici: fuori dalla valutazione
    if len(sys.argv) > 2:
        f = f[f.case.isin(sys.argv[2].split(","))]
    jobs = list(zip(f.case, f.fold, f.source))
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 6) as p:
        res = p.map(process, jobs)
    d = pd.DataFrame([r for x in res for r in x])
    d.to_csv(f"{OUT}/eval_per_case.csv", index=False)
    txt = summarize(d)
    open(f"{OUT}/summary.txt", "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
