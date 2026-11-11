"""
Combinazioni della mappa di incertezza con i residui dell'autoencoder (C1-C4): scelta dei parametri sul gruppo
di selezione e valutazione.

Uso: AE_DATASET=904 python autoencoder_combine.py select|eval [workers]
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path[:0] = ['/home/grolo/thesis_pipeline/5_autoencoder']  # cartelle dei moduli del progetto importati qui
import autoencoder_eval as E  # noqa: E402
from autoencoder_stats import boot_median, holm  # noqa: E402

SET = os.environ.get("AE_COMBINE_SET", "abc")
VARS = ["a", "a_good", "a_rand"] if SET == "good" else ["a", "b", "c"]
OUT = f"{E.OUT}/combine_good" if SET == "good" else f"{E.OUT}/combine"
GRID = {"C1": [0.005, 0.01, 0.02, 0.05, 0.1], "C2": [1, 3, 10, 30, 100], "C3": [1, 3, 10, 30, 100],
        "C4": [1, 3, 10, 30, 100]}
SEL_VIEWS = ["farVSall2", "farVSall4.5"]
SELECTION_FOLD = 1


def combos(u, rec, resid_logit, params):
    out = {}
    for f, vals in params.items():
        for p in vals:
            if f == "C1":
                out[f"{f}({p:g})"] = u * u / (rec + p)
            elif f == "C2":
                out[f"{f}({p:g})"] = u * (1 + p * np.maximum(u - rec, 0))
            elif f == "C3":
                out[f"{f}({p:g})"] = u * (1 + p * np.abs(u - rec))
            elif f == "C4" and resid_logit is not None:
                out[f"{f}({p:g})"] = u * (1 + p * resid_logit)
    return out


def process(job):
    c, fold, source, params, views_keep, with_refs = job
    D = E.D
    pr = np.load(f"{D}/{c}/pred.npy").astype(bool)
    gt = np.load(f"{D}/{c}/gt.npy").astype(bool)
    ad = np.abs(np.load(f"{D}/{c}/dist.npy").astype(np.float32))
    roi = (distance_transform_edt(~(pr | gt), sampling=1.5) <= 15) & ~E.sinus_fn(c, pr, gt)
    err, ok = (pr ^ gt) & roi, ~(pr ^ gt) & roi
    views = {"roi15": (err, ok)}
    for w in E.WIDTHS:
        views[f"far{w:g}"] = (err & (ad > w), ok & (ad > w))
        views[f"farVSall{w:g}"] = (err & (ad > w), ok)
    views = {k: v for k, v in views.items() if k in views_keep}
    refs = E.reference_maps(c, pr)
    u = refs["entropy"]
    maps = {k: refs[k] for k in ["entropy", "DEU", "EBCU"]} if with_refs else {}
    for v in VARS:
        md = f"{E.R}/{E.VARIANTS[v]}/fold{fold}/maps/{c}"
        rec = np.load(f"{md}/recon.npy").astype(np.float32)
        rl = np.load(f"{md}/resid_logit.npy").astype(np.float32) if os.path.exists(f"{md}/resid_logit.npy") else None
        p = params[v] if isinstance(params, dict) and v in params else params
        maps.update({f"{v}:{k}": m for k, m in combos(u, rec, rl, p).items()})
        if with_refs:
            maps[f"{v}:excess"] = np.maximum(u - rec, 0)
    rows = []
    for vn, (pos, neg) in views.items():
        m = pos | neg
        y = pos[m]
        if y.sum() < 10 or (~y).sum() < 10:
            continue
        prev = float(y.mean())
        for k, mp in maps.items():
            s = np.nan_to_num(mp[m], nan=0.0, posinf=0.0, neginf=0.0)
            ap_ = average_precision_score(y, s)
            rows.append(dict(case=c, source=source, fold=fold, view=vn, map=k, n_pos=int(y.sum()), prev=prev,
                             auroc=roc_auc_score(y, s), auprc=ap_, auprc_norm=(ap_ - prev) / (1 - prev)))
    return rows


def cases():
    f = pd.read_csv(f"{E.D}/folds.csv")
    return f[f.gt_source != "Kidney_L/R (RayStation auto)"]


def run_pool(jobs, workers):
    with Pool(workers) as p:
        res = p.map(process, jobs)
    return pd.DataFrame([r for x in res for r in x])


def select(workers):
    f = cases()
    f = f[f.fold == SELECTION_FOLD]
    d = run_pool([(c, fo, s, GRID, SEL_VIEWS, False) for c, fo, s in zip(f.case, f.fold, f.source)], workers)
    d.to_csv(f"{OUT}/select_per_case.csv", index=False)
    med = d.groupby(["map", "source", "view"]).auprc_norm.median().unstack(["source", "view"])
    med["score"] = med.mean(axis=1)
    med.to_csv(f"{OUT}/select_summary.csv")
    chosen = {}
    for v in VARS:
        chosen[v] = {}
        for fm in GRID:
            s = med[med.index.str.startswith(f"{v}:{fm}(")].score
            if len(s):
                best = s.idxmax()
                chosen[v][fm] = [float(best.split("(")[1].rstrip(")"))]
    json.dump(chosen, open(f"{OUT}/chosen_params.json", "w"), indent=1)
    print(med.round(4).to_string())
    print("scelti:", chosen)


def evaluate(workers):
    chosen = json.load(open(f"{OUT}/chosen_params.json"))
    f = cases()
    views = ["roi15", "far2", "farVSall2", "far4.5", "farVSall4.5"]
    d = run_pool([(c, fo, s, chosen, views, True) for c, fo, s in zip(f.case, f.fold, f.source)], workers)
    d.to_csv(f"{OUT}/eval_per_case.csv", index=False)
    summ, tests = [], []
    for tag, x in {"all": d, "noSelectionFold": d[d.fold != SELECTION_FOLD]}.items():
        for (src, view), g in x.groupby(["source", "view"]):
            w = g.pivot_table(index="case", columns="map", values="auprc_norm")
            for m in w.columns:
                xx = w[m].dropna()
                lo, hi = boot_median(xx)
                summ.append(dict(subset=tag, source=src, view=view, map=m, n=len(xx), median=xx.median(),
                                 ci_lo=lo, ci_hi=hi))
            for a_ in [m for m in w.columns if ":C" in m]:
                opp = ["entropy", "EBCU"]
                if a_.startswith("a_good:"):   # stessa formula: buoni contro controllo e contro run completa
                    fm = a_.split(":")[1].split("(")[0]
                    opp += [m for m in w.columns if m.startswith((f"a_rand:{fm}(", f"a:{fm}("))]
                for b_ in opp:
                    xy = w[[a_, b_]].dropna()
                    diff = xy[a_] - xy[b_]
                    try:
                        p = wilcoxon(xy[a_], xy[b_]).pvalue
                    except ValueError:
                        p = 1.0
                    tests.append(dict(subset=tag, source=src, view=view, A=a_, B=b_, n=len(xy),
                                      median_diff=diff.median(), A_better=int((diff > 0).sum()),
                                      B_better=int((diff < 0).sum()), p_raw=p))
    summ, tests = pd.DataFrame(summ), pd.DataFrame(tests)
    tests["p_holm"] = np.concatenate([holm(g.p_raw.values) for _, g in tests.groupby("subset")])
    summ.to_csv(f"{OUT}/stats_summary.csv", index=False)
    tests.to_csv(f"{OUT}/stats_tests.csv", index=False)
    lines = [f"Combinazioni residui x U, dataset {E.C['ROOT']}; parametri {chosen}",
             "AUPRC normalizzata, mediana per caso [IC 95%]"]
    for (tag, src), g in summ.groupby(["subset", "source"]):
        g = g.assign(val=g["median"].map("{:.3f}".format) + " [" + g.ci_lo.map("{:.3f}".format) + ", "
                     + g.ci_hi.map("{:.3f}".format) + "]")
        lines += [f"\n### {tag} | {src}", g.pivot(index="map", columns="view", values="val")[views].to_string()]
    k = tests[tests.subset == "all"].copy()
    k["sig"] = np.where(k.p_holm < 0.05, "*", "")
    lines += ["\n### Confronti (tutti i casi), Holm", k.drop(columns=["subset", "p_raw"]).round(4).to_string(index=False)]
    txt = "\n".join(lines)
    open(f"{OUT}/stats_summary.txt", "w").write(txt)
    print(txt)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    {"select": select, "eval": evaluate}[sys.argv[1]](w)
