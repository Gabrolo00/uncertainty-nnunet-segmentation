"""
Statistica dei risultati dell'autoencoder: mediana per caso dell'AUPRC normalizzata con intervallo bootstrap e
confronti appaiati (Wilcoxon, correzione di Holm) con entropia ed EBCU.

Uso: AE_DATASET=904 python autoencoder_stats.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path[:0] = ['/home/grolo/thesis_pipeline/5_autoencoder']  # cartelle dei moduli del progetto importati qui
from autoencoder_config import DS, EVAL_DIR  # noqa: E402

VIEWS = ["roi15", "far2", "farVSall2", "far4.5", "farVSall4.5"]
REF = ["entropy", "mi", "DEU", "EBCU"]
SELECTION_FOLD = 1
MASS = ["s0543", "s0545"]


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    out, run = np.empty(len(p)), 0.0
    for i, k in enumerate(o):
        run = max(run, (len(p) - i) * p[k])
        out[k] = min(run, 1.0)
    return out


def boot_median(x, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    b = np.median(rng.choice(x, (n, len(x))), axis=1)
    return np.percentile(b, [2.5, 97.5])


def analyse(d, tag):
    ae_maps = sorted(m for m in d["map"].unique() if ":" in m)
    rows, tests = [], []
    for (src, view), g in d.groupby(["source", "view"]):
        w = g.pivot_table(index="case", columns="map", values="auprc_norm")
        for m in w.columns:
            x = w[m].dropna()
            lo, hi = boot_median(x)
            rows.append(dict(subset=tag, source=src, view=view, map=m, n=len(x), median=x.median(), ci_lo=lo, ci_hi=hi))
        pairs = [(r, "entropy") for r in ["EBCU", "DEU", "mi"]] + [(m, "entropy") for m in ae_maps] + \
                [(m, "DEU") for m in ae_maps]
        for a_, b_ in pairs:
            if a_ in w and b_ in w:
                x = w[[a_, b_]].dropna()
                diff = x[a_] - x[b_]
                try:
                    p = wilcoxon(x[a_], x[b_]).pvalue
                except ValueError:
                    p = 1.0
                tests.append(dict(subset=tag, source=src, view=view, A=a_, B=b_, n=len(x),
                                  median_diff=diff.median(), A_better=int((diff > 0).sum()),
                                  B_better=int((diff < 0).sum()), p_raw=p))
    t = pd.DataFrame(tests)
    t["p_holm"] = holm(t.p_raw.values)
    return pd.DataFrame(rows), t


def main():
    d = pd.read_csv(f"{EVAL_DIR}/eval_per_case.csv")
    if "auprc_norm" not in d:
        raise SystemExit("eval_per_case.csv senza auprc_norm: rilanciare autoencoder_eval.py")
    d = d[d["view"].isin(VIEWS)]
    subsets = {"all": d, "noSelectionFold": d[d.fold != SELECTION_FOLD], "noMass": d[~d.case.isin(MASS)]}
    summ, tests = [], []
    for tag, x in subsets.items():
        s, t = analyse(x, tag)
        summ.append(s)
        tests.append(t)
    summ, tests = pd.concat(summ), pd.concat(tests)
    summ.to_csv(f"{EVAL_DIR}/stats_summary.csv", index=False)
    tests.to_csv(f"{EVAL_DIR}/stats_tests.csv", index=False)
    lines = [f"Dataset {DS} — AUPRC normalizzato (AUPRC - p)/(1 - p), mediana per caso [IC 95% bootstrap]"]
    for (tag, src), g in summ.groupby(["subset", "source"]):
        g = g.assign(val=g["median"].map("{:.3f}".format) + " [" + g.ci_lo.map("{:.3f}".format) + ", " +
                     g.ci_hi.map("{:.3f}".format) + "]")
        lines += [f"\n### {tag} | {src}", g.pivot(index="map", columns="view", values="val")[VIEWS].to_string()]
    k = tests[(tests.subset == "all")].copy()
    k["sig"] = np.where(k.p_holm < 0.05, "*", "")
    lines += ["\n### Confronti (tutti i casi): differenza mediana A-B, p di Holm",
              k[["source", "view", "A", "B", "n", "median_diff", "A_better", "B_better", "p_holm", "sig"]]
              .round(4).to_string(index=False)]
    txt = "\n".join(lines)
    open(f"{EVAL_DIR}/stats_summary.txt", "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
