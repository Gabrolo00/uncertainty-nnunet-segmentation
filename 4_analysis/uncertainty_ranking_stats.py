"""
Classifica statistica delle misure di incertezza: test di Friedman, poi Wilcoxon tra la migliore misura e le
altre con correzione di Benjamini-Hochberg.

Uso: python uncertainty_ranking_stats.py --per_case_csv <metrics_per_case.csv> --out_dir <out>
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

METRICS = ["dice_unc_vs_errors", "f1_score", "recall_total", "recall_FP", "recall_FN",
           "precision", "false_alarm_rate"]
LOWER_IS_BETTER = {"false_alarm_rate"}
ALPHA = 0.05


def fdr_bh(pvals):
    """p corretti con Benjamini-Hochberg (= statsmodels multipletests(method='fdr_bh')[1],
    statsmodels non e' installato nell'ambiente)."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(ranked, 1.0)
    return out


def analyse(df, metric, roi):
    m = df[df.roi_mode == roi].pivot(index="case_id", columns="uncertainty_type", values=metric)
    n_before = len(m)
    m = m.dropna(axis=0, how="any")
    means = m.mean()
    top1 = means.idxmin() if metric in LOWER_IS_BETTER else means.idxmax()

    fried_p = friedmanchisquare(*[m[c].values for c in m.columns]).pvalue if len(m) > 2 else np.nan

    others = [c for c in m.columns if c != top1]
    raw = []
    for c in others:
        try:
            raw.append(wilcoxon(m[top1].values, m[c].values).pvalue)
        except ValueError:  # tutte le differenze nulle
            raw.append(1.0)
    adj = fdr_bh(raw)

    rows = [dict(metric=metric, roi_mode=roi, measure=top1, mean=means[top1], std=m[top1].std(),
                 median=m[top1].median(), p_raw=np.nan, p_adj=np.nan, top1=True, in_top_group=True)]
    for c, pr, pa in zip(others, raw, adj):
        rows.append(dict(metric=metric, roi_mode=roi, measure=c, mean=means[c], std=m[c].std(),
                         median=m[c].median(), p_raw=pr, p_adj=pa, top1=False,
                         in_top_group=bool(pa >= ALPHA)))
    res = pd.DataFrame(rows)
    res["n_patients"] = len(m)
    res["n_patients_dropped_nan"] = n_before - len(m)
    res["friedman_p"] = fried_p
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per_case_csv", required=True)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.per_case_csv)
    noise = sorted(df[df.union_empty].case_id.unique())
    print(f"Casi union_empty esclusi ({len(noise)}): {noise}")
    df = df[~df.union_empty]

    allres = pd.concat([analyse(df, metric, roi) for metric in METRICS
                        for roi in ["roi15mm", "roi15mm_no_border2mm"]], ignore_index=True)
    allres.to_csv(os.path.join(args.out_dir, "step1_stats_all.csv"), index=False)

    # Tabella riassuntiva: medie per misura x (metrica, ROI)
    for roi in ["roi15mm", "roi15mm_no_border2mm"]:
        t = allres[allres.roi_mode == roi].pivot(index="measure", columns="metric", values="mean")
        g = allres[allres.roi_mode == roi].pivot(index="measure", columns="metric", values="in_top_group")
        t = t[METRICS].sort_values("dice_unc_vs_errors", ascending=False)
        t.to_csv(os.path.join(args.out_dir, f"step1_means_{roi}.csv"))
        mark = t.round(3).astype(str) + g[METRICS].reindex(t.index).replace({True: " *", False: ""})
        info = allres[(allres.roi_mode == roi)].groupby("metric")[["n_patients", "friedman_p"]].first()
        print(f"\n=== {roi}  (* = nel gruppo di testa: non significativamente diversa dalla migliore, "
              f"Wilcoxon + FDR-BH) ===")
        print(info.loc[METRICS].T.to_string())
        print(mark.to_string())

    # Casi rumore: falsi allarmi sull'intero... (la ROI degenera: si riporta solo n_uncertain)
    print(f"\nRisultati salvati in {args.out_dir}")


if __name__ == "__main__":
    main()
