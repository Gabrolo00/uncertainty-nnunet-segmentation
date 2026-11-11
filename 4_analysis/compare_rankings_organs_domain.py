"""
Concordanza delle classifiche delle misure tra strutture e tra test interno ed esterno, classifica per famiglia
e confronto delle metriche tra i due domini.

Uso: python compare_rankings_organs_domain.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, mannwhitneyu, spearmanr

TS = "/mnt/sda2/grolo"
H = "/mnt/sda2/grolo/humanitas_source"
OUT = f"{TS}/comparisons/organs_and_domain"
STATS = {"rene904_deep": f"{TS}/kidney/deep/ranking_stats/step1_stats_all.csv",
         "rene904_checkpoint": f"{TS}/kidney/checkpoint/ranking_stats/step1_stats_all.csv",
         "trachea_scratch": f"{TS}/trachea/deep/ranking_stats/step1_stats_all.csv"}
FAMILIES = {"mean prediction": ["entropy", "confidence_map", "sample_mean_uncertainty"],
            "disagreement": ["variance", "std", "mutual_information", "epkl", "max_fold_deviation", "boundary_disagreement"],
            "anti-confidence": ["anti_confidence_fn", "anti_confidence_fp", "anti_confidence_total"],
            "distance-aware": ["DEU_entropy", "ebcu_entropy"]}   # le 4 famiglie della tesi (cap. 5)
FAMILY_OF = {m: f for f, ms in FAMILIES.items() for m in ms}
METRICS = ["dice_unc_vs_errors", "recall_total", "precision"]
ROIS = ["roi15mm", "roi15mm_no_border2mm"]


def fdr_bh(p):
    p = np.asarray(p, float)
    ok = ~np.isnan(p)
    q = np.full_like(p, np.nan)
    if ok.sum():
        pv = p[ok]
        o = np.argsort(pv)
        r = pv[o] * len(pv) / np.arange(1, len(pv) + 1)
        r = np.minimum.accumulate(r[::-1])[::-1]
        qq = np.empty_like(pv)
        qq[o] = np.minimum(r, 1)
        q[ok] = qq
    return q


def part_a():
    s = {k: pd.read_csv(v) for k, v in STATS.items()}
    rows, fam_rows = [], []
    for a, b in [("rene904_deep", "trachea_scratch"), ("rene904_deep", "rene904_checkpoint")]:
        for met in METRICS:
            for roi in ROIS:
                x = s[a][(s[a].metric == met) & (s[a].roi_mode == roi)].set_index("measure")
                y = s[b][(s[b].metric == met) & (s[b].roi_mode == roi)].set_index("measure")
                m = x.index.intersection(y.index)
                rho, prho = spearmanr(x.loc[m, "mean"], y.loc[m, "mean"])
                tau, ptau = kendalltau(x.loc[m, "mean"], y.loc[m, "mean"])
                ta, tb = set(x.index[x.in_top_group]), set(y.index[y.in_top_group])
                rows.append(dict(pair=f"{a} vs {b}", metric=met, roi_mode=roi, n_measures=len(m), spearman=rho,
                                 p_spearman=prho, kendall=tau, p_kendall=ptau, top1_a=x["mean"].idxmax(),
                                 top1_b=y["mean"].idxmax(), top_group_a=sorted(ta), top_group_b=sorted(tb),
                                 jaccard_top=len(ta & tb) / max(len(ta | tb), 1)))
    for k, d in s.items():
        for met in METRICS:
            for roi in ROIS:
                x = d[(d.metric == met) & (d.roi_mode == roi)].set_index("measure")["mean"]
                best = {f: x.reindex(ms).max() for f, ms in FAMILIES.items()}
                order = sorted(best, key=best.get, reverse=True)
                fam_rows.append(dict(model=k, metric=met, roi_mode=roi, family_rank=" > ".join(order),
                                     **{f"best_{f}": best[f] for f in FAMILIES},
                                     **{f"argbest_{f}": x.reindex(FAMILIES[f]).idxmax() for f in FAMILIES}))
    r, f = pd.DataFrame(rows), pd.DataFrame(fam_rows)
    r.to_csv(f"{OUT}/A_ranking_concordance.csv", index=False)
    f.to_csv(f"{OUT}/A_family_ranking.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80)
    print("=== A. Concordanza delle classifiche (14 misure)")
    print(r[["pair", "metric", "roi_mode", "spearman", "p_spearman", "kendall", "top1_a", "top1_b", "jaccard_top"]]
          .round(3).to_string(index=False))
    print("\n=== A. Classifica per famiglie (migliore misura di ogni famiglia), F1")
    print(f[f.metric == "dice_unc_vs_errors"][["model", "roi_mode", "family_rank"] + [f"argbest_{x}" for x in FAMILIES]]
          .to_string(index=False))


def part_b():
    ts = pd.read_csv(f"{TS}/kidney/deep/kidney_eval_threshold_ALL/metrics_per_case_4.csv")
    hu = pd.read_csv(f"/mnt/sda2/grolo/kidney/external/eval/evaluate_threshold_deep904/metrics_per_case_4.csv")
    c = pd.read_csv(f"{H}/cohort.csv", dtype=str).fillna("")
    clinical = set(c[(c.excluded == "") & (c.gt_source == "Rene (clinico)")].case)
    hu = hu[hu.case_id.isin(clinical)]
    ts, hu = ts[~ts.union_empty], hu[~hu.union_empty]
    print(f"\nTS: {ts.case_id.nunique()} casi | Humanitas (contorni clinici): {hu.case_id.nunique()} casi")
    hu.to_csv(f"{OUT}/B_humanitas904_clinical_metrics_per_case.csv", index=False)
    rows = []
    for roi in ROIS:
        for u in sorted(ts.uncertainty_type.unique()):
            a = ts[(ts.roi_mode == roi) & (ts.uncertainty_type == u)]
            b = hu[(hu.roi_mode == roi) & (hu.uncertainty_type == u)]
            row = dict(roi_mode=roi, measure=u, family=FAMILY_OF.get(u, ""), tau_TS=a.t_star_train.median())
            for met in METRICS + ["false_alarm_rate"]:
                xa, xb = a[met].dropna(), b[met].dropna()
                row[f"{met}_TS"], row[f"{met}_HUM"] = xa.mean(), xb.mean()
                row[f"{met}_delta"] = xb.mean() - xa.mean()
                row[f"{met}_p"] = mannwhitneyu(xa, xb).pvalue if len(xa) > 5 and len(xb) > 5 else np.nan
            rows.append(row)
    d = pd.DataFrame(rows)
    for met in METRICS + ["false_alarm_rate"]:
        d[f"{met}_q"] = fdr_bh(d[f"{met}_p"])
    d.to_csv(f"{OUT}/B_domain_shift_TS_vs_humanitas.csv", index=False)
    rk = []
    for roi in ROIS:
        x = d[d.roi_mode == roi]
        for met in METRICS:
            rho, p = spearmanr(x[f"{met}_TS"], x[f"{met}_HUM"])
            rk.append(dict(roi_mode=roi, metric=met, spearman_TS_vs_HUM=rho, p=p,
                           top1_TS=x.loc[x[f"{met}_TS"].idxmax(), "measure"],
                           top1_HUM=x.loc[x[f"{met}_HUM"].idxmax(), "measure"]))
    rk = pd.DataFrame(rk)
    rk.to_csv(f"{OUT}/B_ranking_TS_vs_humanitas.csv", index=False)
    print("\n=== B. TotalSegmentator -> Humanitas, F1 (dice_unc_vs_errors) con le tau* di TS")
    for roi in ROIS:
        x = d[d.roi_mode == roi].sort_values("dice_unc_vs_errors_TS", ascending=False)
        print(f"\n-- {roi}")
        print(x[["measure", "tau_TS", "dice_unc_vs_errors_TS", "dice_unc_vs_errors_HUM", "dice_unc_vs_errors_delta",
                 "dice_unc_vs_errors_q", "recall_total_TS", "recall_total_HUM", "precision_TS", "precision_HUM"]]
              .round(3).to_string(index=False))
    print("\n=== B. Concordanza delle classifiche TS vs Humanitas")
    print(rk.round(3).to_string(index=False))


if __name__ == "__main__":
    if os.path.isdir(OUT) and os.listdir(OUT):
        sys.exit(f"{OUT} contiene gia' file: non sovrascrivo")
    os.makedirs(OUT, exist_ok=True)
    part_a()
    part_b()
