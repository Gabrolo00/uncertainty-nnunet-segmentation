"""
Concordanza delle classifiche delle 14 misure tra i modelli (Spearman, Kendall) e classifica per famiglia,
inclusi i checkpoint ensemble.

Uso: python compare_rankings_checkpoint_pairs.py
"""
import pandas as pd
from scipy.stats import kendalltau, spearmanr

T = "/mnt/sda2/grolo"
OUT = f"{T}/comparisons/organs_and_domain"
S = {"rene_deep": f"{T}/kidney/deep/ranking_stats/step1_stats_all.csv",
     "rene_checkpoint": f"{T}/kidney/checkpoint/ranking_stats/step1_stats_all.csv",
     "trachea_deep": f"{T}/trachea/deep/ranking_stats/step1_stats_all.csv",
     "trachea_checkpoint": f"{T}/trachea/checkpoint/ranking_stats/step1_stats_all.csv"}
PAIRS = [("rene_deep", "trachea_deep"), ("rene_checkpoint", "trachea_checkpoint"),
         ("rene_deep", "rene_checkpoint"), ("trachea_deep", "trachea_checkpoint")]
FAMILIES = {"mean prediction": ["entropy", "confidence_map", "sample_mean_uncertainty"],
            "disagreement": ["variance", "std", "mutual_information", "epkl", "max_fold_deviation", "boundary_disagreement"],
            "anti-confidence": ["anti_confidence_total", "anti_confidence_fn", "anti_confidence_fp"],
            "distance-aware": ["DEU_entropy", "ebcu_entropy"]}
ROIS = ["roi15mm", "roi15mm_no_border2mm"]
s = {k: pd.read_csv(v) for k, v in S.items()}
rows = []
for a, b in PAIRS:
    for roi in ROIS:
        x = s[a][(s[a].metric == "dice_unc_vs_errors") & (s[a].roi_mode == roi)].set_index("measure")
        y = s[b][(s[b].metric == "dice_unc_vs_errors") & (s[b].roi_mode == roi)].set_index("measure")
        m = x.index.intersection(y.index)
        assert len(m) == 14, (a, b, len(m))
        ta, tb = set(x.index[x.in_top_group]), set(y.index[y.in_top_group])
        rows.append(dict(pair=f"{a} vs {b}", roi_mode=roi, spearman=spearmanr(x.loc[m, "mean"], y.loc[m, "mean"])[0],
                         p=spearmanr(x.loc[m, "mean"], y.loc[m, "mean"])[1], kendall=kendalltau(x.loc[m, "mean"], y.loc[m, "mean"])[0],
                         top1_a=x["mean"].idxmax(), top1_b=y["mean"].idxmax(), jaccard_top=len(ta & tb) / max(len(ta | tb), 1)))
r = pd.DataFrame(rows); r.to_csv(f"{OUT}/A_ranking_concordance_all_models.csv", index=False)
pd.set_option("display.width", 200)
print(r.round(3).to_string(index=False))
fam = []
for k, d in s.items():
    for roi in ROIS:
        x = d[(d.metric == "dice_unc_vs_errors") & (d.roi_mode == roi)].set_index("measure")["mean"]
        best = {f: (x.reindex(ms).max(), x.reindex(ms).idxmax()) for f, ms in FAMILIES.items()}
        order = sorted(best, key=lambda f: best[f][0], reverse=True)
        fam.append(dict(model=k, roi_mode=roi, family_rank=" > ".join(order), **{f: f"{best[f][1]} {best[f][0]:.3f}" for f in FAMILIES}))
f = pd.DataFrame(fam); f.to_csv(f"{OUT}/A_family_ranking_4families.csv", index=False)
print(f[["model", "roi_mode", "family_rank"]].to_string(index=False))
