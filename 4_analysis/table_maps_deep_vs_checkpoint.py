"""
Tabella recall/F1 delle mappe rappresentative di ogni famiglia, deep contro checkpoint, con test di Wilcoxon
e correzione di Benjamini-Hochberg.

Uso: python table_maps_deep_vs_checkpoint.py --config kidney904|trachea
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

T = "/mnt/sda2/grolo"
# --config kidney904 (default, Tabella 5.3) | trachea (casi union_empty esclusi: GT e predizione vuote)
CONFIGS = {
    "kidney904": dict(deep=f"{T}/kidney/deep/kidney_eval_threshold_ALL/metrics_per_case_4.csv",
                      ckpt=f"{T}/kidney/checkpoint/kidney_cyclr_eval_threshold_ALL/metrics_per_case_4.csv",
                      out=f"{T}/comparisons/deep_vs_checkpoint_kidney/table_maps_deep_vs_checkpoint.csv"),
    "trachea": dict(deep=f"{T}/trachea/deep/trachea_eval_threshold_ALL/metrics_per_case_4.csv",
                    ckpt=f"{T}/trachea/checkpoint/trachea_cyclr_eval_threshold_ALL/metrics_per_case_4.csv",
                    out=f"{T}/comparisons/deep_vs_checkpoint_trachea/table_maps_deep_vs_checkpoint.csv"),
}
DEEP, CKPT, OUT = CONFIGS["kidney904"]["deep"], CONFIGS["kidney904"]["ckpt"], CONFIGS["kidney904"]["out"]
MAPS = ["entropy", "variance", "mutual_information", "anti_confidence_total", "DEU_entropy", "ebcu_entropy"]


def main(rule="errors_nan0"):
    d, c = pd.read_csv(DEEP), pd.read_csv(CKPT)
    d, c = d[~d.union_empty], c[~c.union_empty]
    # 14 misure (le combinate, se presenti, non entrano nella regola dei casi completi)
    d, c = d[~d.uncertainty_type.str.startswith("combined_")], c[~c.uncertainty_type.str.startswith("combined_")]
    rows = []
    for roi in ["roi15mm", "roi15mm_no_border2mm"]:
        for met in ["recall_total", "dice_unc_vs_errors"]:
            for u in MAPS:
                a = d[(d.roi_mode == roi) & (d.uncertainty_type == u)].set_index("case_id")
                b = c[(c.roi_mode == roi) & (c.uncertainty_type == u)].set_index("case_id")
                if rule == "complete":  # come la statistica di classifica: casi con la metrica
                    # definita per TUTTE le misure, in entrambi i modelli; nessun NaN contato come 0
                    pa = d[d.roi_mode == roi].pivot(index="case_id", columns="uncertainty_type", values=met).dropna()
                    pb = c[c.roi_mode == roi].pivot(index="case_id", columns="uncertainty_type", values=met).dropna()
                    keep = pa.index.intersection(pb.index)
                else:
                    keep = a.index[a.num_errors_total > 0]
                x, y = a.loc[keep, met].fillna(0), b.loc[keep, met].fillna(0)
                rows.append(dict(roi_mode=roi, metric=met, map=u, n=len(keep), deep=x.mean(), checkpoint=y.mean(),
                                 p=wilcoxon(x, y).pvalue))
    r = pd.DataFrame(rows)
    p = r.p.values
    o = np.argsort(p)
    q = np.empty_like(p)
    q[o] = np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    r["q_fdr"] = np.minimum(q, 1)
    r["significant"] = r.q_fdr < 0.05
    r.to_csv(OUT, index=False)
    print(r.round(4).to_string(index=False))


if __name__ == "__main__":
    import argparse
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="kidney904", choices=list(CONFIGS))
    cfg = CONFIGS[ap.parse_args().config]
    DEEP, CKPT, OUT = cfg["deep"], cfg["ckpt"], cfg["out"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    main()
    print("\n=== regola dei casi completi ===")
    OUT = OUT.replace(".csv", "_complete_cases.csv")
    main(rule="complete")
