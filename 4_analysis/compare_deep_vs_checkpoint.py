"""
Confronto deep ensemble contro checkpoint ensemble: calibrazione (NLL, ECE, Brier a T=1 e T*) e metriche
delle mappe per caso, con test di Wilcoxon appaiati.

Uso: python compare_deep_vs_checkpoint.py --config 904|trachea [--overwrite]
"""
import argparse
import os
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
sys.path[:0] = ['/home/grolo/thesis_pipeline/3_uncertainty/temperature']  # cartelle dei moduli del progetto importati qui
import loo_temperature_scaling as lts

TS = "/mnt/sda2/grolo"
# --config: "904" = rene pulito (Dataset904, 97 casi), output in una cartella separata, comparisons/deep_vs_checkpoint_kidney/.
CONFIGS = {
    "904": dict(P="/mnt/sda2/grolo/kidney/predictions",
                GT="/mnt/sda2/grolo/nnUNet/raw/Dataset904_KidneyTSMechanical/labelsTs",
                OUT=f"{TS}/comparisons/deep_vs_checkpoint_kidney",
                deep=dict(tstar=f"{TS}/kidney/deep/loo_T_summary.csv",
                          metrics=f"{TS}/kidney/deep/kidney_eval_threshold_ALL/metrics_per_case_4.csv"),
                ckpt=dict(tstar=f"{TS}/kidney/checkpoint/loo_T_summary.csv",
                          metrics=f"{TS}/kidney/checkpoint/kidney_cyclr_eval_threshold_ALL/metrics_per_case_4.csv")),
    # Trachea from-scratch (Dataset903, 60 casi di cui 9 senza trachea: ROI vuota, esclusi dalla calibrazione
    # e dai confronti appaiati tramite union_empty). Le predizioni del deep stanno in deep.
    "trachea": dict(P="/mnt/sda2/grolo/trachea/predictions",
                    GT="/mnt/sda2/grolo/nnUNet/raw/Dataset903_TracheaTS/labelsTs",
                    OUT=f"{TS}/comparisons/deep_vs_checkpoint_trachea", deep_dir="deep",
                    deep=dict(tstar=f"{TS}/trachea/deep/loo_T_summary.csv",
                              metrics=f"{TS}/trachea/deep/trachea_eval_threshold_ALL/metrics_per_case_4.csv"),
                    ckpt=dict(tstar=f"{TS}/trachea/checkpoint/loo_T_summary.csv",
                              metrics=f"{TS}/trachea/checkpoint/trachea_cyclr_eval_threshold_ALL/metrics_per_case_4.csv")),
}
ap = argparse.ArgumentParser()
ap.add_argument("--config", default="904", choices=list(CONFIGS))
ap.add_argument("--overwrite", action="store_true", help="riscrive i CSV se la cartella di output esiste gia'")
cfgs = CONFIGS[ap.parse_args().config]
P, GT, OUT = cfgs["P"], cfgs["GT"], cfgs["OUT"]
if os.path.isdir(OUT) and os.listdir(OUT) and not ap.parse_args().overwrite:
    sys.exit(f"{OUT} contiene gia' file: non sovrascrivo (usa --overwrite)")
os.makedirs(OUT, exist_ok=True)
E = {"deep": dict(logits=f"{P}/{cfgs.get('deep_dir', 'deep')}", **cfgs["deep"]),
     "checkpoint": dict(logits=f"{P}/checkpoint", **cfgs["ckpt"])}
print(f"[deep vs checkpoint] {E}", flush=True)

rows = []
cases = sorted(p.name.replace(".nii.gz", "") for p in Path(GT).glob("*.nii.gz"))
for ens, cfg in E.items():
    T = float(pd.read_csv(cfg["tstar"]).T_star.median())
    for c in cases:
        gt, sp = lts.load_gt(Path(f"{GT}/{c}.nii.gz")); gt = (gt > 0).astype(np.int64)
        lg = lts.align_logits_to_gt_transpose_then_resample(lts.load_logits_pt(Path(f"{cfg['logits']}/{c}_logits.pt")), gt.shape)
        roi = lts.build_roi_mask(gt, np.argmax(lg, 0), sp, 15.0)
        if not roi.any():  # GT e predizione vuote (trachea fuori dal campo): calibrazione non definita, caso escluso
            print(ens, c, "ROI vuota: escluso", flush=True)
            continue
        X = lts.logits_to_flat(lg)[roi.reshape(-1)]; y = gt.reshape(-1)[roi.reshape(-1)]
        r = dict(ensemble=ens, case=c, T_star=T)
        for tag, t in [("T1", 1.0), ("Tstar", T)]:
            pr = lts.softmax_T_flat(X, t)
            r[f"nll_{tag}"] = lts.nll_from_logits_flat(X, y, t); r[f"ece_{tag}"] = lts.ece_multiclass_confidence(pr, y)
            r[f"brier_{tag}"] = lts.binary_brier_from_probs(pr, y)
        rows.append(r); print(ens, c, round(r["nll_T1"], 4), round(r["nll_Tstar"], 4), flush=True)
cal = pd.DataFrame(rows); cal.to_csv(f"{OUT}/calibration_per_case.csv", index=False)
s = cal.groupby("ensemble")[["T_star", "nll_T1", "nll_Tstar", "ece_T1", "ece_Tstar", "brier_T1", "brier_Tstar"]].mean()
for m in ["nll", "ece", "brier"]:
    s[f"{m}_delta_%"] = 100 * (s[f"{m}_Tstar"] - s[f"{m}_T1"]) / s[f"{m}_T1"]
s.to_csv(f"{OUT}/calibration_summary.csv"); print(s.round(4).T.to_string())
d, k = cal[cal.ensemble == "deep"].set_index("case"), cal[cal.ensemble == "checkpoint"].set_index("case")
for m in ["nll_Tstar", "ece_Tstar"]:
    print(f"{m}: deep {d[m].mean():.4f} vs checkpoint {k[m].mean():.4f}, Wilcoxon p={wilcoxon(d[m], k.loc[d.index, m]).pvalue:.2e}")

# (2) metriche di incertezza appaiate per misura
md = pd.read_csv(E["deep"]["metrics"]); mk = pd.read_csv(E["checkpoint"]["metrics"])
out = []
for roi in ["roi15mm", "roi15mm_no_border2mm"]:
    for u in sorted(set(md.uncertainty_type) & set(mk.uncertainty_type)):
        a = md[(md.roi_mode == roi) & (md.uncertainty_type == u) & ~md.union_empty].set_index("case_id")
        b = mk[(mk.roi_mode == roi) & (mk.uncertainty_type == u) & ~mk.union_empty].set_index("case_id")
        for met in ["dice_unc_vs_errors", "recall_total", "precision"]:
            j = pd.concat([a[met], b[met]], axis=1, keys=["deep", "ckpt"]).dropna()
            p = wilcoxon(j.deep, j.ckpt).pvalue if len(j) > 5 and (j.deep != j.ckpt).any() else np.nan
            out.append(dict(roi_mode=roi, measure=u, metric=met, n=len(j), deep=j.deep.mean(), checkpoint=j.ckpt.mean(), p=p))
o = pd.DataFrame(out); o.to_csv(f"{OUT}/uncertainty_deep_vs_checkpoint.csv", index=False)
pd.set_option("display.width", 220)
for roi in ["roi15mm", "roi15mm_no_border2mm"]:
    x = o[(o.roi_mode == roi) & (o.metric == "dice_unc_vs_errors")].sort_values("deep", ascending=False)
    print(f"\n=== F1 (dice_unc_vs_errors), {roi}"); print(x[["measure", "n", "deep", "checkpoint", "p"]].round(4).to_string(index=False))
