"""
Valutazione sul test esterno Humanitas, senza ricalibrazione: soglie trasferite dal test interno, segmentazione,
calibrazione e metriche senza soglia delle mappe.

Uso: python humanitas_eval.py --config 904|904ft --stage thresholds|segmentation|calibration|threshold_free
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

B = "/mnt/sda2/grolo/humanitas_source"
TS = "/mnt/sda2/grolo"
# --config: "904" = deep ensemble from scratch, con T* e soglie del test interno;
# "904ft" = from scratch contro fine-tuned sullo stesso ritaglio (solo segmentazione, calibrazione e metriche senza soglia).
CONFIGS = {
    "904": dict(work=f"/mnt/sda2/grolo/kidney/external", models={
        "deep904": dict(pred=f"/mnt/sda2/grolo/kidney/external/pred_deep", maps=f"/mnt/sda2/grolo/kidney/external/maps_deep", tstar_csv=f"{TS}/kidney/deep/loo_T_summary.csv",
                        tau_csv=f"{TS}/kidney/deep/kidney_threshold_calibration_ALL.csv")}),
    # Il fine-tuned non ha soglie calibrate (il suo test interno era nei dati di pretraining).
    "904ft": dict(work=f"/mnt/sda2/grolo/kidney/external", out=f"/mnt/sda2/grolo/kidney/external/eval_scratch_vs_finetune", models={
        "deep904": dict(pred=f"/mnt/sda2/grolo/kidney/external/pred_deep", maps=f"/mnt/sda2/grolo/kidney/external/maps_deep", tstar_csv=f"{TS}/kidney/deep/loo_T_summary.csv",
                        tau_csv=None),
        "finetune904": dict(pred=f"/mnt/sda2/grolo/kidney/external/pred_finetune", maps=f"/mnt/sda2/grolo/kidney/external/maps_finetune",
                            tstar_csv=f"{TS}/kidney/finetune/loo_T_summary.csv", tau_csv=None)}),
}
MEASURES = ["entropy", "DEU_entropy", "ebcu_entropy", "variance", "std", "mutual_information", "epkl",
            "sample_mean_uncertainty", "confidence_map", "boundary_disagreement", "max_fold_deviation",
            "anti_confidence_fn", "anti_confidence_fp", "anti_confidence_total"]
W = CONFIGS["904"]["work"]  # cartella di lavoro (cropped/, crop_report.csv, eval/), impostata da --config
MODELS = CONFIGS["904"]["models"]
OUT = f"{W}/eval"


def cohort():
    c = pd.read_csv(f"{B}/cohort.csv", dtype=str).fillna("")
    c = c[c.excluded == ""]
    r = pd.read_csv(f"{W}/crop_report.csv")
    return c.merge(r[r.status == "ok"][["case", "pass1_dice_full", "pass1_fp_outside_crop_ml"]], on="case")


def tstar(model):
    return float(pd.read_csv(MODELS[model]["tstar_csv"]).T_star.median())


def stage_thresholds():
    os.makedirs(OUT, exist_ok=True)
    c = cohort()
    for model, cfg in MODELS.items():
        if cfg["tau_csv"] is None:  # modello senza soglie calibrate (fine-tuned)
            print(model, "senza tau*: saltato")
            continue
        t = pd.read_csv(cfg["tau_csv"])
        s = t[t.case_id == "SUMMARY"][["uncertainty_type", "t_star_train"]]
        rows = [dict(case_id=case, uncertainty_type=u, t_star_train=tau, score_test=np.nan)
                for case in c.case for u, tau in zip(s.uncertainty_type, s.t_star_train)]
        pd.DataFrame(rows).to_csv(f"{OUT}/tau_from_TS_{model}.csv", index=False)
        print(model, "tau* da TotalSegmentator:", dict(zip(s.uncertainty_type, s.t_star_train.round(3))))


def dice(a, b):
    s = a.sum() + b.sum()
    return 2 * (a & b).sum() / s if s else np.nan


def stage_segmentation():
    import SimpleITK as sitk
    from scipy.stats import wilcoxon
    c = cohort()
    rows = []
    for _, r in c.iterrows():
        gi = sitk.ReadImage(f"{W}/cropped/labelsTs/{r.case}.nii.gz")
        g = sitk.GetArrayFromImage(gi) > 0
        v = np.prod(gi.GetSpacing()) / 1000
        row = dict(case=r.case, gt_source=r.gt_source, contour_system=r.contour_system, approval=r.approval,
                   gt_ml=g.sum() * v, pass1_dice_full_ct=r.pass1_dice_full,
                   pass1_fp_outside_crop_ml=r.pass1_fp_outside_crop_ml)
        for model, cfg in MODELS.items():
            p = sitk.GetArrayFromImage(sitk.ReadImage(f"{cfg['pred']}/{r.case}.nii.gz")) > 0
            row[f"dice_{model}"] = dice(p, g)
            row[f"pred_ml_{model}"] = p.sum() * v
            row[f"fp_ml_{model}"] = (p & ~g).sum() * v
            row[f"fn_ml_{model}"] = (~p & g).sum() * v
        rows.append(row)
    d = pd.DataFrame(rows)
    d.to_csv(f"{OUT}/segmentation_per_case.csv", index=False)
    names = list(MODELS)
    for name, x in [("contorni clinici", d[d.gt_source == "Rene (clinico)"]), ("tutti", d)]:
        txt = " | ".join(f"Dice {m} {x[f'dice_{m}'].mean():.4f} (mediana {x[f'dice_{m}'].median():.4f})" for m in names)
        if len(names) == 2:  # confronto appaiato tra i due modelli
            a, b = (x[f"dice_{m}"] for m in names)
            txt += f" | Wilcoxon p={wilcoxon(a, b).pvalue:.2e} | {names[1]} migliore in {(b > a).sum()}/{len(x)}"
        print(f"[{name}, n={len(x)}] {txt}")
    print(d.groupby(["gt_source", "contour_system", "approval"])[[f"dice_{m}" for m in names]].agg(["mean", "count"]).round(4).to_string())


def stage_calibration():
    sys.path[:0] = ['/home/grolo/thesis_pipeline/3_uncertainty/temperature']  # cartelle dei moduli del progetto importati qui
    from pathlib import Path
    import loo_temperature_scaling as lts
    c = cohort()
    rows = []
    for model, cfg in MODELS.items():
        T = tstar(model)
        for _, r in c.iterrows():
            gt, sp = lts.load_gt(Path(f"{W}/cropped/labelsTs/{r.case}.nii.gz"))
            gt = (gt > 0).astype(np.int64)
            lg = lts.align_logits_to_gt_transpose_then_resample(
                lts.load_logits_pt(Path(f"{cfg['pred']}/{r.case}_logits.pt")), gt.shape)
            pred = np.argmax(lg, axis=0)
            roi = lts.build_roi_mask(gt, pred, sp, 15.0)
            X = lts.logits_to_flat(lg)[roi.reshape(-1)]
            y = gt.reshape(-1)[roi.reshape(-1)]
            row = dict(model=model, case=r.case, gt_source=r.gt_source, T_star=T, n_vox_roi=int(roi.sum()))
            for tag, t in [("T1", 1.0), ("Tstar", T)]:
                pr = lts.softmax_T_flat(X, t)
                row[f"nll_{tag}"] = lts.nll_from_logits_flat(X, y, t)
                row[f"ece_{tag}"] = lts.ece_multiclass_confidence(pr, y)
                row[f"brier_{tag}"] = lts.binary_brier_from_probs(pr, y)
            rows.append(row)
            print(model, r.case, {k: round(v, 4) for k, v in row.items() if k.startswith(("nll", "ece"))}, flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(f"{OUT}/calibration_per_case.csv", index=False)
    x = d[d.gt_source == "Rene (clinico)"]
    s = x.groupby("model")[["nll_T1", "nll_Tstar", "ece_T1", "ece_Tstar", "brier_T1", "brier_Tstar"]].mean()
    for m in ["nll", "ece", "brier"]:
        s[f"{m}_delta_%"] = 100 * (s[f"{m}_Tstar"] - s[f"{m}_T1"]) / s[f"{m}_T1"]
    print(s.round(4).to_string())


def stage_threshold_free():
    import SimpleITK as sitk
    from scipy.ndimage import distance_transform_edt
    from sklearn.metrics import roc_auc_score, average_precision_score
    c = cohort()
    rows = []
    for _, r in c.iterrows():
        gi = sitk.ReadImage(f"{W}/cropped/labelsTs/{r.case}.nii.gz")
        g = sitk.GetArrayFromImage(gi) > 0
        sp = np.array(gi.GetSpacing())[::-1]
        for model, cfg in MODELS.items():
            p = sitk.GetArrayFromImage(sitk.ReadImage(f"{cfg['pred']}/{r.case}.nii.gz")) > 0
            err = p ^ g
            # stesse ROI di evaluate_threshold.py: unione + 15 mm; no-border = |dist dal bordo della pred| > 2 mm
            union = p | g
            roi15 = distance_transform_edt(~union, sampling=sp) <= 15.0
            sd_pred = distance_transform_edt(p, sampling=sp) - distance_transform_edt(~p, sampling=sp)
            rois = {"roi15mm": roi15, "roi15mm_no_border2mm": roi15 & (np.abs(sd_pred) > 2.0)}
            for u in MEASURES:
                f = f"{cfg['maps']}/{r.case}_{u}.nii.gz"
                if not os.path.exists(f):
                    continue
                unc = sitk.GetArrayFromImage(sitk.ReadImage(f)).astype(np.float32)
                if u == "confidence_map":
                    unc = -unc  # piu' alto = piu' incerto
                for rname, roi in rois.items():
                    e, s_ = err[roi], unc[roi]
                    if e.sum() == 0 or e.sum() == e.size:
                        auroc = auprc = np.nan
                    else:
                        auroc = roc_auc_score(e, s_)
                        auprc = average_precision_score(e, s_)
                    rows.append(dict(model=model, case=r.case, gt_source=r.gt_source, measure=u, roi_mode=rname,
                                     n_err=int(e.sum()), n_roi=int(e.size), prevalence=e.mean(),
                                     auroc=auroc, auprc=auprc))
        print(r.case, flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(f"{OUT}/threshold_free_per_case.csv", index=False)
    x = d[d.gt_source == "Rene (clinico)"]
    pd.set_option("display.width", 250)
    for rname in ["roi15mm", "roi15mm_no_border2mm"]:
        t = x[x.roi_mode == rname].pivot_table(index="measure", columns="model", values=["auroc", "auprc"], aggfunc="mean")
        print(f"\n=== {rname} (contorni clinici) ===")
        print(t.round(3).sort_values(("auprc", list(MODELS)[0]), ascending=False).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["thresholds", "segmentation", "calibration", "threshold_free"])
    ap.add_argument("--config", default="904", choices=list(CONFIGS))
    a = ap.parse_args()
    W = CONFIGS[a.config]["work"]
    MODELS = CONFIGS[a.config]["models"]
    OUT = CONFIGS[a.config].get("out", f"{W}/eval")
    print(f"[humanitas_eval] config {a.config}: work_dir {W}, modelli {list(MODELS)}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    {"thresholds": stage_thresholds, "segmentation": stage_segmentation, "calibration": stage_calibration,
     "threshold_free": stage_threshold_free}[a.stage]()
