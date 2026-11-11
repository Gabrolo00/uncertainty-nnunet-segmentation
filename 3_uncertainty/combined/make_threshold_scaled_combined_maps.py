"""
Mappe combinate: ogni misura e' normalizzata e divisa per la propria soglia calibrata, poi le misure sono unite
(switch entropia/deviazione standard e DEU/MI a 2 mm dal bordo, oppure massimo). Sulla mappa unita la soglia e' 1.

Uso: python make_threshold_scaled_combined_maps.py --maps_dir <mappe> --pred_dir <predizioni> --tau_csv <soglie.csv> --out_dir <out> [--link_into_maps]
"""
import argparse
import os
import re
import sys

import nibabel as nib
import numpy as np
import pandas as pd

sys.path[:0] = ['/home/grolo/thesis_pipeline/3_uncertainty/evaluation']  # cartelle dei moduli del progetto importati qui
from evaluate_threshold import compute_signed_distance_transform, normalize_minmax_full  # noqa: E402

BAND_MM = 2.0
COMBOS = {  # nome: (vicino al bordo, lontano dal bordo, modo)
    "combined_thr_switch_entropy_std": ("entropy", "std", "switch"),
    "combined_thr_max_entropy_std": ("entropy", "std", "max"),
    "combined_thr_switch_DEU_MI": ("DEU_entropy", "mutual_information", "switch"),
}
TAU_MIN = 1e-6  # tau = 0 vorrebbe dire "tutto incerto": si evita la divisione per zero


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps_dir", required=True)
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--tau_csv", required=True, help="CSV con case_id, uncertainty_type, t_star_train (righe per caso)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--link_into_maps", action="store_true")
    a = ap.parse_args()
    if os.path.isdir(a.out_dir) and os.listdir(a.out_dir):
        sys.exit(f"{a.out_dir} contiene gia' file: non sovrascrivo")
    os.makedirs(a.out_dir, exist_ok=True)
    tau = pd.read_csv(a.tau_csv)
    tau = tau[tau.case_id != "SUMMARY"].set_index(["case_id", "uncertainty_type"]).t_star_train
    cases = sorted(m.group(1) for f in os.listdir(a.maps_dir) if (m := re.match(r"^(.+)_entropy\.nii\.gz$", f))
                   and not f.endswith(("_DEU_entropy.nii.gz", "_ebcu_entropy.nii.gz", "_expected_entropy.nii.gz",
                                       "_total_entropy.nii.gz")))
    print(f"{len(cases)} casi | soglie da {a.tau_csv}", flush=True)
    rows = []
    for i, c in enumerate(cases, 1):
        pred_img = nib.load(f"{a.pred_dir}/{c}.nii.gz")
        pred = pred_img.get_fdata() > 0.5
        spacing = pred_img.header.get_zooms()[:3]
        band = (np.abs(compute_signed_distance_transform(pred, spacing)) <= BAND_MM) if pred.any() \
            else np.zeros(pred.shape, bool)
        cache = {}
        for name, (near, far, mode) in COMBOS.items():
            for u in (near, far):
                if u not in cache:
                    img = nib.load(f"{a.maps_dir}/{c}_{u}.nii.gz")
                    assert img.shape == pred.shape and np.allclose(img.affine, pred_img.affine, atol=1e-3), (c, u)
                    t = max(float(tau.loc[(c, u)]), TAU_MIN)
                    cache[u] = normalize_minmax_full(img.get_fdata()).astype(np.float64) / t
            x = np.where(band, cache[near], cache[far]) if mode == "switch" else np.maximum(cache[near], cache[far])
            mn, mx = float(x.min()), float(x.max())
            t_eq = (1.0 - mn) / (mx - mn) if mx > mn else (0.0 if mn >= 1.0 else 2.0)
            out = nib.Nifti1Image(x.astype(np.float32), pred_img.affine)
            out.header.set_zooms(spacing)
            dst = f"{a.out_dir}/{c}_{name}.nii.gz"
            nib.save(out, dst)
            if a.link_into_maps:
                link = f"{a.maps_dir}/{c}_{name}.nii.gz"
                if os.path.lexists(link):
                    os.remove(link)
                os.symlink(dst, link)
            rows.append(dict(case_id=c, uncertainty_type=name, t_star_train=t_eq, score_test=np.nan,
                             frac_above_1=float((x >= 1.0).mean()), tau_near=float(tau.loc[(c, near)]),
                             tau_far=float(tau.loc[(c, far)])))
        print(f"[{i}/{len(cases)}] {c}", flush=True)
    pd.DataFrame(rows).to_csv(f"{a.out_dir}/thresholds_for_evaluate.csv", index=False)


if __name__ == "__main__":
    main()
