#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Temperature scaling con leave-one-out: T* che minimizza la NLL nella regione di interesse (15 mm attorno a
reference e predizione), con NLL, ECE e Brier prima e dopo.

Uso: python loo_temperature_scaling.py --logits_dir <logit OOF> --gt_dir <label> --out_summary_csv <out.csv>
"""

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, zoom


# -------------------------
# I/O helpers
# -------------------------

def parse_case_id_from_logits(path: Path) -> str:
    m = re.match(r"^(.*)_logits\.pt$", path.name)
    if not m:
        raise ValueError(f"Unexpected logits filename: {path.name} (expected <CASE>_logits.pt)")
    return m.group(1)


def load_gt(gt_path: Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    nii = nib.load(str(gt_path))
    gt = nii.get_fdata().astype(np.int16)
    zooms = nii.header.get_zooms()[:3]  # (x,y,z) in mm
    spacing_xyz = (float(zooms[0]), float(zooms[1]), float(zooms[2]))
    return gt, spacing_xyz


def load_logits_pt(logits_path: Path) -> np.ndarray:
    t = torch.load(str(logits_path), map_location="cpu")
    if isinstance(t, dict):
        for k in ["logits", "pred_logits", "output", "out"]:
            if k in t:
                t = t[k]
                break
        if isinstance(t, dict):
            raise ValueError(f"Loaded dict from {logits_path}, but couldn't find logits tensor key.")

    if not torch.is_tensor(t):
        raise ValueError(f"Expected torch tensor in {logits_path}, got {type(t)}")

    t = t.detach().float()

    # Accept (1,C,D,H,W) or (C,D,H,W)
    if t.ndim == 5:
        t = t[0]
    if t.ndim != 4:
        raise ValueError(f"Expected logits (C,D,H,W), got {tuple(t.shape)} in {logits_path}")

    return t.numpy()  # (C,D,H,W)


def align_logits_to_gt_transpose_then_resample(
    logits: np.ndarray,  # (C,D,H,W)
    gt_shape_hwd: Tuple[int, int, int],
) -> np.ndarray:
    """
    ALWAYS: (C, D, H, W) -> (C, H, W, D)
    THEN: if spatial != gt_shape -> resample (order=1) to gt_shape
    """
    if logits.ndim != 4:
        raise ValueError(f"Expected logits with 4 dims (C,D,H,W), got {logits.shape}")

    C = logits.shape[0]

    # 1) fixed transpose
    logits = np.transpose(logits, (0, 3, 2, 1))  # (C,H,W,D)

    # 2) resample if needed
    spatial = logits.shape[1:]  # (H,W,D)
    if spatial != gt_shape_hwd:
        zoom_factors = (
            gt_shape_hwd[0] / spatial[0],
            gt_shape_hwd[1] / spatial[1],
            gt_shape_hwd[2] / spatial[2],
        )
        out = np.empty((C, gt_shape_hwd[0], gt_shape_hwd[1], gt_shape_hwd[2]), dtype=np.float32)
        for c in range(C):
            out[c] = zoom(
                logits[c].astype(np.float32, copy=False),
                zoom_factors,
                order=1,
                mode="nearest"
            )
        return out

    return logits.astype(np.float32, copy=False)


# -------------------------
# ROI + flattening
# -------------------------

def build_roi_mask(
    gt_hwd: np.ndarray,
    pred_hwd: np.ndarray,
    spacing_xyz: Tuple[float, float, float],
    dilate_mm: float
) -> np.ndarray:
    """
    ROI = distance_transform_edt(~union, sampling=spacing_xyz) <= dilate_mm
    union = (gt>0) ∪ (pred>0)
    Note: gt and pred are (H,W,D). spacing is (x,y,z) but our array axes are (H,W,D).
    In nibabel, zooms are (x,y,z) corresponding to array axes (i,j,k). Here (H,W,D) aligns
    to (x,y,z) if your data is stored that way (common in nnUNet exports). If not, use with caution.
    """
    union = (gt_hwd > 0) | (pred_hwd > 0)
    if not np.any(union):
        return np.zeros_like(union, dtype=bool)

    dist = distance_transform_edt(~union, sampling=spacing_xyz)
    roi = dist <= dilate_mm
    return roi.astype(bool)


def logits_to_flat(logits_chwd: np.ndarray) -> np.ndarray:
    # (C,H,W,D) -> (N,C)
    C = logits_chwd.shape[0]
    flat = np.moveaxis(logits_chwd, 0, -1).reshape(-1, C)
    return flat


# -------------------------
# Metrics
# -------------------------

def softmax_T_flat(logits_flat: np.ndarray, T: float) -> np.ndarray:
    z = logits_flat / float(T)
    z = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=1, keepdims=True)


def nll_from_logits_flat(logits_flat: np.ndarray, gt_flat: np.ndarray, T: float) -> float:
    probs = softmax_T_flat(logits_flat, T)
    eps = 1e-8
    return float(-np.mean(np.log(probs[np.arange(len(gt_flat)), gt_flat] + eps)))


def binary_brier_from_probs(
    probs: np.ndarray,           # shape (N, C), già softmax e già ROI
    gt: np.ndarray,              # shape (N,), già ROI
    background_class: int = 0,
) -> float:
    probs = np.clip(probs, 0.0, 1.0)
    p_fg = 1.0 - probs[:, background_class]                 # P(foreground)
    y_fg = (gt != background_class).astype(np.float32)      # GT binario
    return float(np.mean((p_fg - y_fg) ** 2))



def ece_multiclass_confidence(probs: np.ndarray, gt: np.ndarray, n_bins: int = 15) -> float:
    """
    Multiclass ECE on:
      confidence = max_c p(c|x)
      correctness = 1(argmax==gt)
    """
    conf = np.max(probs, axis=1)
    pred = np.argmax(probs, axis=1)
    corr = (pred == gt).astype(np.float32)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(gt)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            m = (conf >= lo) & (conf <= hi)
        else:
            m = (conf >= lo) & (conf < hi)
        n = int(np.sum(m))
        if n == 0:
            continue
        acc = float(np.mean(corr[m]))
        cbar = float(np.mean(conf[m]))
        ece += (n / N) * abs(acc - cbar)

    return float(ece)


# -------------------------
# T* selection by grid + refine
# -------------------------

def logspace_T(T_min: float, T_max: float, K: int) -> np.ndarray:
    # log-spaced between T_min and T_max inclusive
    if T_min <= 0 or T_max <= 0:
        raise ValueError("T_min and T_max must be > 0")
    if K < 2:
        return np.array([T_min], dtype=np.float64)
    return np.logspace(np.log10(T_min), np.log10(T_max), K)


def select_T_star_by_coarse_then_refine(
    logits_cal: np.ndarray,
    gt_cal: np.ndarray,
    n_classes: int,
    ece_bins: int,
    T_min: float,
    T_max: float,
    K_coarse: int,
    K_refine: int,
    topk: int = 3,
) -> Tuple[float, List[Dict]]:
    """
    Returns:
      T_star
      cal_grid_rows: list of dict rows (T, split='cal', stage='coarse/refine', nll,ece,brier)
    """
    rows: List[Dict] = []

    # 1) coarse grid
    Ts_coarse = logspace_T(T_min, T_max, K_coarse)
    vals = []
    for T in Ts_coarse:
        probs = softmax_T_flat(logits_cal, float(T))
        nll = nll_from_logits_flat(logits_cal, gt_cal, float(T))
        ece = ece_multiclass_confidence(probs, gt_cal, n_bins=ece_bins)
        brier = binary_brier_from_probs(probs, gt_cal, background_class=0)
        rows.append({"split": "cal", "stage": "coarse", "T": float(T), "nll": float(nll), "ece": float(ece), "brier": float(brier)})
        vals.append((float(nll), float(T)))

    vals.sort(key=lambda x: x[0])
    top = [t for (_, t) in vals[:max(1, topk)]]
    top.sort()

    # If only one T or all equal, skip refine robustly
    if len(top) == 1:
        T_star = top[0]
        return T_star, rows

    # 2) refine interval from best 3 Ts (min and max among them)
    T_low = max(T_min, top[0])
    T_high = min(T_max, top[-1])
    if T_high <= T_low:
        # fallback: small neighborhood around best T
        bestT = vals[0][1]
        T_low = max(T_min, bestT / 1.5)
        T_high = min(T_max, bestT * 1.5)
        if T_high <= T_low:
            return bestT, rows

    Ts_refine = logspace_T(T_low, T_high, K_refine)
    for T in Ts_refine:
        probs = softmax_T_flat(logits_cal, float(T))
        nll = nll_from_logits_flat(logits_cal, gt_cal, float(T))
        ece = ece_multiclass_confidence(probs, gt_cal, n_bins=ece_bins)
        brier = binary_brier_from_probs(probs, gt_cal, background_class=0)
        rows.append({"split": "cal", "stage": "refine", "T": float(T), "nll": float(nll), "ece": float(ece), "brier": float(brier)})

    # 3) choose global best among all evaluated rows
    best_row = min((r for r in rows if r["split"] == "cal"), key=lambda r: r["nll"])
    return float(best_row["T"]), rows


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logits_dir", type=str, required=True)
    ap.add_argument("--gt_dir", type=str, required=True)
    ap.add_argument("--dilate_mm", type=float, default=15.0)
    ap.add_argument("--ece_bins", type=int, default=15)
    ap.add_argument("--T_min", type=float, default=0.05)
    ap.add_argument("--T_max", type=float, default=12.0)

    ap.add_argument("--K_coarse", type=int, default=20, help="Number of coarse log-spaced T samples")
    ap.add_argument("--K_refine", type=int, default=25, help="Number of refined log-spaced T samples")
    ap.add_argument("--topk", type=int, default=3, help="Take best top-k coarse Ts to define refine interval")

    ap.add_argument("--ignore_label", type=int, default=None,
                    help="If set, voxels with this GT label are excluded (e.g., -1).")
    ap.add_argument("--out_summary_csv", type=str, default="loo_T_summary.csv")
    ap.add_argument("--out_grid_csv", type=str, default="loo_T_grid.csv")
    # Parallelizzazione: ogni processo tratta come held-out solo i casi
    # case_ids[shard_index::num_shards], ma calibra sempre su TUTTI gli altri casi -> risultati per caso
    # identici alla run unica; i CSV dei blocchi si concatenano dopo.
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_index", type=int, default=0)
    args = ap.parse_args()

    logits_dir = Path(args.logits_dir)
    gt_dir = Path(args.gt_dir)

    logits_files = sorted(logits_dir.glob("*_logits.pt"))
    if len(logits_files) < 2:
        raise SystemExit(f"Found {len(logits_files)} logits files in {logits_dir}. Need at least 2 for LOO.")

    cases: List[Tuple[str, Path, Path]] = []
    for lf in logits_files:
        case_id = parse_case_id_from_logits(lf)
        gt_path = gt_dir / f"{case_id}.nii.gz"
        if not gt_path.exists():
            raise SystemExit(f"Missing GT for case {case_id}: {gt_path}")
        cases.append((case_id, lf, gt_path))

    # Preload and cache ROI-flattened data per case
    cache: Dict[str, Dict] = {}
    n_classes_set = set()

    for case_id, lf, gp in cases:
        gt, spacing = load_gt(gp)  # gt: (H,W,D)
        logits_raw = load_logits_pt(lf)  # (C,D,H,W)

        logits = align_logits_to_gt_transpose_then_resample(logits_raw, gt_shape_hwd=gt.shape)  # (C,H,W,D)
        if logits.shape[1:] != gt.shape:
            raise SystemExit(f"Alignment failed for {case_id}: logits {logits.shape} vs gt {gt.shape}")

        C = logits.shape[0]
        n_classes_set.add(C)

        # prediction at T=1 via argmax of logits (independent of T)
        pred = np.argmax(logits, axis=0).astype(np.int16)  # (H,W,D)

        roi = build_roi_mask(gt, pred, spacing_xyz=spacing, dilate_mm=args.dilate_mm)  # (H,W,D)
        logits_flat = logits_to_flat(logits)  # (N,C)
        gt_flat = gt.reshape(-1).astype(np.int64)
        roi_flat = roi.reshape(-1)

        # ROI select
        logits_roi = logits_flat[roi_flat]
        gt_roi = gt_flat[roi_flat]

        # Optional ignore label filter
        if args.ignore_label is not None:
            keep = (gt_roi != int(args.ignore_label))
            logits_roi = logits_roi[keep]
            gt_roi = gt_roi[keep]

        cache[case_id] = {
            "logits_roi": logits_roi.astype(np.float32, copy=False),
            "gt_roi": gt_roi.astype(np.int64, copy=False),
            "n_vox": int(len(gt_roi)),
        }

    if len(n_classes_set) != 1:
        raise SystemExit(f"Inconsistent number of classes across cases: {n_classes_set}")
    n_classes = int(list(n_classes_set)[0])

    # Sanity check labels range. Empty ROI is expected (not an error) for
    # cases with empty GT AND empty prediction (e.g. TotalSegmentator "noise"
    # cases where the target structure is not in the scan's FOV): the 15mm
    # dilation of GT-union-pred has nothing to dilate, so there is no
    # boundary region to contribute to calibration. Skip these cases (like
    # NaN Dice is excluded from Dice averages elsewhere in the pipeline)
    # instead of aborting the whole run.
    empty_roi_cases = []
    for cid in list(cache.keys()):
        g = cache[cid]["gt_roi"]
        if g.size == 0:
            empty_roi_cases.append(cid)
            del cache[cid]
            continue
        if g.min() < 0 or g.max() >= n_classes:
            raise SystemExit(f"Bad GT labels in {cid}: min={int(g.min())} max={int(g.max())} n_classes={n_classes}")

    if empty_roi_cases:
        print(f"Skipping {len(empty_roi_cases)} case(s) with empty ROI (empty GT + empty prediction, "
              f"no boundary region): {sorted(empty_roi_cases)}")

    case_ids = list(cache.keys())
    if len(case_ids) < 2:
        raise SystemExit(f"Only {len(case_ids)} case(s) left with non-empty ROI. Need at least 2 for LOO.")

    summary_rows: List[Dict] = []
    grid_rows: List[Dict] = []

    held_ids = case_ids[args.shard_index::args.num_shards]
    print(f"[shard {args.shard_index}/{args.num_shards}] {len(held_ids)} held-out su {len(case_ids)} casi")
    for held in held_ids:
        cal_ids = [cid for cid in case_ids if cid != held]
        logits_cal = np.concatenate([cache[cid]["logits_roi"] for cid in cal_ids], axis=0)
        gt_cal = np.concatenate([cache[cid]["gt_roi"] for cid in cal_ids], axis=0)

        # Find T* using coarse+refine on CAL set
        T_star, cal_rows = select_T_star_by_coarse_then_refine(
            logits_cal=logits_cal,
            gt_cal=gt_cal,
            n_classes=n_classes,
            ece_bins=args.ece_bins,
            T_min=args.T_min,
            T_max=args.T_max,
            K_coarse=args.K_coarse,
            K_refine=args.K_refine,
            topk=args.topk,
        )

        # Add calibration-grid rows to global grid CSV (tag with heldout id)
        for r in cal_rows:
            grid_rows.append({
                "heldout_case": held,
                "split": r["split"],        # cal
                "stage": r["stage"],        # coarse/refine
                "T": r["T"],
                "nll": r["nll"],
                "ece": r["ece"],
                "brier": r["brier"],
                "T_star_for_case": float(T_star),
                "n_vox": int(len(gt_cal)),
            })

        # Evaluate on held-out at T*
        logits_eval = cache[held]["logits_roi"]
        gt_eval = cache[held]["gt_roi"]

        probs_eval = softmax_T_flat(logits_eval, T_star)
        nll_eval = nll_from_logits_flat(logits_eval, gt_eval, T_star)
        ece_eval = ece_multiclass_confidence(probs_eval, gt_eval, n_bins=args.ece_bins)
        brier_eval = binary_brier_from_probs(probs_eval, gt_eval, background_class=0)

        summary_rows.append({
            "heldout_case": held,
            "T_star": float(T_star),
            "nll_eval": float(nll_eval),
            "ece_eval": float(ece_eval),
            "brier_eval": float(brier_eval),
            "n_vox_eval": int(len(gt_eval)),
            "n_vox_cal": int(len(gt_cal)),
        })

        # Also write eval-grid metrics for the SAME Ts we evaluated in cal (coarse+refine)
        # (useful to compare cal-vs-eval behavior across Ts)
        tried_Ts = [r["T"] for r in cal_rows]  # may include duplicates; de-dup
        tried_Ts = sorted(set(float(t) for t in tried_Ts))

        for T in tried_Ts:
            probs = softmax_T_flat(logits_eval, T)
            nll = nll_from_logits_flat(logits_eval, gt_eval, T)
            ece = ece_multiclass_confidence(probs, gt_eval, n_bins=args.ece_bins)
            brier = binary_brier_from_probs(probs, gt_eval, background_class=0)
            grid_rows.append({
                "heldout_case": held,
                "split": "eval",
                "stage": "coarse+refine",
                "T": float(T),
                "nll": float(nll),
                "ece": float(ece),
                "brier": float(brier),
                "T_star_for_case": float(T_star),
                "n_vox": int(len(gt_eval)),
            })

        print(f"[LOO] held-out={held}  T*={T_star:.4f}  "
              f"NLL={nll_eval:.4f}  ECE={ece_eval:.4f}  Brier={brier_eval:.4f}  "
              f"vox(cal)={len(gt_cal)} vox(eval)={len(gt_eval)}")

    # Write CSVs
    out_summary = Path(args.out_summary_csv)
    with out_summary.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    out_grid = Path(args.out_grid_csv)
    with out_grid.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(grid_rows[0].keys()))
        w.writeheader()
        w.writerows(grid_rows)

    Ts = np.array([r["T_star"] for r in summary_rows], dtype=np.float64)
    print("\nSummary:")
    print(f"  T*: mean={Ts.mean():.4f}  std={Ts.std(ddof=1) if len(Ts)>1 else 0.0:.4f}  "
          f"min={Ts.min():.4f}  max={Ts.max():.4f}")
    print(f"  Saved summary: {out_summary}")
    print(f"  Saved grid:    {out_grid}")


if __name__ == "__main__":
    main()
