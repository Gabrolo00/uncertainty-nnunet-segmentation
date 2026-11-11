"""
Calibrazione leave-one-out della soglia di ogni mappa di incertezza: tau* che massimizza il Dice tra mappa
binarizzata ed errori di segmentazione.

Uso: python Calibrate_threshold_loo_v2.py --prediction_dir <pred> --gt_dir <label> --ct_dir <ct> --uncertainty_dir <mappe> --uncertainty_type entropy --output_csv <out.csv> --qa_dir <dir>
"""
from __future__ import annotations

import os
import csv
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import nibabel as nib
from nibabel.orientations import aff2axcodes
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
import time


# ----------------------------
# Data structures
# ----------------------------

@dataclass
class CasePaths:
    """Container for all per-case file paths."""
    case_id: str
    pred_path: str
    gt_path: str
    ct_path: str
    unc_path: str


@dataclass
class AlignmentReport:
    """Basic alignment report used for fail-fast checks."""
    ok: bool
    reason: str
    ref_shape: Tuple[int, int, int]
    pred_shape: Tuple[int, int, int]
    gt_shape: Tuple[int, int, int]
    unc_shape: Tuple[int, int, int]
    ref_axcodes: Tuple[str, str, str]
    pred_axcodes: Tuple[str, str, str]
    gt_axcodes: Tuple[str, str, str]
    unc_axcodes: Tuple[str, str, str]


# ----------------------------
# Path resolution
# ----------------------------

def _basename_without_nii_gz(filename: str) -> str:
    """Return basename without .nii.gz suffix."""
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    return os.path.splitext(filename)[0]


def list_case_ids_from_gt(gt_dir: str) -> List[str]:
    """List case IDs based on GT filenames (AUTOMI_00006.nii.gz -> AUTOMI_00006)."""
    case_ids = []
    for fn in os.listdir(gt_dir):
        if fn.endswith(".nii.gz"):
            case_ids.append(_basename_without_nii_gz(fn))
    case_ids = sorted(set(case_ids))
    return case_ids


def build_case_paths(
    case_id: str,
    prediction_dir: str,
    gt_dir: str,
    ct_dir: str,
    uncertainty_dir: str,
    uncertainty_type: str,
) -> CasePaths:
    """
    Resolve file paths based on case_id and uncertainty_type.
    Assumptions from user:
      - prediction binary at PREDICTION_DIR/{case_id}.nii.gz
      - gt at GT_DIR/{case_id}.nii.gz
      - ct at TC_DIR/{case_id}_0000.nii.gz
      - uncertainty at UNCERTAINTY_DIR/{case_id}_{uncertainty_type}.nii.gz
    """
    pred_path = os.path.join(prediction_dir, f"{case_id}.nii.gz")
    gt_path = os.path.join(gt_dir, f"{case_id}.nii.gz")
    ct_path = os.path.join(ct_dir, f"{case_id}_0000.nii.gz")
    unc_path = os.path.join(uncertainty_dir, f"{case_id}_{uncertainty_type}.nii.gz")

    for p, name in [(pred_path, "PRED"), (gt_path, "GT"), (ct_path, "CT"), (unc_path, "UNC")]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing {name} file for case_id={case_id}: {p}")

    return CasePaths(case_id=case_id, pred_path=pred_path, gt_path=gt_path, ct_path=ct_path, unc_path=unc_path)


# ----------------------------
# Loading helpers
# ----------------------------

def to_uncertainty_map(unc_data: np.ndarray, uncertainty_type: str) -> np.ndarray:
    """
    Convert any input map into an 'uncertainty' map where higher = more uncertain.
    """
    # Confidence: higher means more confident => invert
    if uncertainty_type == "confidence_map":
        print("[DEBUG] Converting confidence_map to uncertainty_map by inversion.")
        return float(np.max(unc_data)) - unc_data


    return unc_data


def load_nifti(path: str) -> Tuple[np.ndarray, np.ndarray, nib.Nifti1Image]:
    """
    Load a NIfTI and return (data, affine, nib_image).
    Data is returned as a numpy array in the stored voxel order.
    """
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    affine = img.affine
    return data, affine, img


def get_voxel_sizes_mm(nib_img: nib.Nifti1Image) -> Tuple[float, float, float]:
    """Return voxel sizes (mm) from header zooms."""
    zooms = nib_img.header.get_zooms()
    if len(zooms) < 3:
        raise ValueError("NIfTI header has < 3 zoom values.")
    return float(zooms[0]), float(zooms[1]), float(zooms[2])


# ----------------------------
# Alignment checks (fail-fast)
# ----------------------------

def check_alignment_fail_fast(
    ct_img: nib.Nifti1Image,
    pred_img: nib.Nifti1Image,
    gt_img: nib.Nifti1Image,
    unc_img: nib.Nifti1Image,
    affine_tol: float = 1e-3,
) -> AlignmentReport:
    """
    Fail-fast alignment check.
    Checks:
      - shapes identical
      - axcodes identical (orientation)
      - affine close (within affine_tol)
    Uses nibabel aff2axcodes for orientation sanity. [web:67]
    """
    ref_shape = ct_img.shape[:3]
    pred_shape = pred_img.shape[:3]
    gt_shape = gt_img.shape[:3]
    unc_shape = unc_img.shape[:3]

    ref_ax = aff2axcodes(ct_img.affine)
    pred_ax = aff2axcodes(pred_img.affine)
    gt_ax = aff2axcodes(gt_img.affine)
    unc_ax = aff2axcodes(unc_img.affine)

    if pred_shape != ref_shape or gt_shape != ref_shape or unc_shape != ref_shape:
        return AlignmentReport(
            ok=False,
            reason="Shape mismatch",
            ref_shape=ref_shape,
            pred_shape=pred_shape,
            gt_shape=gt_shape,
            unc_shape=unc_shape,
            ref_axcodes=ref_ax,
            pred_axcodes=pred_ax,
            gt_axcodes=gt_ax,
            unc_axcodes=unc_ax,
        )

    if pred_ax != ref_ax or gt_ax != ref_ax or unc_ax != ref_ax:
        return AlignmentReport(
            ok=False,
            reason="Orientation axcodes mismatch",
            ref_shape=ref_shape,
            pred_shape=pred_shape,
            gt_shape=gt_shape,
            unc_shape=unc_shape,
            ref_axcodes=ref_ax,
            pred_axcodes=pred_ax,
            gt_axcodes=gt_ax,
            unc_axcodes=unc_ax,
        )

    # Affine closeness check (covers origin/direction differences).
    def _affine_close(a: np.ndarray, b: np.ndarray) -> bool:
        return np.allclose(a, b, atol=affine_tol, rtol=0)

    if not _affine_close(pred_img.affine, ct_img.affine):
        return AlignmentReport(False, "Mismatch", ref_shape, pred_shape, gt_shape, unc_shape, ref_ax, pred_ax, gt_ax, unc_ax)  # will be overwritten below

    if not _affine_close(gt_img.affine, ct_img.affine):
        return AlignmentReport(False, "Mismatch", ref_shape, pred_shape, gt_shape, unc_shape, ref_ax, pred_ax, gt_ax, unc_ax)

    if not _affine_close(unc_img.affine, ct_img.affine):
        return AlignmentReport(False, "Mismatch", ref_shape, pred_shape, gt_shape, unc_shape, ref_ax, pred_ax, gt_ax, unc_ax)

    return AlignmentReport(
        ok=True,
        reason="OK",
        ref_shape=ref_shape,
        pred_shape=pred_shape,
        gt_shape=gt_shape,
        unc_shape=unc_shape,
        ref_axcodes=ref_ax,
        pred_axcodes=pred_ax,
        gt_axcodes=gt_ax,
        unc_axcodes=unc_ax,
    )


# ----------------------------
# ROI: buffer within 15 mm
# ----------------------------

def build_roi_buffer_15mm(ref_mask: np.ndarray, voxel_sizes_mm: Tuple[float, float, float], radius_mm: float = 15.0) -> np.ndarray:
    """Build an ROI mask as the set of voxels within ``radius_mm`` of ``ref_mask`` foreground.

    Notes:
    - ``ref_mask`` can be GT-only, Pred-only, or a union (GT ∪ Pred).
    - Uses ``distance_transform_edt`` with ``sampling = voxel_sizes_mm`` (in mm).
    """
    ref = (ref_mask > 0.5).astype(np.uint8)

    # distance_transform_edt computes distance to the nearest zero in the input.
    # So we invert: foreground becomes 0, background becomes 1, then distance gives mm to foreground.
    inv = (ref == 0).astype(np.uint8)
    dist_mm = distance_transform_edt(inv, sampling=list(voxel_sizes_mm))
    roi = dist_mm <= float(radius_mm)
    return roi.astype(np.uint8)



# ----------------------------
# Uncertainty normalization (full volume min-max -> [0,1])
# ----------------------------

def normalize_minmax_full(U: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, float, float]:
    """Min-max normalization over the full volume to [0,1]."""
    umin = float(np.min(U))
    umax = float(np.max(U))
    if abs(umax - umin) < eps:
        return np.zeros_like(U, dtype=np.float32), umin, umax
    U_norm = (U - umin) / (umax - umin)
    U_norm = np.clip(U_norm, 0.0, 1.0).astype(np.float32)
    return U_norm, umin, umax


# ----------------------------
# QA overlay (always saved)
# ----------------------------

def _choose_axial_slice_center_of_gt(gt: np.ndarray) -> int:
    """Pick axial slice index at the center of GT foreground; fallback to mid-slice."""
    coords = np.argwhere(gt > 0.5)
    if coords.size == 0:
        return gt.shape[2] // 2
    z_mean = int(round(coords[:, 2].mean()))
    return int(np.clip(z_mean, 0, gt.shape[2] - 1))


def save_qa_png_axial(
    out_png_path: str,
    ct: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray,
    roi: np.ndarray,
    U_norm: np.ndarray,
    case_id: str,
    uncertainty_type: str,
) -> None:
    """
    Save a single axial PNG overlay:
      - CT grayscale
      - uncertainty heatmap (alpha)
      - GT contour (red)
      - Pred contour (blue)
      - ROI contour (yellow)
    """
    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)

    z = _choose_axial_slice_center_of_gt(gt)

    ct_sl = ct[:, :, z]
    gt_sl = gt[:, :, z] > 0.5
    pred_sl = pred[:, :, z] > 0.5
    roi_sl = roi[:, :, z] > 0.5
    u_sl = U_norm[:, :, z]

    # Basic windowing for CT display (robust-ish):
    # Use percentiles to avoid extreme outliers in visualization.
    vmin = np.percentile(ct_sl, 1)
    vmax = np.percentile(ct_sl, 99)

    plt.figure(figsize=(8, 8), dpi=150)
    plt.imshow(ct_sl.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    plt.imshow(u_sl.T, cmap="jet", origin="lower", alpha=0.35, vmin=0.0, vmax=1.0)

    # Contours (transpose to match displayed orientation)
    # Using simple contour plotting on boolean masks.
    plt.contour(roi_sl.T.astype(np.uint8), levels=[0.5], colors="yellow", linewidths=0.6)
    plt.contour(gt_sl.T.astype(np.uint8), levels=[0.5], colors="red", linewidths=1.0)
    plt.contour(pred_sl.T.astype(np.uint8), levels=[0.5], colors="cyan", linewidths=1.0)

    plt.title(f"{case_id} | {uncertainty_type} | axial z={z}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()


# ----------------------------
# Confusion masks + coverage + scoring
# ----------------------------

def confusion_masks(gt: np.ndarray, pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return TP, FP, FN, TN boolean masks."""
    gt_b = gt > 0.5
    pred_b = pred > 0.5
    tp = pred_b & gt_b
    fp = pred_b & (~gt_b)
    fn = (~pred_b) & gt_b
    tn = (~pred_b) & (~gt_b)
    return tp, fp, fn, tn


def coverage(U_norm: np.ndarray, class_mask: np.ndarray, roi: np.ndarray, t: float, ge_is_uncertain: bool = True) -> float:
    """
    Coverage = fraction of class voxels (inside ROI) flagged as uncertain.
    """
    M = class_mask & (roi > 0)
    denom = int(M.sum())
    if denom == 0:
        return float("nan")
    if ge_is_uncertain:
        flagged = (U_norm >= t) & M
    else:
        flagged = (U_norm <= t) & M
    return float(flagged.sum() / denom)

def ueo_dice(err: np.ndarray, unc: np.ndarray) -> float:
    """
    Uncertainty–Error Overlap (UEO) as Dice overlap between:
      - err: error voxels (FP or FN) inside ROI
      - unc: uncertain voxels inside ROI
    """
    a = int((err & unc).sum())
    b = int(err.sum())
    c = int(unc.sum())
    denom = (b + c)
    if denom == 0:
        return float("nan")
    return float(2.0 * a / denom)



def score_from_coverages(cov_fn: float, cov_fp: float, cov_tp: float, cov_tn: float, w: Dict[str, float]) -> float:
    """
    Score using coverages (fractions).
    """
    score = (w["wFN"] * cov_fn) + (w["wFP"] * cov_fp) - (w["wTP"] * cov_tp) - (w["wTN"] * cov_tn)
    return score

# ----------------------------
# Grid search (coarse-to-fine on [0,1])
# ----------------------------

def build_coarse_grid_0_1(step: float = 0.1) -> List[float]:
    """Create coarse grid in [0,1] inclusive."""
    n = int(round(1.0 / step))
    grid = [round(i * step, 10) for i in range(0, n + 1)]
    if grid[-1] != 1.0:
        grid.append(1.0)
    return sorted(set(grid))


def refine_grid_between_best_points(best_points: List[float], fine_step: float) -> List[float]:
    """
    Refine thresholds only inside [min(best_points), max(best_points)].
    """
    left = float(min(best_points))
    right = float(max(best_points))

    left = max(0.0, left)
    right = min(1.0, right)

    if right < left:
        left, right = right, left

    n = int(math.floor((right - left) / fine_step))
    grid = [left + i * fine_step for i in range(n + 1)]
    grid.append(right)

    grid = sorted(set([round(float(np.clip(x, 0.0, 1.0)), 10) for x in grid]))
    return grid



def find_best_threshold_coarse_to_fine(
    cases,
    weights,
    ge_is_uncertain,
    roi_mode,
    uncertainty_type,
    coarse_step=0.1,
    fine_step=0.01,
    top_k=3,
    case_indices=None,
    case_ueo=None,
):
    """
    Deterministic coarse-to-fine search.

    With --cache_case_scores: if `case_ueo` is given, the per-case UEO at threshold t comes from a cache
    shared by all LOO folds (case_ueo(idx, t) -> float, or None if undefined) instead of reloading and recomputing
    every training case in every fold; `case_indices` lists the training cases in the same order as `cases`, so the
    mean is taken over the same values in the same order (identical results).

    New process:
    1) Evaluate coarse grid: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
       - Take top point t1.
    2) Evaluate t2=t1-0.1 (clip at 0) and t3=t1+0.1 (clip at 1).
       - Take the top2 among {t2, t1, t3} => (ta, tb) and require adjacency in that local grid.
    3) Evaluate refine grid: [ta, ta+0.02, ta+0.04, ta+0.06, ta+0.08, tb]
       - Take top point m1 inside this refine grid.
       - If m1 is at the border (ta or tb), shift the zoom by +/-0.02 repeatedly (max 10)
         until the best is not at the border or we hit [0,1].
    4) Evaluate m2=m1-0.01 (clip at 0) and m3=m1+0.01 (clip at 1).
       - Return best among {m2, m1, m3}.

    Notes:
    - Requires existing helpers: load_nifti, get_voxel_sizes_mm, build_roi_buffer_15mm,
      normalize_minmax_full, confusion_masks, score_from_coverages, to_uncertainty_map.
    """

    # ----------------------------
    # Preload: build `pre` to speed up mean_score_at_t (coverage mode)
    # ----------------------------
    pre = []
    for c in (cases if case_ueo is None else []):
        ct, _, ct_img = load_nifti(c.ct_path)
        gt, _, gt_img = load_nifti(c.gt_path)
        pred, _, pred_img = load_nifti(c.pred_path)
        U, _, unc_img = load_nifti(c.unc_path)

        voxel_sizes = get_voxel_sizes_mm(ct_img)

        if roi_mode == "full":
            roi = np.ones(gt.shape, dtype=np.uint8)
        elif roi_mode == "buffer_15mm":
            # Solution A: ROI buffer around the union of GT and prediction.
            # This includes regions where anatomy exists (GT) and where the model predicts anatomy (Pred).
            union_mask = ((gt > 0.5) | (pred > 0.5)).astype(np.uint8)
            roi = build_roi_buffer_15mm(union_mask, voxel_sizes_mm=voxel_sizes, radius_mm=15.0)

        else:
            raise ValueError(f"Unknown roi_mode: {roi_mode}")

        # Ensure 'higher = more uncertain'
        U = to_uncertainty_map(U, uncertainty_type)

        # Normalize to [0,1] (full-volume)
        U_norm, _, _ = normalize_minmax_full(U)
        U_norm = U_norm.astype(np.float32)

        tp, fp, fn, tn = confusion_masks(gt, pred)
        roi_b = (roi > 0)

        mFN = fn & roi_b
        mFP = fp & roi_b
        mERR = (mFN | mFP)  # errors inside ROI
        mROI = roi_b  # ROI mask (bool)

        pre.append((U_norm, mERR, mROI))

    # ----------------------------
    # Mean score function (coverage, fast; no calls to `coverage()`)
    # ----------------------------
    def mean_score_at_t(t: float) -> float:
        scores = []
        print(f"[DEBUG] Evaluating mean_score_at_t for t={t:.4f}")

        if case_ueo is not None:
            for idx in case_indices:
                s = case_ueo(idx, t)
                if s is not None:
                    scores.append(s)
        for U_norm, mERR, mROI in pre:
            if ge_is_uncertain:
                unc = (U_norm >= t) & mROI
            else:
                unc = (U_norm <= t) & mROI

            inter = int((mERR & unc).sum())
            den = int(mERR.sum()) + int(unc.sum())

            # If both are empty, UEO is undefined -> skip this case
            if den == 0:
                continue

            ueo = float(2.0 * inter / den)  # Dice overlap [0,1]
            scores.append(ueo)

        if not scores:
            return float("-inf")

        mean_s = float(np.mean(scores))
        print(f"[DEBUG] Mean UEO at t={t:.4f}: {mean_s}")
        return mean_s

    # ----------------------------
    # Cached evaluation wrapper
    # ----------------------------
    evals = {}

    def _rt(t: float) -> float:
        # Round to prevent float-key duplication issues.
        return float(round(t, 10))

    def _clip01(t: float) -> float:
        return float(min(1.0, max(0.0, t)))

    def f(t: float) -> float:
        tt = _rt(_clip01(t))
        if tt not in evals:
            evals[tt] = mean_score_at_t(tt)
        return evals[tt]

    def _top1(points):
        """Return the best point by score."""
        pts = sorted(_rt(_clip01(p)) for p in points)
        scored = [(p, f(p)) for p in pts]
        return max(scored, key=lambda x: x[1])[0]

    def _top2_adjacent(points):
        """
        Return (a,b) = top2 points by score, requiring adjacency in the sorted list.
        If top2 is not adjacent to top1, fall back to the best adjacent neighbor of top1.
        """
        pts = sorted(_rt(_clip01(p)) for p in points)
        scored = [(p, f(p)) for p in pts]
        scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)

        a = scored_sorted[0][0]
        b = scored_sorted[1][0]

        ia = pts.index(a)
        ib = pts.index(b)

        if abs(ia - ib) != 1:
            # top2 is not adjacent to top1: pick the best neighbor of top1
            score_map = dict(scored)
            neighbors = []
            if ia > 0:
                neighbors.append(pts[ia - 1])
            if ia < len(pts) - 1:
                neighbors.append(pts[ia + 1])
            b = max(neighbors, key=lambda x: score_map[x])
            ib = pts.index(b)
            print(f"[DEBUG] top2 not adjacent to top1={a}, fallback to best neighbor b={b}")

        return (a, b) if ia < ib else (b, a)

    def _make_refine_grid(ta: float, tb: float, step: float = 0.02):
        """
        Build [ta, ta+step, ..., ta+0.08, tb] and keep only points within [ta,tb].
        Assumes tb >= ta.
        """
        ta = _rt(ta)
        tb = _rt(tb)

        pts = [ta + k * step for k in range(0, 5)] + [tb]
        pts = [_rt(_clip01(p)) for p in pts if p >= ta - 1e-12 and p <= tb + 1e-12]
        return sorted(set(pts))

    # ----------------------------
    # 1) Coarse grid: pick top1
    # ----------------------------
    coarse_grid = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for t in coarse_grid:
        f(t)

    t1 = _top1(coarse_grid)
    print(f"[DEBUG] Coarse grid best t1={t1:.4f}")

    # ----------------------------
    # 2) Local bracket around t1: t1-0.1, t1, t1+0.1 (clip)
    #    Pick top2 among them and require adjacency
    # ----------------------------
    t2 = _rt(_clip01(t1 - 0.1))
    t3 = _rt(_clip01(t1 + 0.1))
    for t in [t2, t1, t3]:
        f(t)

    local_grid = sorted(set([t2, _rt(t1), t3]))
    # If local grid collapses to <3 points at borders, adjacency still makes sense on the sorted list.
    ta, tb = _top2_adjacent(local_grid)
    print(f"[DEBUG] Local bracket best (ta,tb)=({ta:.4f}, {tb:.4f})")

    # ----------------------------
    # 3) Refine grid inside [ta,tb], then "border-walk" if max is at border
    # ----------------------------
    step_refine = 0.02
    max_shifts = 10

    # Ensure ordered
    ta = float(ta)
    tb = float(tb)
    if tb < ta:
        ta, tb = tb, ta

    # Border-walk loop:
    # - Evaluate refine grid
    # - Take best point m1
    # - If m1 is at ta, shift interval left by step_refine
    # - If m1 is at tb, shift interval right by step_refine
    # - Stop when m1 is internal, or boundaries hit 0/1, or max_shifts exceeded
    m1 = None
    for _ in range(max_shifts):
        refine_grid = _make_refine_grid(ta, tb, step=step_refine)
        for t in refine_grid:
            f(t)

        m1 = _top1(refine_grid)

        # If best is at left border, shift left
        if abs(m1 - ta) < 1e-12:
            if ta <= 0.0 + 1e-12:
                break  # cannot shift left
            ta = _rt(_clip01(ta - step_refine))
            tb = _rt(_clip01(tb - step_refine))
            continue

        # If best is at right border, shift right
        if abs(m1 - tb) < 1e-12:
            if tb >= 1.0 - 1e-12:
                break  # cannot shift right
            ta = _rt(_clip01(ta + step_refine))
            tb = _rt(_clip01(tb + step_refine))
            continue

        # Internal maximum found
        break

    if m1 is None:
        # Fallback: should never happen, but keep safe
        m1 = t1

    # ----------------------------
    # 4) Local check around m1 with +/- 0.01
    # ----------------------------
    m2 = _rt(_clip01(m1 - 0.01))
    m3 = _rt(_clip01(m1 + 0.01))
    for t in [m2, m1, m3]:
        f(t)

    final_candidates = sorted(set([m2, _rt(m1), m3]))
    best_t, best_s = max(((t, f(t)) for t in final_candidates), key=lambda x: x[1])

    return float(best_t), float(best_s)




# ----------------------------
# CSV append helpers
# ----------------------------
def ensure_trailing_newline(file_path: str) -> None:
    """
    Ensure the file ends with a newline so that CSV appends start on a new row.
    """
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return

    with open(file_path, "rb") as f:
        f.seek(-1, os.SEEK_END)
        last_char = f.read(1)

    if last_char != b"\n":
        with open(file_path, "ab") as f:
            f.write(b"\n")


def append_rows_to_csv(csv_path: str, rows: list[dict]) -> None:
    """
    Append rows to CSV. If missing, create and write header.
    Supports both ',' and ';' delimiters (auto-detected from existing header).
    """
    import os
    import csv

    if not rows:
        return

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    fieldnames = list(rows[0].keys())
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0

    # Default delimiter for new files
    delimiter = ","

    if file_exists:
        # Detect delimiter from first line
        with open(csv_path, "r", newline="") as f:
            first_line = f.readline().strip()
        delimiter = ";" if ";" in first_line and "," not in first_line else ","

        # Read header with detected delimiter
        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f, delimiter=delimiter)
            header = next(reader, None)

        if header is None:
            raise ValueError("CSV exists but has no header.")

        if header != fieldnames:
            raise ValueError(
                "CSV header mismatch.\n"
                f"Existing: {header}\n"
                f"New:      {fieldnames}\n"
                "Refuse to append to avoid mixed schemas."
            )

        # Append rows using same delimiter
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
            ensure_trailing_newline(csv_path)
            writer.writerows(rows)

    else:
        # Create new CSV
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
            writer.writeheader()
            writer.writerows(rows)



# ----------------------------
# Main runner: LOO calibration
# ----------------------------

def calibrate_threshold_loo_nibabel(
    prediction_dir: str,
    gt_dir: str,
    ct_dir: str,
    uncertainty_dir: str,
    uncertainty_type: str,
    output_csv_path: str,
    qa_output_dir: str,
    roi_mode: str = "buffer_15mm",
    ge_is_uncertain: bool = True,
    weights: Optional[Dict[str, float]] = None,
    coarse_step: float = 0.1,
    fine_step: float = 0.01,
    top_k: int = 3,
    affine_tol: float = 1e-3,
    cache_case_scores: bool = False,
) -> float:
    """
    End-to-end LOO calibration.
    Returns the final threshold = median of per-fold optimal thresholds.

    Notes:
      - Always generates QA PNG per case (axial slice).
      - Stops immediately on any alignment mismatch.
    """
    if weights is None:
        # Default bias: prioritize catching FN more than FP (change as needed).
        weights = {"wFN": 2.0, "wFP": 1.0, "wTP": 1.0, "wTN": 1.0}

    case_ids = list_case_ids_from_gt(gt_dir)
    t0 = time.perf_counter()
    print(f"[INFO] Found {len(case_ids)} cases in GT_DIR.")
    print(f"[INFO] uncertainty_type = {uncertainty_type}")
    print(f"[INFO] roi_mode = {roi_mode}, grid coarse_step={coarse_step}, fine_step={fine_step}, top_k={top_k}")
    print(f"[INFO] output_csv = {output_csv_path}")
    print(f"[INFO] qa_output_dir = {qa_output_dir}")

    if len(case_ids) < 2:
        raise ValueError("Need at least 2 cases for LOO calibration.")

    os.makedirs(qa_output_dir, exist_ok=True)

    # Resolve all paths upfront
    cases = [
        build_case_paths(
            case_id=cid,
            prediction_dir=prediction_dir,
            gt_dir=gt_dir,
            ct_dir=ct_dir,
            uncertainty_dir=uncertainty_dir,
            uncertainty_type=uncertainty_type,
        )
        for cid in case_ids
    ]

    loo_rows: List[Dict[str, object]] = []
    loo_thresholds: List[float] = []

    # Pre-run: for each case, always generate QA and check alignment (fail-fast).
    # This also ensures you do not start LOO with broken inputs.
    for c in cases:
        ct, _, ct_img = load_nifti(c.ct_path)
        gt, _, gt_img = load_nifti(c.gt_path)
        pred, _, pred_img = load_nifti(c.pred_path)
        U, _, unc_img = load_nifti(c.unc_path)

        voxel_sizes = get_voxel_sizes_mm(ct_img)
        if roi_mode == "full":
            roi = np.ones(gt.shape, dtype=np.uint8)
        else:
            # Solution A: ROI buffer around the union of GT and prediction.
            union_mask = ((gt > 0.5) | (pred > 0.5)).astype(np.uint8)
            roi = build_roi_buffer_15mm(union_mask, voxel_sizes, 15.0)

        U_norm, umin, umax = normalize_minmax_full(U)

        qa_png_path = os.path.join(qa_output_dir, f"{c.case_id}_{uncertainty_type}_qa.png")
        save_qa_png_axial(
            out_png_path=qa_png_path,
            ct=ct,
            pred=pred,
            gt=gt,
            roi=roi,
            U_norm=U_norm,
            case_id=c.case_id,
            uncertainty_type=uncertainty_type,
        )

        rep = check_alignment_fail_fast(ct_img, pred_img, gt_img, unc_img, affine_tol=affine_tol)
        if not rep.ok:
            raise RuntimeError(
                f"Alignment mismatch (fail-fast) for case_id={c.case_id}.\n"
                f"Reason: {rep.reason}\n"
                f"Ref shape: {rep.ref_shape}, pred: {rep.pred_shape}, gt: {rep.gt_shape}, unc: {rep.unc_shape}\n"
                f"Axcodes ref: {rep.ref_axcodes}, pred: {rep.pred_axcodes}, gt: {rep.gt_axcodes}, unc: {rep.unc_axcodes}\n"
                f"QA saved at: {qa_png_path}"
            )

    # /10 (--cache_case_scores): load each case once (same steps as the preload in
    # find_best_threshold_coarse_to_fine) and cache its UEO per threshold, shared by all folds.
    case_ueo = None
    if cache_case_scores:
        pre_all = []
        for c in cases:
            ct, _, ct_img = load_nifti(c.ct_path)
            gt, _, gt_img = load_nifti(c.gt_path)
            pred, _, pred_img = load_nifti(c.pred_path)
            U, _, unc_img = load_nifti(c.unc_path)
            voxel_sizes = get_voxel_sizes_mm(ct_img)
            if roi_mode == "full":
                roi = np.ones(gt.shape, dtype=np.uint8)
            else:
                union_mask = ((gt > 0.5) | (pred > 0.5)).astype(np.uint8)
                roi = build_roi_buffer_15mm(union_mask, voxel_sizes_mm=voxel_sizes, radius_mm=15.0)
            U = to_uncertainty_map(U, uncertainty_type)
            U_norm, _, _ = normalize_minmax_full(U)
            U_norm = U_norm.astype(np.float32)
            tp, fp, fn, tn = confusion_masks(gt, pred)
            roi_b = (roi > 0)
            pre_all.append((U_norm, (fn & roi_b) | (fp & roi_b), roi_b))
            del ct, gt, pred, U, roi
        ueo_cache: Dict[Tuple[int, float], Optional[float]] = {}

        def case_ueo(idx: int, t: float) -> Optional[float]:
            key = (idx, t)
            if key not in ueo_cache:
                U_norm, mERR, mROI = pre_all[idx]
                unc = ((U_norm >= t) if ge_is_uncertain else (U_norm <= t)) & mROI
                inter = int((mERR & unc).sum())
                den = int(mERR.sum()) + int(unc.sum())
                ueo_cache[key] = None if den == 0 else float(2.0 * inter / den)
            return ueo_cache[key]

    # LOO loop
    print("[INFO] Starting LOO calibration.")
    for i, test_case in enumerate(cases):
        print(f"[INFO] LOO fold {i + 1}/{len(cases)} | test_case={test_case.case_id}")
        train_cases = [c for j, c in enumerate(cases) if j != i]

        print(f"[INFO] Searching best threshold on {len(train_cases)} training cases ...")
        t_star, train_score = find_best_threshold_coarse_to_fine(
            cases=train_cases,
            weights=weights,
            ge_is_uncertain=ge_is_uncertain,
            roi_mode=roi_mode,
            uncertainty_type=uncertainty_type,
            coarse_step=coarse_step,
            fine_step=fine_step,
            top_k=top_k,
            case_indices=[j for j in range(len(cases)) if j != i],
            case_ueo=case_ueo,
        )
        loo_thresholds.append(t_star)

        print(f"[INFO] Best t_star={t_star:.4f}, train_score={train_score:.6f}")

        # Evaluate on held-out test_case (single-case metrics at t_star)
        ct, _, ct_img = load_nifti(test_case.ct_path)
        gt, _, gt_img = load_nifti(test_case.gt_path)
        pred, _, pred_img = load_nifti(test_case.pred_path)
        U, _, unc_img = load_nifti(test_case.unc_path)

        voxel_sizes = get_voxel_sizes_mm(ct_img)
        if roi_mode == "full":
            roi = np.ones(gt.shape, dtype=np.uint8)
        else:
            # Solution A: ROI buffer around the union of GT and prediction.
            union_mask = ((gt > 0.5) | (pred > 0.5)).astype(np.uint8)
            roi = build_roi_buffer_15mm(union_mask, voxel_sizes, 15.0)
            
        U = to_uncertainty_map(U, uncertainty_type)
        U_norm, umin, umax = normalize_minmax_full(U)

        tp, fp, fn, tn = confusion_masks(gt, pred)

        roi_b = (roi > 0)

        # Error voxels inside ROI (FP or FN)
        err = (fp | fn) & roi_b

        # Uncertain voxels inside ROI at t_star
        if ge_is_uncertain:
            unc = (U_norm >= t_star) & roi_b
        else:
            unc = (U_norm <= t_star) & roi_b

        # UEO (Dice overlap) in [0,1]
        intersection = int((err & unc).sum())
        den = int(err.sum()) + int(unc.sum())
        test_score = float("nan") if den == 0 else float(2.0 * intersection / den)

        qa_png_path = os.path.join(qa_output_dir, f"{test_case.case_id}_{uncertainty_type}_qa.png")

        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "uncertainty_type": uncertainty_type,
            "case_id": test_case.case_id,
            "roi_mode": roi_mode,
            "ge_is_uncertain": ge_is_uncertain,
            "coarse_step": coarse_step,
            "fine_step": fine_step,
            "top_k": top_k,
            "t_star_train": t_star,
            "train_score_mean": train_score,

            "score_test": test_score,

            "unc_min_full": umin,
            "unc_max_full": umax,
            "qa_png_path": qa_png_path,
        }
        loo_rows.append(row)

    final_threshold = float(np.median(np.array(loo_thresholds, dtype=np.float32)))

    # Append LOO rows + summary row
    append_rows_to_csv(output_csv_path, loo_rows)

    summary_row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "uncertainty_type": uncertainty_type,
        "case_id": "SUMMARY",
        "roi_mode": roi_mode,
        "ge_is_uncertain": ge_is_uncertain,
        "coarse_step": coarse_step,
        "fine_step": fine_step,
        "top_k": top_k,
        "t_star_train": final_threshold,
        "train_score_mean": float("nan"),


        "score_test": float("nan"),

        "unc_min_full": float("nan"),
        "unc_max_full": float("nan"),
        "qa_png_path": "",
    }
    append_rows_to_csv(output_csv_path, [summary_row])

    print(f"[INFO] Final threshold (median over folds) = {final_threshold:.6f}")
    print(f"[INFO] Total runtime: {time.perf_counter() - t0:.2f}s")

    return final_threshold

def main() -> None:
    """
    CLI entrypoint.
    Parse terminal arguments and run LOO threshold calibration.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="LOO calibration of uncertainty threshold (per uncertainty_type) with QA PNG and fail-fast alignment checks."
    )

    # Required paths
    parser.add_argument("--prediction_dir", type=str, required=True, help="Dir with binary predictions: {case_id}.nii.gz")
    parser.add_argument("--gt_dir", type=str, required=True, help="Dir with GT masks: {case_id}.nii.gz")
    parser.add_argument("--ct_dir", type=str, required=True, help="Dir with CT images: {case_id}_0000.nii.gz")
    parser.add_argument("--uncertainty_dir", type=str, required=True, help="Dir with uncertainty maps")
    parser.add_argument("--uncertainty_type", type=str, required=True, help="Suffix used in uncertainty filenames")
    parser.add_argument("--output_csv", type=str, required=True, help="CSV path to create/append results")
    parser.add_argument("--qa_dir", type=str, required=True, help="Dir where QA PNGs are saved")

    # Optional behavior
    parser.add_argument("--roi_mode", type=str, default="buffer_15mm", choices=["buffer_15mm", "full"])
    parser.add_argument("--ge_is_uncertain", action="store_true", help="If set, uncertain if U>=t (default True)")
    parser.add_argument("--le_is_uncertain", action="store_true", help="If set, uncertain if U<=t (overrides ge_is_uncertain)")

    # Optional grid parameters
    parser.add_argument("--coarse_step", type=float, default=0.1)
    parser.add_argument("--fine_step", type=float, default=0.01)
    parser.add_argument("--top_k", type=int, default=2)

    # Optional weights
    parser.add_argument("--wFN", type=float, default=1.0)
    parser.add_argument("--wFP", type=float, default=1.0)
    parser.add_argument("--wTP", type=float, default=1.0)
    parser.add_argument("--wTN", type=float, default=1.5)

    # Alignment tolerance
    parser.add_argument("--affine_tol", type=float, default=1e-3)
    parser.add_argument("--cache_case_scores", action="store_true",
                        help="Load each case once and reuse its per-threshold UEO across LOO folds (identical results, "
                             "much faster)")

    args = parser.parse_args()

    # Uncertainty comparison direction
    ge_is_uncertain = True
    if args.le_is_uncertain:
        ge_is_uncertain = False
    elif args.ge_is_uncertain:
        ge_is_uncertain = True

    weights = {"wFN": args.wFN, "wFP": args.wFP, "wTP": args.wTP, "wTN": args.wTN}

    final_t = calibrate_threshold_loo_nibabel(
        prediction_dir=args.prediction_dir,
        gt_dir=args.gt_dir,
        ct_dir=args.ct_dir,
        uncertainty_dir=args.uncertainty_dir,
        uncertainty_type=args.uncertainty_type,
        output_csv_path=args.output_csv,
        qa_output_dir=args.qa_dir,
        roi_mode=args.roi_mode,
        ge_is_uncertain=ge_is_uncertain,
        weights=weights,
        coarse_step=args.coarse_step,
        fine_step=args.fine_step,
        top_k=args.top_k,
        affine_tol=args.affine_tol,
        cache_case_scores=args.cache_case_scores,
    )

    print(f"Final threshold (median over LOO folds) for {args.uncertainty_type}: {final_t:.6f}")


if __name__ == "__main__":
    main()
