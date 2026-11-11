#!/usr/bin/env python3
"""
Tabella dell'incertezza attesa in funzione della distanza dal bordo predetto, E[U | d, lato], stimata su casi
non di test; serve a DEU ed EBCU.

Uso: python build_baseline_EU_given_d.py --entropy_dir <mappe> --pred_dir <predizioni> --out_dir <out> --uncertainty_type entropy
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
from scipy.ndimage import binary_erosion, distance_transform_edt
import argparse



@dataclass
class BaselineStats:
    count: np.ndarray
    mean: np.ndarray
    median: np.ndarray
    p10: np.ndarray
    p90: np.ndarray
    p99: np.ndarray


def load_img(path: Path) -> nib.Nifti1Image:
    return nib.load(str(path))


def get_spacing(img: nib.Nifti1Image) -> Tuple[float, float, float]:
    return tuple(float(x) for x in img.header.get_zooms()[:3])


def extract_case_id_from_entropy(fname: str) -> str | None:
    # ORIGINALE: pattern = rf"^(AUTOMI_\d+)_{suffix}\.nii(\.gz)?$"
    # Prefix-agnostic, matcha sia AUTOMI_XXXXX che sXXXX
    suffix = re.escape(args.uncertainty_type)
    pattern = rf"^(.+)_{suffix}\.nii(\.gz)?$"
    m = re.match(pattern, fname)
    return m.group(1) if m else None


def compute_boundary(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    if not m.any():
        return np.zeros_like(m, dtype=bool)
    er = binary_erosion(m, iterations=1, border_value=0)
    return m & (~er)


def signed_distance_to_boundary_mm(mask: np.ndarray, spacing: Tuple[float, float, float]) -> Tuple[np.ndarray, np.ndarray]:
    m = mask.astype(bool)
    b = compute_boundary(mask)
    if not b.any():
        d_in = np.zeros_like(mask, dtype=np.float32)
        d_out = np.full_like(mask, np.inf, dtype=np.float32)
        return d_in, d_out

    # distance to boundary voxels: boundary==True treated as zeros
    dist_to_boundary = distance_transform_edt(~b, sampling=spacing).astype(np.float32)

    # /correctness: NaN (not 0.0) for the wrong-side voxels, so
    # bin_by_mm's `np.isfinite` filter excludes them instead of dumping every
    # background voxel of the volume into bin 0.
    d_in = np.full_like(dist_to_boundary, np.nan, dtype=np.float32)
    d_out = np.full_like(dist_to_boundary, np.nan, dtype=np.float32)
    d_in[m] = dist_to_boundary[m]
    d_out[~m] = dist_to_boundary[~m]
    return d_in, d_out


def bin_by_mm(dist_mm: np.ndarray, U: np.ndarray, max_mm: int) -> List[np.ndarray]:
    bins: List[list] = [[] for _ in range(max_mm + 1)]
    finite = np.isfinite(dist_mm)
    d = dist_mm[finite].ravel()
    u = U[finite].ravel()

    keep = d >= 0.0
    d = d[keep]
    u = u[keep]

    k = np.floor(d).astype(np.int32)
    in_range = (k >= 0) & (k <= max_mm)
    k = k[in_range]
    u = u[in_range]

    if k.size == 0:
        return [np.array([], dtype=np.float32) for _ in range(max_mm + 1)]

    order = np.argsort(k)
    k_sorted = k[order]
    u_sorted = u[order]

    start = 0
    out = [np.array([], dtype=np.float32) for _ in range(max_mm + 1)]
    for mm in range(max_mm + 1):
        end = np.searchsorted(k_sorted, mm, side="right")
        out[mm] = u_sorted[start:end].astype(np.float32, copy=False)
        start = end
    return out


def summarize_bins(bins: List[np.ndarray]) -> BaselineStats:
    max_mm = len(bins) - 1
    count = np.zeros(max_mm + 1, dtype=np.int64)
    mean = np.full(max_mm + 1, np.nan, dtype=np.float32)
    median = np.full(max_mm + 1, np.nan, dtype=np.float32)
    p10 = np.full(max_mm + 1, np.nan, dtype=np.float32)
    p90 = np.full(max_mm + 1, np.nan, dtype=np.float32)
    p99 = np.full(max_mm + 1, np.nan, dtype=np.float32)

    for k, arr in enumerate(bins):
        n = int(arr.size)
        count[k] = n
        if n == 0:
            continue
        mean[k] = float(np.mean(arr))
        median[k] = float(np.median(arr))
        p10[k] = float(np.percentile(arr, 10))
        p90[k] = float(np.percentile(arr, 90))
        p99[k] = float(np.percentile(arr, 99))
    return BaselineStats(count=count, mean=mean, median=median, p10=p10, p90=p90, p99=p99)


def resample_uncertainty_to_mask(
    U_img: nib.Nifti1Image,
    mask_img: nib.Nifti1Image,
    order: int = 1,
) -> np.ndarray:
    """
    Resample uncertainty image to the mask grid (shape+affine).
    order=1 (linear) is appropriate for continuous uncertainty maps.
    """
    U_res = resample_from_to(U_img, (mask_img.shape, mask_img.affine), order=order)
    return U_res.get_fdata(dtype=np.float32)


def build_baseline(
    entropy_dir: Path,
    pred_dir: Path,
    out_npz: Path,
    out_json: Path,
    max_mm: int = 40,
    resample_if_mismatch: bool = True,
) -> None:
    u_type = args.uncertainty_type

    # stampa i percorsi e il tipo per debug
    print(f"[INFO] Entropy directory: {entropy_dir}")
    print(f"[INFO] Uncertainty type: {u_type}")

    # Pattern prefix-agnostic (matcha sia AUTOMI_XXXXX che sXXXX)
    print(f"[INFO] *_{u_type}.nii.gz")

    print("files in dir:", len(list(entropy_dir.glob("*"))))


    entropy_paths = (
            sorted(entropy_dir.glob(f"*_{u_type}.nii.gz")) +
            sorted(entropy_dir.glob(f"*_{u_type}.nii"))
    )
    if not entropy_paths:
        raise FileNotFoundError(f"No entropy maps found in {entropy_dir}")

    paths = entropy_paths
    print("matched:", len(paths))

    acc_in: List[List[np.ndarray]] = [[] for _ in range(max_mm + 1)]
    acc_out: List[List[np.ndarray]] = [[] for _ in range(max_mm + 1)]

    used_cases = 0
    skipped = []

    for ep in entropy_paths:
        case_id = extract_case_id_from_entropy(ep.name)
        print("Processing:", ep.name, "case_id:", case_id)
        if case_id is None:
            continue

        pp = pred_dir / f"{case_id}.nii.gz"
        if not pp.exists():
            pp = pred_dir / f"{case_id}.nii"
        if not pp.exists():
            skipped.append((case_id, "missing prediction mask"))
            continue


        U_img = load_img(ep)
        M_img = load_img(pp)

        M = M_img.get_fdata(dtype=np.float32)
        mask = (M > 0.5).astype(np.uint8)  # your masks are binary; this is safe
        spacing = get_spacing(M_img)

        # --- NEW: handle grid mismatch by resampling U to mask grid ---
        if U_img.shape != M_img.shape or not np.allclose(U_img.affine, M_img.affine, atol=1e-4):
            if not resample_if_mismatch:
                skipped.append((case_id, f"grid mismatch U{U_img.shape} vs M{M_img.shape} and resample disabled"))
                continue
            try:
                U = resample_uncertainty_to_mask(U_img, M_img, order=1)
                # optional sanity check
                if U.shape != mask.shape:
                    skipped.append((case_id, f"resample failed: got U{U.shape} expected {mask.shape}"))
                    continue
            except Exception as e:
                skipped.append((case_id, f"resample error: {type(e).__name__}: {e}"))
                continue
        else:
            U = U_img.get_fdata(dtype=np.float32)

        d_in, d_out = signed_distance_to_boundary_mm(mask, spacing=spacing)

        bins_in_case = bin_by_mm(d_in, U, max_mm=max_mm)
        bins_out_case = bin_by_mm(d_out, U, max_mm=max_mm)

        for k in range(max_mm + 1):
            if bins_in_case[k].size:
                acc_in[k].append(bins_in_case[k])
            if bins_out_case[k].size:
                acc_out[k].append(bins_out_case[k])

        used_cases += 1

    bins_in_all = [np.concatenate(acc_in[k]) if acc_in[k] else np.array([], dtype=np.float32) for k in range(max_mm + 1)]
    bins_out_all = [np.concatenate(acc_out[k]) if acc_out[k] else np.array([], dtype=np.float32) for k in range(max_mm + 1)]

    stats_in = summarize_bins(bins_in_all)
    stats_out = summarize_bins(bins_out_all)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        max_mm=np.int32(max_mm),
        mm=np.arange(max_mm + 1, dtype=np.int32),
        inside_count=stats_in.count,
        inside_mean=stats_in.mean,
        inside_median=stats_in.median,
        inside_p10=stats_in.p10,
        inside_p90=stats_in.p90,
        inside_p99=stats_in.p99,
        outside_count=stats_out.count,
        outside_mean=stats_out.mean,
        outside_median=stats_out.median,
        outside_p10=stats_out.p10,
        outside_p90=stats_out.p90,
        outside_p99=stats_out.p99,
        used_cases=np.int32(used_cases),
    )

    payload = {
        "max_mm": int(max_mm),
        "mm": list(range(max_mm + 1)),
        "used_cases": int(used_cases),
        "inside": {
            "count": stats_in.count.tolist(),
            "mean": stats_in.mean.tolist(),
            "median": stats_in.median.tolist(),
            "p10": stats_in.p10.tolist(),
            "p90": stats_in.p90.tolist(),
            "p99": stats_in.p99.tolist(),
        },
        "outside": {
            "count": stats_out.count.tolist(),
            "mean": stats_out.mean.tolist(),
            "median": stats_out.median.tolist(),
            "p10": stats_out.p10.tolist(),
            "p90": stats_out.p90.tolist(),
            "p99": stats_out.p99.tolist(),
        },
        "skipped_cases": [{"case_id": cid, "reason": reason} for cid, reason in skipped],
        "notes": {
            "distance_definition": "min Euclidean distance (mm) to predicted binary boundary; signed by inside/outside",
            "resampling": "uncertainty maps are resampled to mask grid if shape/affine mismatch (linear interpolation)",
        },
    }
    out_json.write_text(json.dumps(payload, indent=2))

    print(f"[OK] Baseline saved to:\n  {out_npz}\n  {out_json}")
    print(f"[INFO] Used cases: {used_cases}, skipped: {len(skipped)}")
    if skipped:
        print("[INFO] First skipped examples:", skipped[:10])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute baseline E[U | d] from training uncertainty maps and predicted binary masks"
    )

    parser.add_argument(
        "--entropy_dir",
        type=Path,
        required=True,
        help="Directory with entropy maps (training+validation)",
    )
    parser.add_argument(
        "--pred_dir",
        type=Path,
        required=True,
        help="Directory with predicted binary masks corresponding to entropy maps",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Output directory for baseline files",
    )
    parser.add_argument(
        "--max_mm",
        type=int,
        default=60,
        help="Maximum distance (in mm) for baseline bins (default: 60)",
    )
    parser.add_argument(
        "--no_resample",
        action="store_true",
        help="Disable resampling if entropy and mask grids differ",
    )
    parser.add_argument(
        "--interp_order",
        type=int,
        default=1,
        choices=[0, 1, 3],
        help="Interpolation order for resampling uncertainty maps (default: 1 = linear)",
    )
    parser.add_argument(
        "--uncertainty_type",
        type=str,
        required=True,
        choices=["entropy", "anti_confidence_fn", "anti_confidence_fp"],
        help="Type of uncertainty map to use"
    )

    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    out_npz = args.out_dir / "baseline_EU_given_d_mm_inside_outside.npz"
    out_json = args.out_dir / "baseline_EU_given_d_mm_inside_outside.json"

    build_baseline(
        entropy_dir=args.entropy_dir,
        pred_dir=args.pred_dir,
        out_npz=out_npz,
        out_json=out_json,
        max_mm=args.max_mm,
        resample_if_mismatch=not args.no_resample,
    )

