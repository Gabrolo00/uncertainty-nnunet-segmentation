#!/usr/bin/env python3
"""
Mappe DEU: incertezza meno quella attesa alla stessa distanza dal bordo (tabella E[U | d, lato]).

Uso: python compute_DEU.py --baseline_npz <tabella.npz> --uncertainty_dir <mappe> --pred_dir <predizioni> --out_dir <out> --uncertainty_type entropy
"""

from __future__ import annotations

import re
import argparse
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
from scipy.ndimage import binary_erosion, distance_transform_edt


def load_img(path: Path) -> nib.Nifti1Image:
    """Load a NIfTI image from disk."""
    return nib.load(str(path))


def get_spacing(img: nib.Nifti1Image) -> Tuple[float, float, float]:
    """Extract voxel spacing (x, y, z) in mm from NIfTI header."""
    return tuple(float(x) for x in img.header.get_zooms()[:3])


def extract_case_id_from_uncertainty(fname: str, uncertainty_type: str) -> Optional[str]:
    """
    Extract case ID from uncertainty filename.
    Expected pattern: {case_id}_{uncertainty_type}.nii(.gz)
    """
    # Prefix-agnostic, matcha sia AUTOMI_XXXXX che sXXXX
    suffix = re.escape(uncertainty_type)
    pattern = rf"^(.+)_{suffix}\.nii(\.gz)?$"
    m = re.match(pattern, fname)
    return m.group(1) if m else None


def compute_boundary(mask: np.ndarray) -> np.ndarray:
    """Compute a 1-voxel-thick boundary of a binary mask using erosion."""
    m = mask.astype(bool)
    if not m.any():
        return np.zeros_like(m, dtype=bool)
    er = binary_erosion(m, iterations=1, border_value=0)
    return m & (~er)


def signed_distance_to_boundary_mm(
    mask: np.ndarray,
    spacing: Tuple[float, float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute distance (in mm) to the predicted boundary, separately for inside/outside voxels.

    Returns:
      d_in: distance for voxels inside mask; 0 elsewhere
      d_out: distance for voxels outside mask; 0 elsewhere (or inf if no boundary exists)
    """
    m = mask.astype(bool)
    b = compute_boundary(mask)

    if not b.any():
        d_in = np.zeros_like(mask, dtype=np.float32)
        d_out = np.full_like(mask, np.inf, dtype=np.float32)
        return d_in, d_out

    # Boundary voxels are treated as zeros for distance transform.
    dist_to_boundary = distance_transform_edt(~b, sampling=spacing).astype(np.float32)

    d_in = np.zeros_like(dist_to_boundary, dtype=np.float32)
    d_out = np.zeros_like(dist_to_boundary, dtype=np.float32)
    d_in[m] = dist_to_boundary[m]
    d_out[~m] = dist_to_boundary[~m]
    return d_in, d_out


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


def compute_deu_map(
    U: np.ndarray,
    mask: np.ndarray,
    spacing: Tuple[float, float, float],
    mm: np.ndarray,
    inside_mean: np.ndarray,
    outside_mean: np.ndarray,
    *,
    verbose: bool = False,
) -> np.ndarray:
    """
    Compute DEU map = max(U - E[U|d,side], 0) for voxels with distance < max_mm.
    Voxels with distance >= max_mm are set to 0.
    """
    if U.shape != mask.shape:
        raise ValueError(f"Shape mismatch: U{U.shape} vs mask{mask.shape}")

    if mm.ndim != 1 or mm.size == 0:
        raise ValueError("Baseline mm must be a non-empty 1D array")

    # Derive max_mm from the last available bin in baseline mm.
    max_mm = int(mm[-1])

    if inside_mean.shape[0] <= max_mm or outside_mean.shape[0] <= max_mm:
        raise ValueError(
            "Baseline mean arrays are shorter than max_mm derived from mm[-1]. "
            f"max_mm={max_mm}, inside_mean_len={inside_mean.shape[0]}, "
            f"outside_mean_len={outside_mean.shape[0]}"
        )

    m = (mask > 0.5)

    d_in, d_out = signed_distance_to_boundary_mm(mask.astype(np.uint8), spacing=spacing)

    deu = np.zeros_like(U, dtype=np.float32)

    # INSIDE
    if np.any(m):
        d = d_in[m]
        k = np.floor(d).astype(np.int32)

        # Use only voxels with distance < max_mm (strict)
        valid = (k >= 0) & (k < max_mm)
        if np.any(valid):
            exp_mean = inside_mean[k[valid]].astype(np.float32, copy=False)
            diff = U[m][valid] - exp_mean
            pos = diff > 0.0
            if np.any(pos):
                tmp = np.zeros_like(diff, dtype=np.float32)
                tmp[pos] = diff[pos].astype(np.float32, copy=False)
                idx = np.flatnonzero(m)
                deu.flat[idx[valid]] = tmp

    # OUTSIDE
    if np.any(~m):
        d = d_out[~m]
        k = np.floor(d).astype(np.int32)

        valid = (k >= 0) & (k < max_mm)
        if np.any(valid):
            exp_mean = outside_mean[k[valid]].astype(np.float32, copy=False)
            diff = U[~m][valid] - exp_mean
            pos = diff > 0.0
            if np.any(pos):
                tmp = np.zeros_like(diff, dtype=np.float32)
                tmp[pos] = diff[pos].astype(np.float32, copy=False)
                idx = np.flatnonzero(~m)
                deu.flat[idx[valid]] = tmp

    # Safety: avoid NaN/inf propagation
    deu[~np.isfinite(deu)] = 0.0

    if verbose:
        nz = int(np.count_nonzero(deu))
        print(f"[INFO] max_mm={max_mm} (from baseline mm[-1])")
        print(f"[INFO] DEU non-zero voxels: {nz} / {deu.size}")

    return deu


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute DEU maps from uncertainty maps using baseline E[U|d,side]."
    )

    parser.add_argument(
        "--baseline_npz",
        type=Path,
        required=True,
        help="Path to baseline npz (e.g., baseline_EU_given_d_mm_inside_outside.npz).",
    )
    parser.add_argument(
        "--uncertainty_dir",
        type=Path,
        required=True,
        help="Directory with uncertainty maps named AUTOMI_XXXXX_{uncertainty_type}.nii(.gz).",
    )
    parser.add_argument(
        "--pred_dir",
        type=Path,
        required=True,
        help="Directory with predicted binary masks named AUTOMI_XXXXX.nii(.gz).",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Output directory for DEU maps.",
    )
    parser.add_argument(
        "--uncertainty_type",
        type=str,
        required=True,
        choices=["entropy", "anti_confidence_fn", "anti_confidence_fp"],
        help="Type of uncertainty map to use.",
    )
    parser.add_argument(
        "--no_resample",
        action="store_true",
        help="Disable resampling if uncertainty and mask grids differ.",
    )
    parser.add_argument(
        "--interp_order",
        type=int,
        default=1,
        choices=[0, 1, 3],
        help="Interpolation order for resampling uncertainty maps (default: 1=linear).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load baseline
    if not args.baseline_npz.exists():
        raise FileNotFoundError(f"Baseline file not found: {args.baseline_npz}")

    base = np.load(str(args.baseline_npz))
    mm = base["mm"].astype(np.int32, copy=False)
    inside_mean = base["inside_mean"].astype(np.float32, copy=False)
    outside_mean = base["outside_mean"].astype(np.float32, copy=False)

    if args.verbose:
        print(f"[INFO] Baseline: {args.baseline_npz}")
        print(f"[INFO] mm bins: {mm[0]}..{mm[-1]} (n={mm.size})")

    # Gather uncertainty maps
    utype = args.uncertainty_type
    # Pattern prefix-agnostic (matcha sia AUTOMI_XXXXX che sXXXX)
    uncertainty_paths = (
        sorted(args.uncertainty_dir.glob(f"*_{utype}.nii.gz"))
        + sorted(args.uncertainty_dir.glob(f"*_{utype}.nii"))
    )

    if not uncertainty_paths:
        raise FileNotFoundError(f"No uncertainty maps found in {args.uncertainty_dir}")

    used = 0
    skipped = []

    for up in uncertainty_paths:
        case_id = extract_case_id_from_uncertainty(up.name, utype)
        if case_id is None:
            skipped.append((up.name, "filename does not match expected pattern"))
            continue

        # Find prediction mask
        mp = args.pred_dir / f"{case_id}.nii.gz"
        if not mp.exists():
            mp = args.pred_dir / f"{case_id}.nii"
        if not mp.exists():
            skipped.append((case_id, "missing prediction mask"))
            continue

        if args.verbose:
            print(f"[INFO] Processing {case_id}")

        U_img = load_img(up)
        M_img = load_img(mp)

        M = M_img.get_fdata(dtype=np.float32)
        mask = (M > 0.5).astype(np.uint8)

        spacing = get_spacing(M_img)

        # Align U to mask grid if needed
        if U_img.shape != M_img.shape or not np.allclose(U_img.affine, M_img.affine, atol=1e-4):
            if args.no_resample:
                skipped.append((case_id, f"grid mismatch U{U_img.shape} vs M{M_img.shape} and resample disabled"))
                continue
            try:
                U = resample_uncertainty_to_mask(U_img, M_img, order=args.interp_order)
                if U.shape != mask.shape:
                    skipped.append((case_id, f"resample failed: got U{U.shape} expected {mask.shape}"))
                    continue
            except Exception as e:
                skipped.append((case_id, f"resample error: {type(e).__name__}: {e}"))
                continue
        else:
            U = U_img.get_fdata(dtype=np.float32)

        # Compute DEU
        deu = compute_deu_map(
            U=U,
            mask=mask,
            spacing=spacing,
            mm=mm,
            inside_mean=inside_mean,
            outside_mean=outside_mean,
            verbose=args.verbose,
        )

        # Save
        out_name = f"{case_id}_DEU_{utype}.nii.gz"
        out_path = args.out_dir / out_name

        out_img = nib.Nifti1Image(deu.astype(np.float32, copy=False), affine=M_img.affine, header=M_img.header)
        out_img.set_data_dtype(np.float32)
        nib.save(out_img, str(out_path))

        used += 1

    print(f"[OK] Saved DEU maps to: {args.out_dir}")
    print(f"[INFO] Used cases: {used}, skipped: {len(skipped)}")
    if args.verbose and skipped:
        print("[INFO] First skipped examples:", skipped[:10])


if __name__ == "__main__":
    main()
