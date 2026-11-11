#!/usr/bin/env python3
"""
Mappe EBCU: quanto l'incertezza supera i percentili attesi alla stessa distanza dal bordo, in [0, 1].

Uso: python make_ebcu_maps.py --baseline_npz <tabella.npz> --entropy_dir <mappe> --pred_dir <predizioni> --out_dir <out> --uncertainty_type entropy
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Tuple

import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
from scipy.ndimage import binary_erosion, distance_transform_edt


# -----------------------------
# I/O utilities
# -----------------------------
def load_img(path: Path) -> nib.Nifti1Image:
    """Load a NIfTI image (keeps header+affine)."""
    return nib.load(str(path))


def get_spacing(img: nib.Nifti1Image) -> Tuple[float, float, float]:
    """Return voxel spacing (dx, dy, dz) in mm from NIfTI header."""
    return tuple(float(x) for x in img.header.get_zooms()[:3])


def extract_case_id_from_entropy(fname: str, suffix) -> str | None:
    # Prefix-agnostic, matcha sia AUTOMI_XXXXX che sXXXX
    pattern = rf"^(.+)_{suffix}\.nii(\.gz)?$"
    m = re.match(pattern, fname)
    return m.group(1) if m else None


# -----------------------------
# Geometry / distance utilities
# -----------------------------
def compute_boundary(mask: np.ndarray) -> np.ndarray:
    """
    Compute a simple binary boundary of a binary mask using erosion:
      boundary = mask AND NOT(erode(mask))

    mask: array with values {0,1} (or boolean)
    returns: boolean boundary array
    """
    m = mask.astype(bool)
    if not m.any():
        return np.zeros_like(m, dtype=bool)

    # Erode by 1 voxel; border_value=0 means outside image is background
    er = binary_erosion(m, iterations=1, border_value=0)
    boundary = m & (~er)
    return boundary


def signed_distance_to_boundary_mm(
    mask: np.ndarray, spacing: Tuple[float, float, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute distance (in mm) to the predicted binary boundary, separately:
      - d_in_mm: distances for voxels INSIDE the mask, 0 outside
      - d_out_mm: distances for voxels OUTSIDE the mask, 0 inside

    Distances are computed in physical space using sampling=spacing.

    Notes:
    - boundary voxels have distance 0
    - if the mask is empty, boundary does not exist; we set outside distances to +inf.
    """
    m = mask.astype(bool)
    b = compute_boundary(mask)

    if not b.any():
        # Empty prediction or no boundary: cannot compute meaningful outside distance
        d_in = np.zeros_like(mask, dtype=np.float32)
        d_out = np.full_like(mask, np.inf, dtype=np.float32)
        return d_in, d_out

    # distance_transform_edt returns distance to the nearest zero-valued voxel.
    # We want distance to boundary voxels (b==True), so we make boundary zeros by passing ~b.
    dist_to_boundary = distance_transform_edt(~b, sampling=spacing).astype(np.float32)

    d_in = np.zeros_like(dist_to_boundary, dtype=np.float32)
    d_out = np.zeros_like(dist_to_boundary, dtype=np.float32)

    # Fill distances only on the appropriate side
    d_in[m] = dist_to_boundary[m]     # inside distances
    d_out[~m] = dist_to_boundary[~m]  # outside distances

    return d_in, d_out


def resample_to_ref(
    src_img: nib.Nifti1Image, ref_img: nib.Nifti1Image, order: int = 1
) -> np.ndarray:
    """
    Resample src_img to match ref_img grid (shape + affine).
    order=1 (linear) is appropriate for continuous uncertainty maps.
    """
    res = resample_from_to(src_img, (ref_img.shape, ref_img.affine), order=order)
    return res.get_fdata(dtype=np.float32)


# -----------------------------
# EBCU computation
# -----------------------------
def ebcu_from_baseline_percentiles(
    U: np.ndarray,
    d_in: np.ndarray,
    d_out: np.ndarray,
    inside_q90: np.ndarray,
    inside_q99: np.ndarray,
    outside_q90: np.ndarray,
    outside_q99: np.ndarray,
    clip01: bool = True,
    eps: float = 1e-8,
    verbose: bool = False,
) -> np.ndarray:
    """
    Compute EBCU (percentile-exceedance version) separately inside/outside.

    For each voxel x:
      Let k = floor(d(x)) be the integer millimeter bin
      EBCU(x) = max(0, (U(x) - q90[k]) / (q99[k] - q90[k] + eps))

    - Inside voxels use inside_q90/inside_q99
    - Outside voxels use outside_q90/outside_q99
    - Boundary voxels (d=0) are left as 0 by design (no "boundary alarm")

    Returns:
      ebcu map aligned with U.
    """
    max_mm = len(inside_q90) - 1
    ebcu = np.zeros_like(U, dtype=np.float32)

    # -------------------------
    # Inside mask
    # -------------------------
    in_mask = d_in > 0  # strictly away from boundary inside
    k_in = np.floor(d_in[in_mask]).astype(np.int32)
    valid_in = (k_in >= 0) & (k_in <= max_mm)

    if np.any(valid_in):
        # indices of inside voxels
        idx = np.where(in_mask)
        # keep only those with valid bins
        vox = tuple(a[valid_in] for a in idx)
        kk = k_in[valid_in]

        q90 = inside_q90[kk]
        q99 = inside_q99[kk]
        val = (U[vox] - q90) / (q99 - q90 + eps)
        ebcu[vox] = np.maximum(0.0, val).astype(np.float32)

    if verbose:
        n_in = int(in_mask.sum())
        n_in_valid = int(valid_in.sum()) if k_in.size else 0
        print(f"    Inside voxels: {n_in} (valid bins: {n_in_valid})")

    # -------------------------
    # Outside mask
    # -------------------------
    out_mask = d_out > 0  # strictly away from boundary outside
    k_out = np.floor(d_out[out_mask]).astype(np.int32)
    valid_out = (k_out >= 0) & (k_out <= max_mm)

    if np.any(valid_out):
        idx = np.where(out_mask)
        vox = tuple(a[valid_out] for a in idx)
        kk = k_out[valid_out]

        q90 = outside_q90[kk]
        q99 = outside_q99[kk]
        val = (U[vox] - q90) / (q99 - q90 + eps)
        ebcu[vox] = np.maximum(0.0, val).astype(np.float32)

    if verbose:
        n_out = int(out_mask.sum())
        n_out_valid = int(valid_out.sum()) if k_out.size else 0
        print(f"    Outside voxels: {n_out} (valid bins: {n_out_valid})")

    # Optional: clip to [0,1] so heatmap is easy to view/threshold
    if clip01:
        ebcu = np.clip(ebcu, 0.0, 1.0)

    return ebcu


# -----------------------------
# Main script
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate EBCU maps for test cases using a precomputed baseline NPZ."
    )
    ap.add_argument(
        "--baseline_npz",
        type=Path,
        required=True,
        help="Path to baseline_EU_given_d_mm_inside_outside.npz",
    )
    ap.add_argument(
        "--entropy_dir",
        type=Path,
        required=True,
        help="Directory with test entropy maps: AUTOMI_xxx_entropy.nii.gz",
    )
    ap.add_argument(
        "--pred_dir",
        type=Path,
        required=True,
        help="Directory with predicted binary masks: AUTOMI_xxx.nii.gz",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Output directory for EBCU maps",
    )
    ap.add_argument(
        "--interp_order",
        type=int,
        default=1,
        choices=[0, 1, 3],
        help="Resampling interpolation order if grids differ (default: 1=linear)",
    )
    ap.add_argument(
        "--no_clip01",
        action="store_true",
        help="Do not clip EBCU to [0,1] (default: clip enabled)",
    )
    ap.add_argument(
        "--max_cases",
        type=int,
        default=0,
        help="If >0, process only the first N cases (debug option)",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra debug info for each case",
    )
    ap.add_argument(
        "--uncertainty_type",
        type=str,
        required=True,
        help="Type of uncertainty used to compute the maps",
    )

    args = ap.parse_args()

    print("\n[INFO] Loading baseline from:", args.baseline_npz)
    base = np.load(args.baseline_npz)

    # Baseline percentiles per integer distance bin (mm)
    inside_q90 = base["inside_p90"].astype(np.float32)
    inside_q99 = base["inside_p99"].astype(np.float32)
    outside_q90 = base["outside_p90"].astype(np.float32)
    outside_q99 = base["outside_p99"].astype(np.float32)

    max_mm = len(inside_q90) - 1
    print(f"[INFO] Baseline bins: 0..{max_mm} mm")
    print(f"[INFO] Output EBCU will be clipped to [0,1]: {not args.no_clip01}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("[INFO] Output directory:", args.out_dir)

    # Find all test entropy maps
    # Pattern prefix-agnostic (matcha sia AUTOMI_XXXXX che sXXXX)
    entropy_paths = sorted(args.entropy_dir.glob(f"*_{args.uncertainty_type}.nii.gz")) + sorted(
        args.entropy_dir.glob(f"*_{args.uncertainty_type}.nii")
    )

    if not entropy_paths:
        raise FileNotFoundError(f"No entropy maps found in {args.entropy_dir}")

    if args.max_cases and args.max_cases > 0:
        entropy_paths = entropy_paths[: args.max_cases]
        print(f"[INFO] Debug: processing only first {len(entropy_paths)} cases")

    print(f"[INFO] Found {len(entropy_paths)} uncertainty maps in:", args.entropy_dir)

    # Process each case
    for i, ep in enumerate(entropy_paths, start=1):
        case_id = extract_case_id_from_entropy(ep.name,  re.escape(args.uncertainty_type))
        if case_id is None:
            print(f"[WARN] Skipping file with unexpected name: {ep.name}")
            continue

        # Predicted binary mask path
        mask_path = args.pred_dir / f"{case_id}.nii.gz"
        if not mask_path.exists():
            mask_path = args.pred_dir / f"{case_id}.nii"

        print(f"\n[CASE {i}/{len(entropy_paths)}] {case_id}")
        print(f"  - Uncertainty map: {ep}")
        print(f"  - Mask file  : {mask_path}")

        if not mask_path.exists():
            print(f"  [SKIP] Missing mask for {case_id} in {args.pred_dir}")
            continue

        # Load images
        U_img = load_img(ep)
        M_img = load_img(mask_path)

        # Mask grid is the "reference" grid (viewer uses this prediction)
        M = M_img.get_fdata(dtype=np.float32)
        mask = (M > 0.5).astype(np.uint8)
        spacing = get_spacing(M_img)

        if args.verbose:
            print(f"  - Mask shape  : {mask.shape}")
            print(f"  - Mask spacing: {spacing} mm")

        # Ensure uncertainty map is on the same grid (shape + affine)
        if U_img.shape != M_img.shape or not np.allclose(U_img.affine, M_img.affine, atol=1e-4):
            print("  - Grid mismatch detected (entropy vs mask). Resampling entropy to mask grid...")
            U = resample_to_ref(U_img, M_img, order=args.interp_order)
            if args.verbose:
                print(f"    Resampled entropy shape: {U.shape}")
        else:
            U = U_img.get_fdata(dtype=np.float32)

        if args.verbose:
            print(f"  - Entropy stats: min={float(np.nanmin(U)):.4f}, max={float(np.nanmax(U)):.4f}, mean={float(np.nanmean(U)):.4f}")

        # Compute signed distances to boundary (in mm)
        print("  - Computing signed distance-to-boundary (mm) ...")
        d_in, d_out = signed_distance_to_boundary_mm(mask, spacing=spacing)

        if args.verbose:
            # Some diagnostics about distances
            din_vals = d_in[d_in > 0]
            dout_vals = d_out[d_out > 0]
            if din_vals.size:
                print(f"    d_in:  min={float(din_vals.min()):.2f} mm, max={float(din_vals.max()):.2f} mm")
            else:
                print("    d_in:  no inside voxels away from boundary (empty or single-voxel structure)")
            if dout_vals.size and np.isfinite(dout_vals).all():
                print(f"    d_out: min={float(dout_vals.min()):.2f} mm, max={float(dout_vals.max()):.2f} mm")
            else:
                print("    d_out: contains inf (likely empty mask) or no outside voxels away from boundary")

        # Compute EBCU using baseline percentiles
        print("  - Computing EBCU map from baseline percentiles (inside/outside) ...")
        ebcu = ebcu_from_baseline_percentiles(
            U=U,
            d_in=d_in,
            d_out=d_out,
            inside_q90=inside_q90,
            inside_q99=inside_q99,
            outside_q90=outside_q90,
            outside_q99=outside_q99,
            clip01=not args.no_clip01,
            verbose=args.verbose,
        )

        if args.verbose:
            print(f"  - EBCU stats: min={float(np.nanmin(ebcu)):.4f}, max={float(np.nanmax(ebcu)):.4f}, mean={float(np.nanmean(ebcu)):.4f}")

        # Save as NIfTI aligned to mask/prediction grid
        out_path = args.out_dir / f"{case_id}_ebcu_{args.uncertainty_type}.nii.gz"
        print(f"  - Saving EBCU to: {out_path}")
        nib.save(nib.Nifti1Image(ebcu.astype(np.float32), affine=M_img.affine, header=M_img.header), str(out_path))

        print("  [OK] Done.")

    print("\n[INFO] All done.")


if __name__ == "__main__":
    main()

