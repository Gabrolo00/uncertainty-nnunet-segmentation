"""
Valuta le mappe di incertezza come rilevatori di errori alla soglia calibrata: recall, precision, F1 e falsi
allarmi nella regione di interesse (15 mm) e nella regione senza la banda di bordo.

Uso: python evaluate_threshold.py --pred_dir <predizioni> --gt_dir <label> --unc_dir <mappe> --thresh_csv <soglie.csv> --output_dir <out>
"""
import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional, Tuple


def load_nifti(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load NIfTI file and return data + spacing."""
    img = nib.load(path)
    data = img.get_fdata()
    spacing = img.header.get_zooms()[:3]  # (x, y, z) spacing in mm
    return data, spacing



def normalize_minmax_full(u: np.ndarray) -> np.ndarray:
    """Min-max normalize an array to [0,1] using global min/max. Safe for constant maps."""
    u = u.astype(np.float32, copy=False)
    umin = float(np.nanmin(u))
    umax = float(np.nanmax(u))
    if not np.isfinite(umin) or not np.isfinite(umax) or umax - umin == 0.0:
        return np.zeros_like(u, dtype=np.float32)
    return (u - umin) / (umax - umin)


def compute_signed_distance_transform(binary_mask: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    """
    Compute signed distance transform from binary mask.
    Positive inside, negative outside.
    """
    # Distance from outside to inside (positive)
    dist_inside = distance_transform_edt(binary_mask, sampling=spacing)

    # Distance from inside to outside (negative)
    dist_outside = distance_transform_edt(~binary_mask.astype(bool), sampling=spacing)

    # Combine: positive inside, negative outside
    signed_dist = dist_inside - dist_outside

    return signed_dist


def compute_metrics(
        FP: np.ndarray,
        FN: np.ndarray,
        errors: np.ndarray,
        correct: np.ndarray,
        uncertain: np.ndarray,
        mask: Optional[np.ndarray] = None
) -> Dict:
    """
    Compute all evaluation metrics on a subset of voxels defined by mask.
    If mask is None, use all voxels.
    """
    if mask is not None:
        FP = FP & mask
        FN = FN & mask
        errors = errors & mask
        correct = correct & mask
        uncertain = uncertain & mask

    # Counts
    num_FP = int(FP.sum())
    num_FN = int(FN.sum())
    num_errors = int(errors.sum())
    num_correct = int(correct.sum())
    num_uncertain = int(uncertain.sum())
    num_total = int(FP.size if mask is None else mask.sum())

    # Intersections
    TP_FP = int((FP & uncertain).sum())
    TP_FN = int((FN & uncertain).sum())
    TP_errors = int((errors & uncertain).sum())
    FP_alarms = int((correct & uncertain).sum())

    # Metrics (handle division by zero)
    recall_FP = TP_FP / num_FP if num_FP > 0 else np.nan
    recall_FN = TP_FN / num_FN if num_FN > 0 else np.nan
    recall_total = TP_errors / num_errors if num_errors > 0 else np.nan
    precision = TP_errors / num_uncertain if num_uncertain > 0 else np.nan
    false_alarm_rate = FP_alarms / num_correct if num_correct > 0 else np.nan

    # F1 score
    if precision > 0 and recall_total > 0:
        f1 = 2 * (precision * recall_total) / (precision + recall_total)
    else:
        f1 = np.nan

    # Dice coefficient between uncertain mask and error mask
    dice = 2 * TP_errors / (num_errors + num_uncertain) if (num_errors + num_uncertain) > 0 else np.nan

    return {
        'num_voxels_total': num_total,
        'num_errors_total': num_errors,
        'num_FP': num_FP,
        'num_FN': num_FN,
        'num_uncertain': num_uncertain,
        'recall_FP': recall_FP,
        'recall_FN': recall_FN,
        'recall_total': recall_total,
        'precision': precision,
        'false_alarm_rate': false_alarm_rate,
        'f1_score': f1,
        'dice_unc_vs_errors': dice
    }


def save_visualization(
        case_id: str,
        unc_type: str,
        pred: np.ndarray,
        gt: np.ndarray,
        unc_map: np.ndarray,
        uncertain_mask: np.ndarray,
        errors: np.ndarray,
        output_path: str
):
    """
    Save a multi-panel visualization to check alignment.
    Shows middle slice in axial plane.
    """
    # Get middle slice
    mid_slice = pred.shape[2] // 2

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Original data
    axes[0, 0].imshow(pred[:, :, mid_slice].T, cmap='gray', origin='lower')
    axes[0, 0].set_title('Prediction')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(gt[:, :, mid_slice].T, cmap='gray', origin='lower')
    axes[0, 1].set_title('Ground Truth')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(unc_map[:, :, mid_slice].T, cmap='viridis', origin='lower')
    axes[0, 2].set_title(f'Uncertainty Map\n({unc_type})')
    axes[0, 2].axis('off')

    # Row 2: Overlays
    # Prediction vs GT
    overlay1 = np.zeros((*pred.shape[:2], 3))
    overlay1[pred[:, :, mid_slice] == 1] = [0, 0, 1]  # Blue: prediction
    overlay1[gt[:, :, mid_slice] == 1] = [1, 0, 0]  # Red: GT
    overlay1[(pred[:, :, mid_slice] == 1) & (gt[:, :, mid_slice] == 1)] = [1, 0, 1]  # Purple: overlap
    axes[1, 0].imshow(overlay1.transpose(1, 0, 2), origin='lower')
    axes[1, 0].set_title('Pred (blue) vs GT (red)')
    axes[1, 0].axis('off')

    # Uncertain mask
    axes[1, 1].imshow(uncertain_mask[:, :, mid_slice].T, cmap='Reds', origin='lower', alpha=0.7)
    axes[1, 1].set_title('Uncertain Mask')
    axes[1, 1].axis('off')

    # Errors vs Uncertain
    overlay2 = np.zeros((*pred.shape[:2], 3))
    overlay2[errors[:, :, mid_slice]] = [1, 0, 0]  # Red: errors
    overlay2[uncertain_mask[:, :, mid_slice]] = [0, 1, 0]  # Green: uncertain
    overlay2[errors[:, :, mid_slice] & uncertain_mask[:, :, mid_slice]] = [1, 1, 0]  # Yellow: both
    axes[1, 2].imshow(overlay2.transpose(1, 0, 2), origin='lower')
    axes[1, 2].set_title('Errors (red) vs Uncertain (green)')
    axes[1, 2].axis('off')

    plt.suptitle(f'{case_id} - {unc_type}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def evaluate_uncertainty_maps(dataset_id: str,
                              pred_dir: Optional[str] = None,
                              gt_dir: Optional[str] = None,
                              unc_dir: Optional[str] = None,
                              thresh_csv: Optional[str] = None,
                              output_dir: Optional[str] = None,
                              cases: Optional[list] = None,
                              unc_types: Optional[list] = None,
                              border_mm: Optional[list] = None):
    """
    Main evaluation function for uncertainty maps.

    All the paths are required. `cases` / `unc_types` optionally restrict the rows
    of the threshold CSV that are evaluated (useful for quick tests).
    """
    print(f"\n{'=' * 80}")
    print(f"Starting evaluation for Dataset {dataset_id}")
    print(f"{'=' * 80}\n")

    missing = [n for n, v in [("pred_dir", pred_dir), ("gt_dir", gt_dir), ("unc_dir", unc_dir),
                               ("thresh_csv", thresh_csv), ("output_dir", output_dir)] if v is None]
    if missing:
        raise ValueError(f"Missing required paths: {missing}")

    # Create output directories
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    viz_dir = Path(output_dir) / "visualizations"
    viz_dir.mkdir(exist_ok=True)

    print(f"📂 Output directory: {output_dir}")
    print(f"📂 Visualization directory: {viz_dir}\n")

    # Load threshold CSV
    print(f"📊 Loading thresholds from: {thresh_csv}")
    thresholds_df = pd.read_csv(thresh_csv, sep=',')

    # Filter out SUMMARY rows
    original_len = len(thresholds_df)
    thresholds_df = thresholds_df[thresholds_df['case_id'] != 'SUMMARY']
    print(f"   Loaded {len(thresholds_df)} rows (filtered {original_len - len(thresholds_df)} SUMMARY rows)\n")

    # Optional restriction to a subset of cases / uncertainty types
    if cases:
        thresholds_df = thresholds_df[thresholds_df['case_id'].isin(cases)]
    if unc_types:
        thresholds_df = thresholds_df[thresholds_df['uncertainty_type'].isin(unc_types)]
    if cases or unc_types:
        print(f"   Restricted to {len(thresholds_df)} rows (cases={cases}, unc_types={unc_types})\n")

    # Get unique cases and uncertainty types
    unique_cases = thresholds_df['case_id'].unique()
    unique_unc_types = thresholds_df['uncertainty_type'].unique()

    print(f"📋 Found {len(unique_cases)} cases")
    print(f"📋 Found {len(unique_unc_types)} uncertainty types")
    print(f"   Types: {', '.join(unique_unc_types)}\n")

    results = []
    total_iterations = len(thresholds_df)

    print(f"🚀 Starting processing of {total_iterations} combinations...\n")

    for idx, row in thresholds_df.iterrows():
        case_id = row['case_id']
        unc_type = row['uncertainty_type']
        t_star = row['t_star_train']

        print(f"[{idx + 1}/{total_iterations}] Processing: {case_id} - {unc_type}")

        # Construct file paths
        pred_path = os.path.join(pred_dir, f"{case_id}.nii.gz")
        gt_path = os.path.join(gt_dir, f"{case_id}.nii.gz")
        unc_path = os.path.join(unc_dir, f"{case_id}_{unc_type}.nii.gz")

        # Check if files exist
        if not os.path.exists(pred_path):
            print(f"   ⚠️  Prediction not found: {pred_path}")
            continue
        if not os.path.exists(gt_path):
            print(f"   ⚠️  Ground truth not found: {gt_path}")
            continue
        if not os.path.exists(unc_path):
            print(f"   ⚠️  Uncertainty map not found: {unc_path}")
            continue

        # Load data
        print(f"   📥 Loading volumes...")
        pred, spacing = load_nifti(pred_path)
        gt, _ = load_nifti(gt_path)
        unc_map, _ = load_nifti(unc_path)

        # Convert to binary if needed
        pred = (pred > 0.5).astype(bool)
        gt = (gt > 0.5).astype(bool)

        # Special handling for confidence_map: invert to get uncertainty
        if unc_type == 'confidence_map':
            print(f"   🔄 Inverting confidence_map to uncertainty")
            unc_map = 1.0 - unc_map

        # Normalize uncertainty map to [0,1] (to match threshold calibration scale)
        unc_map = normalize_minmax_full(unc_map)

        # 1. Verifica che spacing e shape coincidano
        print(f"   🔍 Checking alignment...")
        print(f"      Pred shape: {pred.shape}, spacing: {spacing}")
        gt_spacing = nib.load(gt_path).header.get_zooms()[:3]
        unc_spacing = nib.load(unc_path).header.get_zooms()[:3]

        if not np.allclose(spacing, gt_spacing, atol=1e-6):
            print(f"      ⚠️  Spacing mismatch GT: {gt_spacing}")

        if not np.allclose(spacing, unc_spacing, atol=1e-6):
            print(f"      ⚠️  Spacing mismatch UNC: {unc_spacing}")

        # 2. Verifica affine matrix
        pred_img = nib.load(pred_path)
        gt_img = nib.load(gt_path)
        unc_img = nib.load(unc_path)

        if not np.allclose(pred_img.affine, gt_img.affine, atol=1e-6):
            print(f"      ⚠️  Affine mismatch pred-GT!")
            print(f"         Diff: {np.abs(pred_img.affine - gt_img.affine).max()}")

        if not np.allclose(pred_img.affine, unc_img.affine, atol=1e-6):
            print(f"      ⚠️  Affine mismatch pred-UNC!")
            print(f"         Diff: {np.abs(pred_img.affine - unc_img.affine).max()}")

        # 3. Sanity check specifico per anti_confidence
        if unc_type == 'anti_confidence_fn':
            # Conta quanti voxel hanno unc > 0 dove pred = 1
            pred_positive = (pred > 0.5)
            unc_on_positive = unc_map[pred_positive]
            n_nonzero = (unc_on_positive > 1e-8).sum()

            if n_nonzero > 0:
                print(f"      ⚠️  anti_conf_fn: {n_nonzero}/{pred_positive.sum()} voxels > 0 where pred=1")
                print(f"         Max value: {unc_on_positive.max():.6e}")
                print(f"         This suggests spatial misalignment!")

        if unc_type == 'anti_confidence_fp':
            # Conta quanti voxel hanno unc > 0 dove pred = 0
            pred_negative = (pred <= 0.5)
            unc_on_negative = unc_map[pred_negative]
            n_nonzero = (unc_on_negative > 1e-8).sum()

            if n_nonzero > 0:
                print(f"      ⚠️  anti_conf_fp: {n_nonzero}/{pred_negative.sum()} voxels > 0 where pred=0")
                print(f"         Max value: {unc_on_negative.max():.6e}")
                print(f"         This suggests spatial misalignment!")

        # Compute error masks
        print(f"   🧮 Computing error masks...")
        FP = pred & (~gt)
        FN = (~pred) & gt
        errors = FP | FN
        correct = ~errors

        # Binarize uncertainty map with threshold
        print(f"   📏 Applying threshold t* = {t_star:.4f}")
        uncertain = unc_map >= t_star

        # Compute distance transforms
        print(f"   📐 Computing distance transforms (spacing: {spacing})...")

        # ROI: union(pred, gt) + 15mm buffer outside (includes full pred/gt region)
        union_mask = pred | gt
        signed_dist_union = compute_signed_distance_transform(union_mask, spacing)
        roi_15mm = signed_dist_union >= -15.0  # inside union OR within 15mm outside

        # Optional border filter: exclude ±2mm band around prediction boundary (inside and outside)
        signed_dist_pred = compute_signed_distance_transform(pred, spacing)
        no_border_2mm = np.abs(signed_dist_pred) > 2.0

        roi_15mm_no_border = roi_15mm & no_border_2mm
        # Larghezze della fascia esclusa (--border_mm, default 2 mm); la riga a 2 mm ha roi_mode 'roi15mm_no_border2mm'.
        widths = border_mm if border_mm else [2.0]

        # Compute metrics for ROI 15mm
        print(f"   📊 Computing ROI (union + 15mm buffer) metrics...")
        metrics_roi = compute_metrics(FP, FN, errors, correct, uncertain, mask=roi_15mm)
        metrics_roi['roi_mode'] = 'roi15mm'
        metrics_roi['num_voxels_in_roi'] = int(roi_15mm.sum())

        # Compute metrics for ROI 15mm + no-border(w mm), one row per width ()
        metrics_nb_list = []
        for w in widths:
            print(f"   📊 Computing ROI + no-border({w:g}mm) metrics...")
            roi_nb_w = roi_15mm & (np.abs(signed_dist_pred) > w)
            m_nb = compute_metrics(FP, FN, errors, correct, uncertain, mask=roi_nb_w)
            m_nb['roi_mode'] = f'roi15mm_no_border{w:g}mm'
            m_nb['num_voxels_in_roi'] = int(roi_nb_w.sum())
            metrics_nb_list.append(m_nb)
# Add metadata to all rows
        for m in [metrics_roi] + metrics_nb_list:
            m['dataset_id'] = dataset_id
            m['case_id'] = case_id
            m['uncertainty_type'] = unc_type
            m['t_star_train'] = t_star
            # Flag cases with empty GT and empty prediction (e.g. "noise"
            # cases without trachea). There the signed distance is measured from
            # the volume borders, so roi15mm is a band along the image edges and
            # precision / false_alarm_rate are not meaningful: filter on this flag.
            m['union_empty'] = not bool(union_mask.any())

        # Verify dice score consistency with CSV
        csv_dice = row.get('score_test', np.nan)
        computed_dice = metrics_roi['dice_unc_vs_errors']
        if not np.isnan(csv_dice) and not np.isnan(computed_dice):
            dice_diff = abs(csv_dice - computed_dice)
            if dice_diff > 0.01:
                print(f"   ⚠️  Dice mismatch: CSV={csv_dice:.4f}, Computed={computed_dice:.4f}, Diff={dice_diff:.4f}")

        results.extend([metrics_roi] + metrics_nb_list)

        # Save visualization (one per case-uncertainty combination)
        viz_path = viz_dir / f"{case_id}_{unc_type}.png"
        print(f"   🎨 Saving visualization: {viz_path.name}")
        save_visualization(case_id, unc_type, pred, gt, unc_map, uncertain, errors, str(viz_path))

        print(f"   ✅ Completed\n")

    print(f"\n{'=' * 80}")
    print(f"💾 Saving results...")
    print(f"{'=' * 80}\n")

    # Save detailed CSV
    df_detailed = pd.DataFrame(results)
    detailed_csv_path = os.path.join(output_dir, "metrics_per_case_4.csv")
    df_detailed.to_csv(detailed_csv_path, index=False)
    print(f"✅ Saved detailed metrics: {detailed_csv_path}")
    print(f"   Shape: {df_detailed.shape}")

    # Aggregate by (uncertainty_type, roi_mode)
    print(f"\n📊 Computing aggregated statistics...")
    agg_metrics = ['recall_FP', 'recall_FN', 'recall_total', 'precision', 'f1_score', 'false_alarm_rate']

    agg_dict = {metric: ['mean', 'std', 'median'] for metric in agg_metrics}

    df_agg = df_detailed.groupby(['dataset_id', 'uncertainty_type', 'roi_mode']).agg(agg_dict)

    # Flatten column names
    df_agg.columns = ['_'.join(col).strip() for col in df_agg.columns.values]
    df_agg = df_agg.reset_index()

    agg_csv_path = os.path.join(output_dir, "metrics_aggregated_4.csv")
    df_agg.to_csv(agg_csv_path, index=False)
    print(f"✅ Saved aggregated metrics: {agg_csv_path}")
    print(f"   Shape: {df_agg.shape}")

    print(f"\n{'=' * 80}")
    print(f"🎉 Evaluation completed successfully!")
    print(f"{'=' * 80}\n")

    # Print summary statistics
    print(f"📈 Summary:")
    print(f"   Total cases processed: {len(df_detailed['case_id'].unique())}")
    print(f"   Total uncertainty types: {len(df_detailed['uncertainty_type'].unique())}")
    print(f"   Total rows in detailed CSV: {len(df_detailed)}")
    print(f"   Total rows in aggregated CSV: {len(df_agg)}")
    print(f"\n✨ All done!\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate calibrated uncertainty maps at t* (roi15mm and roi15mm_no_border2mm)."
    )
    parser.add_argument("--dataset_id", type=str, default="904",
                        help="Dataset id (written in the output 'dataset_id' column).")
    parser.add_argument("--pred_dir", type=str, default=None, help="Dir with binary predictions {case_id}.nii.gz")
    parser.add_argument("--gt_dir", type=str, default=None, help="Dir with GT labels {case_id}.nii.gz")
    parser.add_argument("--unc_dir", type=str, default=None, help="Dir with maps {case_id}_{uncertainty_type}.nii.gz")
    parser.add_argument("--thresh_csv", type=str, default=None, help="Threshold calibration CSV (from Calibrate_threshold_loo_v2.py)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--cases", type=str, default="", help="Optional comma-separated case ids to evaluate")
    parser.add_argument("--unc_types", type=str, default="", help="Optional comma-separated uncertainty types")
    parser.add_argument("--border_mm", type=str, default="2",
                        help="Larghezze (mm) della fascia esclusa attorno al bordo, separate da virgola (es. '2,3,4.5')")
    args = parser.parse_args()

    evaluate_uncertainty_maps(
        args.dataset_id,
        pred_dir=args.pred_dir,
        gt_dir=args.gt_dir,
        unc_dir=args.unc_dir,
        thresh_csv=args.thresh_csv,
        output_dir=args.output_dir,
        cases=[c for c in args.cases.split(",") if c] or None,
        unc_types=[u for u in args.unc_types.split(",") if u] or None,
        border_mm=[float(w) for w in args.border_mm.split(",") if w],
    )
