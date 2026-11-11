"""
Calcola le mappe di incertezza voxel per voxel dai logit dell'ensemble e dei singoli membri, con temperatura T.

Uso: python total_uncertainty_measures.py --input_dir <logit> --output_dir <mappe> --T <T*> [--measures entropy mutual_information ...]
"""
import torch
import numpy as np
import nibabel as nib
import os
from scipy.special import softmax
from scipy.stats import entropy
import glob
import argparse
import sys
from scipy.ndimage import zoom



# --- HELPER FUNCTIONS ---

def resample_logits_to_reference(logits, ref_shape_zyx):
    """
    Resample logits from (C, Z, Y, X) to match reference shape (Z, Y, X).

    Args:
        logits: numpy array (C, Z, Y, X)
        ref_shape_zyx: target shape as tuple (Z, Y, X)

    Returns:
        resampled logits (C, Z, Y, X) at target resolution
    """
    # Force float32 for scipy compatibility
    logits = logits.astype(np.float32, copy=False)

    C = logits.shape[0]
    current_shape = logits.shape[1:]  # (Z, Y, X)

    if current_shape == ref_shape_zyx:
        return logits  # already aligned

    zoom_factors = [
        ref_shape_zyx[0] / current_shape[0],
        ref_shape_zyx[1] / current_shape[1],
        ref_shape_zyx[2] / current_shape[2]
    ]

    print(f"   [Resample] Logits {current_shape} -> {ref_shape_zyx}, zoom={zoom_factors}")

    # Resample each channel separately using linear interpolation
    resampled = np.zeros((C, *ref_shape_zyx), dtype=np.float32)
    for c in range(C):
        resampled[c] = zoom(logits[c], zoom_factors, order=1)

    return resampled


def kl_divergence(p, q):
    """
    Computes KL divergence between two discrete distributions over axis 0 (classes).
    Includes clipping for numerical stability.
    """
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)
    return np.sum(p * np.log(p / q), axis=0)


def save_measure_nifti(measure_map, measure_name, ref_nii, output_path):
    """
    Saves a measure map as NIfTI.

    Args:
        measure_map: numpy array (Z, Y, X) - already at correct resolution
        measure_name: string identifier
        ref_nii: reference NIfTI object for affine/header
        output_path: output file path

    Note: NO resampling is performed - input must already be at correct resolution
    """

    # Normalize Entropy and MI if needed
    # Anche expected_entropy e total_entropy sono entropie, normalizzate in bit come le altre
    if measure_name in ['entropy', 'mutual_information', 'expected_entropy', 'total_entropy']:
        final_map = measure_map / np.log(2)
    else:
        final_map = measure_map

    # Convert from numpy (Z, Y, X) to NIfTI (X, Y, Z)
    final_map = final_map.transpose(2, 1, 0)

    # Ensure float32
    final_map = final_map.astype(np.float32, copy=False)

    # Save using the reference affine/header
    unc_img = nib.Nifti1Image(final_map, ref_nii.affine, ref_nii.header)
    nib.save(unc_img, output_path)
    print(f"      ✓ Saved: {os.path.basename(output_path)}")


# --- CORE LOGIC ---

def compute_uncertainty_measures(fold_logits_list, mean_logits, pred_binary, selected_measures, T):
    """
    Computes requested uncertainty measures efficiently.

    CRITICAL: All inputs must be at the SAME resolution (already resampled if needed)

    Args:
        fold_logits_list: List of numpy arrays (C, Z, Y, X) - already resampled
        mean_logits: Numpy array (C, Z, Y, X) - already resampled
        pred_binary: Binary prediction (Z, Y, X) with {0,1} values - OFFICIAL segmentation
        selected_measures: Set of strings containing measures to compute
        T: Temperature scaling parameter

    Returns:
        Dictionary of uncertainty maps, each (Z, Y, X)
    """
    results = {}

    print(f"   [Computation] Input shapes: mean_logits={mean_logits.shape}, pred_binary={pred_binary.shape}")

    # Validate alignment
    spatial_shape = mean_logits.shape[1:]  # (Z, Y, X)
    if pred_binary.shape != spatial_shape:
        raise ValueError(
            f"Shape mismatch! pred_binary {pred_binary.shape} != logits spatial {spatial_shape}. "
            f"Resampling should have been done before calling this function."
        )

    # 1. Base Probabilities from mean logits (for measures that don't need exact mean from folds)
    mean_probs_from_mean_logits = softmax(mean_logits / T, axis=0)

    # --- MEASURES: SINGLE PREDICTION (No folds needed) ---

    if 'entropy' in selected_measures:
        # Use mean_logits for entropy (standard approach)
        results['entropy'] = entropy(mean_probs_from_mean_logits, axis=0)

    if 'confidence_map' in selected_measures:
        # Max probability across classes
        results['confidence_map'] = np.max(mean_probs_from_mean_logits, axis=0)

    # Check if we need to process folds
    fold_measures_needed = {
        'variance', 'std', 'sample_mean_uncertainty',
        'mutual_information', 'epkl',
        'anti_confidence_fn', 'anti_confidence_fp', 'anti_confidence_total',
        'boundary_disagreement', 'max_fold_deviation',
        'expected_entropy', 'total_entropy'
    }

    if not selected_measures.intersection(fold_measures_needed):
        return results

    # --- MEASURES: MULTIPLE PREDICTIONS (Folds needed) ---

    if not fold_logits_list:
        print("   [WARNING] Fold measures requested but no folds provided. Skipping fold-based measures.")
        return results

    n_folds = len(fold_logits_list)
    img_shape = mean_logits.shape  # (C, Z, Y, X)

    print(f"   [Computation] Processing {n_folds} folds...")

    # Accumulators
    sum_probs = np.zeros(img_shape, dtype=np.float32)  # For exact mean
    sum_sq_probs = np.zeros(img_shape, dtype=np.float32)  # For variance

    # Conditional accumulators
    # La media delle entropie dei fold serve anche per expected_entropy
    sum_fold_entropy = np.zeros(spatial_shape, dtype=np.float32) if {
        'mutual_information', 'expected_entropy'}.intersection(selected_measures) else None
    sum_kl = np.zeros(spatial_shape, dtype=np.float32) if 'epkl' in selected_measures else None

    # Min/Max accumulators for measures based on Class 1 (foreground/tumor)
    max_tumor_prob = np.zeros(spatial_shape, dtype=np.float32) if {
        'anti_confidence_fn', 'boundary_disagreement'
    }.intersection(selected_measures) else None

    min_tumor_prob = np.ones(spatial_shape, dtype=np.float32) if {
        'anti_confidence_fp', 'boundary_disagreement'
    }.intersection(selected_measures) else None

    print(f"   [Computation] Pass 1: Accumulating statistics...")

    for idx, f_logits in enumerate(fold_logits_list):
        # Validate fold shape
        if f_logits.shape != img_shape:
            raise ValueError(
                f"Fold {idx} shape mismatch! {f_logits.shape} != {img_shape}. "
                f"All folds must be resampled before calling this function."
            )

        f_probs = softmax(f_logits / T, axis=0)

        # Standard accumulation
        sum_probs += f_probs
        if {'variance', 'std', 'sample_mean_uncertainty'}.intersection(selected_measures):
            sum_sq_probs += f_probs ** 2

        # Mutual Information accumulation
        if sum_fold_entropy is not None:
            sum_fold_entropy += entropy(f_probs, axis=0)

        # EPKL accumulation
        if sum_kl is not None:
            sum_kl += kl_divergence(f_probs, mean_probs_from_mean_logits)

        # Min/Max accumulation (Class 1 = foreground)
        tumor_prob = f_probs[1]
        if max_tumor_prob is not None:
            max_tumor_prob = np.maximum(max_tumor_prob, tumor_prob)
        if min_tumor_prob is not None:
            min_tumor_prob = np.minimum(min_tumor_prob, tumor_prob)

        # Free memory
        del f_probs

    # --- COMPUTE DERIVED MEASURES ---

    # Exact mean probability from folds
    mean_probs_exact = sum_probs / n_folds

    # 1. Variance & Std
    if {'variance', 'std', 'sample_mean_uncertainty'}.intersection(selected_measures):
        mean_sq_probs = sum_sq_probs / n_folds
        probs_var_all = mean_sq_probs - mean_probs_exact ** 2
        probs_var = np.mean(probs_var_all, axis=0)  # Average over classes
        probs_var = np.maximum(probs_var, 0.0)  # Clip negative values due to numerical errors

        if 'variance' in selected_measures:
            results['variance'] = probs_var

        if 'std' in selected_measures:
            results['std'] = np.sqrt(probs_var)

        if 'sample_mean_uncertainty' in selected_measures:
            # SMU = Variance + Confidence Penalty
            confidence_penalty = 1.0 - np.max(mean_probs_exact, axis=0)
            results['sample_mean_uncertainty'] = probs_var + confidence_penalty

    # 2. Mutual Information
    if 'mutual_information' in selected_measures:
        # MI = H(mean) - mean(H(samples))
        entropy_of_mean = entropy(mean_probs_exact, axis=0)
        mean_of_entropy = sum_fold_entropy / n_folds
        mi_map = entropy_of_mean - mean_of_entropy
        results['mutual_information'] = np.maximum(mi_map, 0.0)

    # Scomposizione H[mean_p] = E_fold[H] + MI, con le probabilita' mediate
    # sui fold (non la softmax dei logit mediati come la mappa 'entropy').
    #   expected_entropy = E_fold[H(p_fold)]  -> incertezza aleatoria ("attesa")
    #   total_entropy    = H(mean_p)          -> totale; total_entropy - expected_entropy = mutual_information
    if 'expected_entropy' in selected_measures:
        results['expected_entropy'] = sum_fold_entropy / n_folds
    if 'total_entropy' in selected_measures:
        results['total_entropy'] = entropy(mean_probs_exact, axis=0)

    # 3. EPKL (Expected Pairwise KL divergence)
    if 'epkl' in selected_measures:
        epkl_map = sum_kl / n_folds
        epkl_map = np.nan_to_num(epkl_map, nan=0.0, posinf=0.0, neginf=0.0)

        # Apply safe mask: if entropy is very low, model is sure, EPKL should be 0
        if 'entropy' in results:
            safe_mask = results['entropy'] < 0.05
        else:
            safe_mask = entropy(mean_probs_from_mean_logits, axis=0) < 0.05
        epkl_map[safe_mask] = 0.0
        results['epkl'] = epkl_map

    # 4. Anti-Confidence FN (False Negative risk)
    # CRITICAL FIX: Use OFFICIAL binary prediction, not derived from logits
    if 'anti_confidence_fn' in selected_measures:
        # pred_binary: 0 = negative prediction, 1 = positive prediction
        # Active only where prediction is negative (pred = 0)
        # Shows how high the most "optimistic" fold's probability is
        negative_pred_mask = (pred_binary == 0).astype(np.float32)
        results['anti_confidence_fn'] = max_tumor_prob * negative_pred_mask

        print(f"      [anti_conf_fn] Activated on {negative_pred_mask.sum():.0f}/{pred_binary.size} voxels (pred=0)")

    # 5. Anti-Confidence FP (False Positive risk)
    # CRITICAL FIX: Use OFFICIAL binary prediction
    if 'anti_confidence_fp' in selected_measures:
        # Active only where prediction is positive (pred = 1)
        # Shows how low the most "pessimistic" fold's probability is
        positive_pred_mask = (pred_binary == 1).astype(np.float32)
        results['anti_confidence_fp'] = (1.0 - min_tumor_prob) * positive_pred_mask

        print(f"      [anti_conf_fp] Activated on {positive_pred_mask.sum():.0f}/{pred_binary.size} voxels (pred=1)")

    # 6. Anti-Confidence Total
    if 'anti_confidence_total' in selected_measures:
        if 'anti_confidence_fn' not in results or 'anti_confidence_fp' not in results:
            raise ValueError("anti_confidence_total requires both anti_confidence_fn and anti_confidence_fp")
        results['anti_confidence_total'] = results['anti_confidence_fn'] + results['anti_confidence_fp']

    # 7. Boundary Disagreement
    # Shows disagreement where folds cross the decision boundary (0.5)
    if 'boundary_disagreement' in selected_measures:
        crossing_mask = (max_tumor_prob > 0.5) & (min_tumor_prob < 0.5)
        results['boundary_disagreement'] = (max_tumor_prob - min_tumor_prob) * crossing_mask.astype(np.float32)

        print(f"      [boundary_disagreement] Found {crossing_mask.sum():.0f} crossing voxels")

    # 8. Max Fold Deviation (requires second pass)
    if 'max_fold_deviation' in selected_measures:
        print("   [Computation] Pass 2: Computing Max Fold Deviation...")
        max_dev = np.zeros(spatial_shape, dtype=np.float32)
        mean_tumor_exact = mean_probs_exact[1]

        for idx, f_logits in enumerate(fold_logits_list):
            f_probs = softmax(f_logits / T, axis=0)
            dev = np.abs(f_probs[1] - mean_tumor_exact)
            max_dev = np.maximum(max_dev, dev)
            del f_probs

        results['max_fold_deviation'] = max_dev

    print(f"   [Computation] ✓ Computed {len(results)} uncertainty measures")

    return results


def main():
    parser = argparse.ArgumentParser(description="Unified Uncertainty Measures Calculation with Alignment Fix")
    parser.add_argument('--input_dir', type=str, required=True, help="Path containing patient logits")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to save uncertainty maps")
    parser.add_argument(
        "--T",
        type=float,
        default=2.619667,
        help="Temperature for temperature scaling (applied to logits before softmax)"
    )

    # Measure flags
    available_measures = [
        'entropy', 'variance', 'std', 'mutual_information', 'epkl',
        'sample_mean_uncertainty', 'anti_confidence_fn', 'anti_confidence_fp', 'anti_confidence_total',
        'boundary_disagreement', 'max_fold_deviation', 'confidence_map',
        'expected_entropy', 'total_entropy'
    ]

    parser.add_argument('--measures', nargs='+', choices=available_measures, default=[],
                        help="List of measures to compute. If empty, computes ALL.")

    args = parser.parse_args()

    print(f"[total_uncertainty_measures] Temperature (--T) usata: {args.T}"
          f"{' (DEFAULT — non passata esplicitamente, verificare che sia corretta per QUESTO dataset!)' if args.T == 2.619667 else ''}")

    # Determine measures to run
    selected_measures = set(args.measures)
    if not selected_measures:
        selected_measures = set(available_measures)
        print("No specific measures requested. Computing ALL measures.")
    else:
        print(f"Computing selected measures: {', '.join(selected_measures)}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Find patients
    # Un file <caso>_logits.pt per caso
    patient_files = glob.glob(os.path.join(args.input_dir, "*_logits.pt"))
    if not patient_files:
        print(f"❌ No patient files found in {args.input_dir}")
        return

    print(f"\n{'=' * 80}")
    print(f"Found {len(patient_files)} patients to process")
    print(f"{'=' * 80}\n")

    for pt_idx, pt_file in enumerate(patient_files, 1):
        # case_id = nome del file senza il suffisso _logits.pt
        patient_id = os.path.basename(pt_file).replace('_logits.pt', '')
        print(f"[{pt_idx}/{len(patient_files)}] Processing Patient {patient_id}")

        # 1. Load reference NIfTI (for shape, affine, header)
        ref_nii_file = glob.glob(os.path.join(args.input_dir, f"{patient_id}.nii.gz"))
        if not ref_nii_file:
            print(f"   ❌ Error: Reference .nii.gz not found for {patient_id}, skipping...")
            continue

        ref_nii = nib.load(ref_nii_file[0])
        print(f"   📂 Loaded reference: {os.path.basename(ref_nii_file[0])}")
        print(f"      Shape: {ref_nii.shape}, Spacing: {ref_nii.header.get_zooms()[:3]}")

        # 2. Load OFFICIAL binary prediction from reference NIfTI
        pred_data = ref_nii.get_fdata()
        pred_binary = (pred_data > 0.5).astype(np.uint8)
        # Convert from NIfTI (X, Y, Z) to numpy (Z, Y, X)
        pred_binary = pred_binary.transpose(2, 1, 0)
        print(f"   ✓ Loaded official prediction: shape {pred_binary.shape}, foreground={pred_binary.sum()} voxels")

        # Target shape for resampling
        ref_shape_zyx = pred_binary.shape  # (Z, Y, X)

        # 3. Load and resample mean logits
        mean_logits = torch.load(pt_file, weights_only=False).numpy()
        print(f"   📥 Loaded mean logits: shape {mean_logits.shape}")
        mean_logits = resample_logits_to_reference(mean_logits, ref_shape_zyx)

        # 4. Load and resample fold logits (if needed)
        fold_measures = {
            'variance', 'std', 'mutual_information', 'epkl', 'sample_mean_uncertainty',
            'anti_confidence_fn', 'anti_confidence_fp', 'anti_confidence_total',
            'boundary_disagreement', 'max_fold_deviation',
            'expected_entropy', 'total_entropy'
        }

        fold_logits_list = []
        if selected_measures.intersection(fold_measures):
            fold_pattern = os.path.join(args.input_dir, f"fold_logits/{patient_id}_fold_*_logits.pt")
            fold_files = sorted(glob.glob(fold_pattern))

            if fold_files:
                print(f"   📥 Loading {len(fold_files)} fold logits...")
                for f_idx, f_file in enumerate(fold_files):
                    f_logits = torch.load(f_file, weights_only=False).numpy()
                    f_logits = resample_logits_to_reference(f_logits, ref_shape_zyx)
                    fold_logits_list.append(f_logits)
                print(f"      ✓ All folds resampled to {ref_shape_zyx}")
            else:
                print(f"   ⚠️  Warning: No folds found for {patient_id}, but fold measures requested.")
                print(f"      Skipping fold-based measures.")

        # 5. Compute uncertainty measures
        print(f"   🧮 Computing uncertainty measures...")
        try:
            measures_dict = compute_uncertainty_measures(
                fold_logits_list,
                mean_logits,
                pred_binary,
                selected_measures,
                args.T
            )
        except Exception as e:
            print(f"   ❌ Error computing measures: {e}")
            import traceback
            traceback.print_exc()
            continue

        # 6. Save all measures
        print(f"   💾 Saving {len(measures_dict)} uncertainty maps...")
        for m_name, m_map in measures_dict.items():
            out_name = f"{patient_id}_{m_name}.nii.gz"
            out_path = os.path.join(args.output_dir, out_name)

            try:
                save_measure_nifti(m_map, m_name, ref_nii, out_path)
            except Exception as e:
                print(f"      ❌ Error saving {m_name}: {e}")

        print(f"   ✅ Completed Patient {patient_id}\n")

    print(f"{'=' * 80}")
    print(f"🎉 Processing complete! All maps saved to: {args.output_dir}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()