"""
Viewer delle segmentazioni e delle mappe di incertezza: CT, reference, predizione, errori, mappe (continue o
binarizzate alla soglia calibrata), confronto entropia/DEU/EBCU e Dice per fetta.

Uso: streamlit run uncertainty_viewer.py --server.port 8501
"""
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import streamlit as st
from pathlib import Path
import torch
from skimage.transform import resize
from skimage.segmentation import find_boundaries
import io
import pandas as pd


st.set_page_config(layout="wide", page_title="Segmentation Viewer")

# ============================================================================
# CONFIGURAZIONE
# ============================================================================


# Dataset scelto dal menu nella sidebar invece che modificando il codice.
# Per aggiungere un dataset basta aggiungere una voce qui. THRESHOLD_DIR puo' essere un CSV unico
# (es. *_ALL.csv) oppure un pattern glob: i CSV per misura che corrispondono vengono concatenati.
_P = "/mnt/sda2/grolo"
_R = "/mnt/sda2/grolo/nnUNet/raw"
_O = "/mnt/sda2/grolo"
_H = "/mnt/sda2/grolo/humanitas_source"
DATASET_PRESETS = {
    # Rene pulito (Dataset904, soli criteri meccanici), trachea from-scratch e Humanitas con il 904.
    "[904] Rene pulito — deep ensemble": dict(
        DATASET_ID="904",
        OUTPUT_DIR=f"{_P}/kidney/predictions/deep/",
        GT_DIR=f"{_R}/Dataset904_KidneyTSMechanical/labelsTs/",
        INPUT_DIR=f"{_R}/Dataset904_KidneyTSMechanical/imagesTs",
        TOTALSEG_DIR=f"{_R}/Dataset904_KidneyTSMechanical/total_segmentator_structures",  # non esiste: nessun overlay organi
        UNCERTAINTY_DIR=f"{_O}/kidney/deep/kidney_uncertainty_maps/",
        THRESHOLD_DIR=f"{_O}/kidney/deep/kidney_threshold_calibration_*.csv",
    ),
    "[904] Rene pulito — checkpoint ensemble (CycLR)": dict(
        DATASET_ID="904",
        OUTPUT_DIR=f"{_P}/kidney/predictions/checkpoint/",
        GT_DIR=f"{_R}/Dataset904_KidneyTSMechanical/labelsTs/",
        INPUT_DIR=f"{_R}/Dataset904_KidneyTSMechanical/imagesTs",
        TOTALSEG_DIR=f"{_R}/Dataset904_KidneyTSMechanical/total_segmentator_structures",
        UNCERTAINTY_DIR=f"{_O}/kidney/checkpoint/kidney_cyclr_uncertainty_maps/",
        THRESHOLD_DIR=f"{_O}/kidney/checkpoint/kidney_cyclr_threshold_calibration_*.csv",
    ),
    "[904] Test Humanitas · deep ensemble 904": dict(
        DATASET_ID="HUM",
        OUTPUT_DIR=f"/mnt/sda2/grolo/kidney/external/pred_deep/",
        GT_DIR=f"/mnt/sda2/grolo/kidney/external/cropped/labelsTs/",
        INPUT_DIR=f"/mnt/sda2/grolo/kidney/external/cropped/images",
        TOTALSEG_DIR=f"{_H}/total_segmentator_structures",
        UNCERTAINTY_DIR=f"/mnt/sda2/grolo/kidney/external/maps_deep/",
        THRESHOLD_DIR=f"/mnt/sda2/grolo/kidney/external/eval/tau_from_TS_deep904*.csv",
    ),
    # Fine-tuned sul test esterno: non ha soglie calibrate, la vista binarizzata usa la soglia di ripiego.
    "[904] Test Humanitas · fine-tuned da 291 (senza tau*)": dict(
        DATASET_ID="HUM",
        OUTPUT_DIR=f"/mnt/sda2/grolo/kidney/external/pred_finetune/",
        GT_DIR=f"/mnt/sda2/grolo/kidney/external/cropped/labelsTs/",
        INPUT_DIR=f"/mnt/sda2/grolo/kidney/external/cropped/images",
        TOTALSEG_DIR=f"{_H}/total_segmentator_structures",
        UNCERTAINTY_DIR=f"/mnt/sda2/grolo/kidney/external/maps_finetune/",
        THRESHOLD_DIR=f"/mnt/sda2/grolo/kidney/external/eval/tau_from_TS_finetune904.csv",
    ),
    "[903] Trachea from-scratch — checkpoint ensemble (CycLR)": dict(
        DATASET_ID="903",
        OUTPUT_DIR=f"{_P}/trachea/predictions/checkpoint/",
        GT_DIR=f"{_R}/Dataset903_TracheaTS/labelsTs/",
        INPUT_DIR=f"{_R}/Dataset903_TracheaTS/imagesTs",
        TOTALSEG_DIR=f"{_R}/Dataset903_TracheaTS/total_segmentator_structures",
        UNCERTAINTY_DIR=f"{_O}/trachea/checkpoint/trachea_cyclr_uncertainty_maps/",
        THRESHOLD_DIR=f"{_O}/trachea/checkpoint/trachea_cyclr_threshold_calibration_*.csv",
    ),
    "[903] Trachea from-scratch — deep ensemble": dict(
        DATASET_ID="903",
        OUTPUT_DIR=f"{_P}/trachea/predictions/deep/",
        GT_DIR=f"{_R}/Dataset903_TracheaTS/labelsTs/",
        INPUT_DIR=f"{_R}/Dataset903_TracheaTS/imagesTs",
        TOTALSEG_DIR=f"{_R}/Dataset903_TracheaTS/total_segmentator_structures",
        UNCERTAINTY_DIR=f"{_O}/trachea/deep/trachea_uncertainty_maps/",
        THRESHOLD_DIR=f"{_O}/trachea/deep/trachea_threshold_calibration_*.csv",
    ),
}
# Nel menu solo i preset con predizioni gia' su disco (quelli futuri compaiono quando esistono)
DATASET_PRESETS = {k: v for k, v in DATASET_PRESETS.items() if os.path.isdir(v["OUTPUT_DIR"])}
# Nel menu solo i modelli della tesi, con nomi in inglese (Dataset902 e fine-tuned esclusi)
THESIS_PRESETS = {
    "[904] Rene pulito — deep ensemble": "Kidney — deep ensemble",
    "[904] Rene pulito — checkpoint ensemble (CycLR)": "Kidney — checkpoint ensemble",
    "[904] Test Humanitas · deep ensemble 904": "Kidney — deep ensemble, external test set",
    # fine-tuned: nessuna tau* (confronto del cap. 5 solo con metriche senza soglia), la vista binarizzata usa la soglia di ripiego
    "[904] Test Humanitas · fine-tuned da 291 (senza tau*)": "Kidney — fine-tuned deep ensemble, external test set",
    "[903] Trachea from-scratch — deep ensemble": "Trachea — deep ensemble",
    "[903] Trachea from-scratch — checkpoint ensemble (CycLR)": "Trachea — checkpoint ensemble",
}
DATASET_PRESETS = {new: DATASET_PRESETS[old] for old, new in THESIS_PRESETS.items() if old in DATASET_PRESETS}
_preset_name = st.sidebar.selectbox("📁 Dataset / model", list(DATASET_PRESETS))
globals().update(DATASET_PRESETS[_preset_name])
st.sidebar.caption(f"Mappe: `{UNCERTAINTY_DIR}`")

# Raggruppa le ~21 misure d'incertezza in categorie invece di una
# lista flat unica. Ogni misura compare in un solo gruppo. I gruppi vuoti
# (nessuna misura disponibile su disco per il case corrente) vengono
# nascosti a runtime, non qui.
UNCERTAINTY_GROUPS = {
    # Gruppi allineati alla Table measures della tesi; tolte le mappe non usate
    # (DEU/EBCU fn/fp, ebcu grezza, ebcu anti-confidence, combinate senza soglia per misura)
    "Mean prediction": [
        "entropy", "confidence_map", "sample_mean_uncertainty",
    ],
    "Distance-aware (DEU / EBCU)": [
        "DEU_entropy", "ebcu_entropy",
    ],
    # Scomposizione H[mean_p] = expected_entropy (aleatoria, "attesa") + mutual_information
    "Aleatoric / epistemic decomposition": [
        "total_entropy", "expected_entropy",
    ],
    "Ensemble disagreement": [
        "variance", "std", "mutual_information", "epkl", "max_fold_deviation", "boundary_disagreement",
    ],
    "Anti-confidence": [
        "anti_confidence_fn", "anti_confidence_fp", "anti_confidence_total",
    ],
    # Ogni misura divisa per la propria soglia (make_threshold_scaled_combined_maps.py)
    "Combined maps": [
        "combined_thr_switch_entropy_std", "combined_thr_max_entropy_std", "combined_thr_switch_DEU_MI",
    ],
}

# Nomi mostrati nell'interfaccia, uguali a quelli della tesi
DISPLAY_NAMES = {
    "entropy": "Entropy", "confidence_map": "Confidence", "sample_mean_uncertainty": "SMU",
    "DEU_entropy": "DEU", "ebcu_entropy": "EBCU",
    "total_entropy": "Total entropy", "expected_entropy": "Expected entropy",
    "variance": "Variance", "std": "Standard deviation", "mutual_information": "Mutual information",
    "epkl": "EPKL", "max_fold_deviation": "Max member deviation", "boundary_disagreement": "Boundary disagreement",
    "anti_confidence_fn": "Anti-confidence FN", "anti_confidence_fp": "Anti-confidence FP",
    "anti_confidence_total": "Anti-confidence total",
    "combined_thr_switch_entropy_std": "Switch H/σ", "combined_thr_max_entropy_std": "Max H/σ",
    "combined_thr_switch_DEU_MI": "Switch DEU/MI",
}


def display_name(t):
    return DISPLAY_NAMES.get(t, t.replace('_', ' ').title())


# Le tre misure "entropy-based" mostrate fianco a fianco nel compare
# mode (raw entropy -> correzione distance-aware DEU -> normalizzazione EBCU).
COMPARE_MODE_TYPES = ["entropy", "DEU_entropy", "ebcu_entropy"]

# ============================================================================
# FUNZIONI DI CARICAMENTO (con caching)
# ============================================================================
@st.cache_data
def load_thresholds(threshold_dir: str) -> dict:
    """
    Load the optimal thresholds CSV from threshold_dir/resultsVR.csv.
    Auto-detects separator (; or ,).
    Returns a dict keyed by (case_id, uncertainty_type) -> t_star_train value (float).
    Returns an empty dict if the file is missing or unreadable.
    """
    # Pattern glob -> concatena i CSV per misura (es. checkpoint ensemble senza _ALL.csv)
    if any(ch in threshold_dir for ch in "*?["):
        import glob as _glob
        files = [f for f in sorted(_glob.glob(threshold_dir)) if not f.endswith("_ALL.csv")]
        if not files:
            return {}
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        return {(str(r["case_id"]), str(r["uncertainty_type"])): float(r["t_star_train"]) for _, r in df.iterrows()}

    csv_path = threshold_dir if os.path.isfile(threshold_dir) else os.path.join(threshold_dir, "resultsVR.csv")

    if not os.path.exists(csv_path):
        return {}

    # Auto-detect separator: read first line and count ; vs ,
    with open(csv_path, "r") as f:
        first_line = f.readline()
    sep = ";" if first_line.count(";") >= first_line.count(",") else ","

    df = pd.read_csv(csv_path, sep=sep)

    # Build lookup dict: (case_id, uncertainty_type) -> t_star_train
    result = {}
    for _, row in df.iterrows():
        key = (str(row["case_id"]), str(row["uncertainty_type"]))
        result[key] = float(row["t_star_train"])

    return result


def get_tstar(thresholds: dict, case_id: str, uncertainty_type: str) -> float:
    """
    Look up the optimal threshold for a given case and uncertainty type.
    Shows a Streamlit warning and returns 0.0 if not found.
    """
    key = (case_id, uncertainty_type)
    if key in thresholds:
        return thresholds[key]

    st.sidebar.warning(
        f"⚠️ No threshold found for case '{case_id}' / type '{uncertainty_type}'. Using 0.05."
    )
    return 0.05


@st.cache_data
def scan_available_cases(output_dir):
    """Scansiona i case disponibili nella directory di output"""
    cases = []
    for f in sorted(os.listdir(output_dir)):
        if f.endswith('.nii.gz'): #and f.startswith('AUTOMI_'):
            case_id = f.replace('.nii.gz', '')
            cases.append(case_id)
    return cases


@st.cache_data
def get_available_organs(case_id, totalseg_dir):
    """Scansiona quali organi sono disponibili (solo nomi file, non carica dati)"""
    case_folder = os.path.join(totalseg_dir, f"{case_id}_0000")

    if not os.path.exists(case_folder):
        return []

    organs = []
    for f in sorted(os.listdir(case_folder)):
        if f.endswith('.nii.gz') and 'mask_mask_add' not in f:
            organ_name = f.replace('.nii.gz', '')
            organs.append(organ_name)

    return organs


@st.cache_data
def load_single_organ(case_id, organ_name, totalseg_dir):
    """Carica UN SOLO organo alla volta"""
    organ_path = os.path.join(totalseg_dir, f"{case_id}_0000", f"{organ_name}.nii.gz")

    if not os.path.exists(organ_path):
        return None

    organ_nii = nib.load(organ_path)
    return organ_nii.get_fdata().astype(np.float32)


@st.cache_data
def load_uncertainty_maps(case_id, uncertainty_dir, reference_shape):
    """Load advanced uncertainty maps and normalize per-case"""
    # Define uncertainty types
    # Carica solo le mappe dei gruppi mostrati (incluse le combinate)
    uncertainty_types = [t for types in UNCERTAINTY_GROUPS.values() for t in types]

    uncertainty = {}
    uncertainty_min = {}
    uncertainty_max = {}

    for unc_type in uncertainty_types:
        # Build path: {case_id}_{type}.nii.gz
        unc_path = os.path.join(uncertainty_dir, f"{case_id}_{unc_type}.nii.gz")

        if not os.path.exists(unc_path):
            continue

        # Load NIfTI file
        unc_nii = nib.load(unc_path)
        unc_data = unc_nii.get_fdata().astype(np.float32)

        # Verify and correct shape if needed
        if unc_data.shape != reference_shape:
            if unc_data.shape == (reference_shape[2], reference_shape[0], reference_shape[1]):
                # Shape is (D, H, W) instead of (H, W, D) → transpose (2, 1, 0)
                unc_data = np.transpose(unc_data, (2, 1, 0))
                st.sidebar.info(f"🔄 {unc_type}: transposed (2,1,0)")
            else:
                st.sidebar.error(f"❌ Cannot align {unc_type}: {unc_data.shape} vs {reference_shape}")
                continue

        if unc_type == 'confidence_map':
            # Invert confidence to uncertainty
            unc_data = np.max(unc_data) - unc_data

        if unc_type in ['entropy', 'mutual_information', 'expected_entropy', 'total_entropy']:
            # remove the normalization by log(2) to treat as bits if it's binary, so we can normalize later
            unc_data = unc_data * np.log(2)


        # Normalize to [0, 1] for this case (min-max normalization per-patient)
        min_val = unc_data.min()
        max_val = unc_data.max()


        # save original value
        uncertainty_min[unc_type] = min_val
        uncertainty_max[unc_type] = max_val

        if max_val > min_val:
            unc_data = (unc_data - min_val) / (max_val - min_val)
        else:
            # All values are the same, set to 0
            unc_data = np.zeros_like(unc_data)

        uncertainty[unc_type] = unc_data

    return uncertainty, uncertainty_min, uncertainty_max


@st.cache_data
def load_case_data(case_id, output_dir, gt_dir, input_dir):
    """Carica predizione, ground truth e CT originale per un case"""
    # Predizione
    pred_path = os.path.join(output_dir, f"{case_id}.nii.gz")
    pred_nii = nib.load(pred_path)
    pred = pred_nii.get_fdata().astype(np.float32)

    # Ground Truth
    gt_path = os.path.join(gt_dir, f"{case_id}.nii.gz")
    gt_nii = nib.load(gt_path)
    gt = gt_nii.get_fdata().astype(np.float32)

    # CT originale
    ct_path = os.path.join(input_dir, f"{case_id}_0000.nii.gz")
    ct_nii = nib.load(ct_path)
    ct = ct_nii.get_fdata().astype(np.float32)

    # Verifica allineamento
    if pred.shape != gt.shape or pred.shape != ct.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, gt={gt.shape}, ct={ct.shape}")

    voxel_sizes = ct_nii.header.get_zooms()[:3]

    return {
        'pred': pred,
        'gt': gt,
        'ct': ct,
        'voxel_sizes': voxel_sizes,
        'shape': pred.shape
    }


@st.cache_data
def compute_dice_per_slice(case_id, axis, output_dir, gt_dir, input_dir, threshold=0.5):
    """
    Dice score per ogni slice del volume, lungo l'asse scelto.
    load_case_data e' gia' cached, quindi qui non viene rifatto alcun I/O:
    e' solo un loop sulle slice gia' in memoria. Ritorna anche has_gt, la
    maschera delle slice con almeno un voxel di ground truth, per evitare
    che "slice peggiore" punti a una slice vuota (Dice=0 li' e' banale, non
    un fallimento del modello).
    """
    d = load_case_data(case_id, output_dir, gt_dir, input_dir)
    pred, gt = d['pred'], d['gt']
    n_slices = pred.shape[axis]

    dices = np.empty(n_slices, dtype=np.float32)
    has_gt = np.empty(n_slices, dtype=bool)
    for i in range(n_slices):
        p_slice = extract_slice(pred, axis, i)
        g_slice = extract_slice(gt, axis, i)
        has_gt[i] = np.any(g_slice >= threshold)
        dices[i] = compute_dice_2d(p_slice, g_slice, threshold)

    return dices, has_gt


@st.cache_data
def compute_case_metrics(case_id, output_dir, gt_dir, input_dir, threshold=0.5):
    """
    Dice 3D sull'intero case (non solo la slice corrente) e volumi
    di GT / Pred in mL, usando il voxel spacing reale del volume.
    """
    d = load_case_data(case_id, output_dir, gt_dir, input_dir)
    pred_bin = d['pred'] >= threshold
    gt_bin = d['gt'] >= threshold

    intersection = np.sum(pred_bin & gt_bin)
    union = np.sum(pred_bin) + np.sum(gt_bin)
    dice_3d = 2.0 * intersection / union if union > 0 else 0.0

    voxel_vol_mm3 = float(np.prod(d['voxel_sizes']))
    gt_vol_ml = np.sum(gt_bin) * voxel_vol_mm3 / 1000.0
    pred_vol_ml = np.sum(pred_bin) * voxel_vol_mm3 / 1000.0

    return dice_3d, gt_vol_ml, pred_vol_ml

# ============================================================================
# FUNZIONI DI PROCESSING
# ============================================================================
def figure_to_png_bytes(fig, dpi=200):
    """
    Convert a Matplotlib figure to PNG bytes (includes legends/colorbars in the same figure).
    """
    buf = io.BytesIO()

    # Use tight bbox to include legend/colorbar axes placed outside the main axes
    fig.savefig(
        buf,
        format="png",
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.05
    )
    buf.seek(0)
    return buf.getvalue()


def overlay_contour(ax, binary_mask, extent, color="red", alpha=1.0, linewidth_px=2):
    """
    Draw a contour (boundary) overlay on a matplotlib axis using a thick boundary mask.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw on.
    binary_mask : np.ndarray (H, W) of bool
        Binary mask for which to compute boundaries.
    extent : list
        Extent used by imshow to map pixels to mm coordinates.
    color : str
        Color name for the contour.
    alpha : float
        Opacity of the contour overlay.
    linewidth_px : int
        Thickness of the boundary in pixels (approx). Implemented by dilation via repeated boundary extraction.
    """
    if binary_mask is None or not np.any(binary_mask):
        return

    # Compute a "thick" boundary mask
    # mode='thick' already returns a thicker boundary than 'inner'/'outer'
    boundary = find_boundaries(binary_mask.astype(bool), mode='thick')

    # If user wants an even thicker contour, expand it by re-applying boundaries on a grown mask.
    # This is a simple approximation to get thicker lines without extra deps.
    if linewidth_px > 2:
        grown = binary_mask.astype(bool).copy()
        for _ in range(linewidth_px - 2):
            # Grow boundary by OR-ing with boundary itself
            grown = np.logical_or(grown, boundary)
            boundary = find_boundaries(grown, mode='thick')

    # Create an RGBA overlay where only boundary pixels are visible
    overlay = np.zeros((boundary.shape[0], boundary.shape[1], 4), dtype=np.float32)

    # Map simple color names to RGB (extend if you want more)
    if color == "red":
        rgb = (1.0, 0.0, 0.0)
    elif color == "blue":
        rgb = (0.0, 0.0, 1.0)
    elif color == "purple":
        rgb = (1.0, 0.0, 1.0)
    elif color == "green":
        rgb = (0.0, 1.0, 0.0)
    else:
        # Fallback: white
        rgb = (1.0, 1.0, 1.0)

    overlay[boundary, 0] = rgb[0]
    overlay[boundary, 1] = rgb[1]
    overlay[boundary, 2] = rgb[2]
    overlay[boundary, 3] = alpha  # alpha channel only on boundary pixels

    ax.imshow(
        overlay,
        origin="lower",
        extent=extent,
        zorder=55,  # keep above uncertainty (40) and organs, close to GT/Pred layer
        interpolation="nearest",
    )


def overlay_fill(ax, binary_mask, extent, color, alpha=0.5, zorder=45):
    """
    Fill a binary mask with a flat semi-transparent color (region fill, not just
    its boundary). Used for the FP/FN error mask (idea #1) and for the
    binarized uncertainty view (idea #3), so both read as the same visual
    language and can be visually compared voxel-region-to-voxel-region.
    """
    if binary_mask is None or not np.any(binary_mask):
        return

    rgb = mcolors.to_rgb(color)
    overlay = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 4), dtype=np.float32)
    overlay[binary_mask, 0] = rgb[0]
    overlay[binary_mask, 1] = rgb[1]
    overlay[binary_mask, 2] = rgb[2]
    overlay[binary_mask, 3] = alpha

    ax.imshow(
        overlay,
        origin="lower",
        extent=extent,
        zorder=zorder,
        interpolation="nearest",
    )


def extract_slice(volume, axis, index):
    """Estrae una slice 2D da un volume 3D"""
    if axis == 0:  # Sagittal
        return volume[index, :, :]
    elif axis == 1:  # Coronal
        return volume[:, index, :]
    else:  # Axial (default: 2)
        return volume[:, :, index]


def apply_ct_windowing(ct_slice, level, width):
    """Applica windowing a una slice CT"""
    min_val = level - width / 2
    max_val = level + width / 2
    windowed = np.clip(ct_slice, min_val, max_val)
    windowed = (windowed - min_val) / (max_val - min_val)
    return windowed


def compute_dice_2d(pred_slice, gt_slice, threshold=0.5):
    """Calcola Dice Score su una slice 2D"""
    pred_bin = (pred_slice >= threshold).astype(np.uint8)
    gt_bin = (gt_slice >= threshold).astype(np.uint8)

    intersection = np.sum(pred_bin * gt_bin)
    union = np.sum(pred_bin) + np.sum(gt_bin)

    return 2.0 * intersection / union if union > 0 else 0.0


# ============================================================================
# FUNZIONI DI VISUALIZZAZIONE
# ============================================================================

uncertainty_maps, uncertainty_min_vals, uncertainty_max_vals = None, None, None

def create_overlay_visualization(ct_slice, gt_slice, pred_slice,
                                 show_gt, show_pred,
                                 gt_alpha, pred_alpha,
                                 extent,
                                 organ_slices=None,
                                 selected_organs=None,
                                 organ_alphas=None,
                                 uncertainty_slice=None,
                                 uncertainty_min=None,
                                 uncertainty_max=None,
                                 show_uncertainty=False,
                                 uncertainty_alpha=0.5,
                                 min_unc_threshold=0.01,
                                 show_error_mask=False,
                                 fp_slice=None,
                                 fn_slice=None,
                                 binarize_uncertainty=False):
    """Crea figura con overlay"""
    fig, ax = plt.subplots(figsize=(10, 10))

    # CT di base
    ax.imshow(ct_slice, cmap='gray', origin='lower', extent=extent, zorder=0)

    # Render organi (sotto GT e pred)
    if organ_slices and selected_organs and organ_alphas:
        zorder = 1
        for organ_name in selected_organs.keys():
            if organ_name in organ_slices:
                organ_masked = np.ma.masked_where(
                    organ_slices[organ_name] < 0.5,
                    organ_slices[organ_name]
                )
                cmap = ORGAN_COLORS.get(organ_name, 'viridis')
                ax.imshow(organ_masked, cmap=cmap, vmin=0, vmax=1,
                          origin='lower', alpha=organ_alphas[organ_name],
                          extent=extent, zorder=zorder)
                zorder += 1

    # Uncertainty map (zorder 40 continuo / 42 binarizzato, sotto la error mask e i contorni)
    im_unc = None
    if show_uncertainty and uncertainty_slice is not None:
        if binarize_uncertainty:
            # Vista binarizzata (idea #3): un unico colore piatto dove
            # l'incertezza supera la soglia calibrata tau*, nessun gradiente.
            # Stesso linguaggio visivo della error mask, cosi si puo confrontare
            # a colpo d'occhio con FP/FN invece che con una heatmap continua.
            unc_bin_mask = uncertainty_slice >= min_unc_threshold
            overlay_fill(ax, unc_bin_mask, extent, color="#facc15", alpha=uncertainty_alpha, zorder=42)
        else:
            # Maschera valori vicini a zero / sotto soglia
            unc_masked = np.ma.masked_where(uncertainty_slice < min_unc_threshold, uncertainty_slice)
            im_unc = ax.imshow(unc_masked, cmap='viridis', vmin=0, vmax=1,
                               origin='lower', alpha=uncertainty_alpha,
                               extent=extent, zorder=40)

    # Error mask: FP (arancio) / FN (ciano), zorder 45 - idea #1.
    # Riusa gt_slice/pred_slice gia binarizzati a monte, sopra l'incertezza
    # cosi il confronto visivo con la vista binarizzata sopra e diretto.
    if show_error_mask:
        overlay_fill(ax, fp_slice, extent, color="#f97316", alpha=0.5, zorder=45)
        overlay_fill(ax, fn_slice, extent, color="#38bdf8", alpha=0.5, zorder=45)

    # Ground Truth contour (red)
    if show_gt:
        gt_bin = (gt_slice >= 0.5)
        overlay_contour(
            ax=ax,
            binary_mask=gt_bin,
            extent=extent,
            color="red",
            alpha=gt_alpha,  # reuse existing slider as contour opacity
            linewidth_px=2  # change if you want thicker contour
        )

    # Prediction contour (blue)
    if show_pred:
        pred_bin = (pred_slice >= 0.5)
        overlay_contour(
            ax=ax,
            binary_mask=pred_bin,
            extent=extent,
            color="purple",
            alpha=pred_alpha,  # reuse existing slider as contour opacity
            linewidth_px=2
        )

    # Legenda
    handles = []

    # Aggiungi organi alla leggenda
    if organ_slices and selected_organs:
        for organ_name in selected_organs.keys():
            if organ_name in organ_slices and np.sum(organ_slices[organ_name] > 0.5) > 0:
                cmap_name = ORGAN_COLORS.get(organ_name, 'viridis')
                # Prendi il colore dal colormap
                cmap = plt.get_cmap(cmap_name)
                color = cmap(0.6)  # Usa un valore medio del colormap

                label = organ_name.replace('_', ' ').title()
                handles.append(mpatches.Patch(
                    color=color,
                    label=label,
                    alpha=organ_alphas[organ_name]
                ))

    # Aggiungi uncertainty alla leggenda
    if show_uncertainty and uncertainty_slice is not None:
        if binarize_uncertainty:
            handles.append(mpatches.Patch(
                color="#facc15",
                label=f'Uncertainty ≥ τ★ ({min_unc_threshold:.3f})',
                alpha=uncertainty_alpha
            ))
        else:
            handles.append(mpatches.Patch(
                color=plt.get_cmap('viridis')(0.7),
                label='Uncertainty',
                alpha=uncertainty_alpha
            ))

    # Aggiungi error mask alla leggenda
    if show_error_mask:
        if fp_slice is not None and np.any(fp_slice):
            handles.append(mpatches.Patch(color="#f97316", label='False Positive', alpha=0.5))
        if fn_slice is not None and np.any(fn_slice):
            handles.append(mpatches.Patch(color="#38bdf8", label='False Negative', alpha=0.5))

    # Aggiungi GT e Pred
    if show_gt:
        handles.append(mpatches.Patch(color='red', label='Reference', alpha=gt_alpha))
    if show_pred:
        handles.append(mpatches.Patch(color='purple', label='Prediction', alpha=pred_alpha))

    if handles:
        ax.legend(handles=handles, loc='upper right', fontsize='small', framealpha=0.8)

    ax.axis('off')

    # Aggiungi colorbar per uncertainty (solo in modalita heatmap continua:
    # im_unc resta None nella vista binarizzata, quindi qui non scatta)
    if show_uncertainty and im_unc is not None:
        # Crea asse per colorbar sotto l'immagine
        cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.02])  # [left, bottom, width, height]
        cbar = fig.colorbar(im_unc, cax=cbar_ax, orientation='horizontal')

        # Etichette personalizzate con threshold info
        cbar.set_label(f'normalized {uncertainty_type} uncertainty', fontsize=10)
        cbar.ax.tick_params(labelsize=9)

        if uncertainty_min is not None and uncertainty_max is not None:
            cbar.ax.text(0.5, 0.6,
                         f'Original range: [{uncertainty_min:.3f}, {uncertainty_max:.3f}]',
                         transform=cbar.ax.transAxes,
                         ha='center', va='top', fontsize=8)

        # Aggiungi tick custom
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels(['0.0\n(low)', '0.25', '0.5\n(medium)', '0.75', '1.0\n(high)'])

    return fig, ax


# ============================================================================
# STREAMLIT APP
# ============================================================================

st.title("🔬 Medical Image Segmentation Viewer")
st.sidebar.caption("Version - v2")

# Scansiona case disponibili
available_cases = scan_available_cases(OUTPUT_DIR)

if not available_cases:
    st.error(f"Nessun case trovato in {OUTPUT_DIR}")
    st.stop()

# ============================================================================
# SIDEBAR CONTROLS
# ============================================================================
st.sidebar.header("⚙️ Controls")

# Selezione case
selected_case = st.sidebar.selectbox("Select Case", available_cases)

# Carica dati
try:
    data = load_case_data(selected_case, OUTPUT_DIR, GT_DIR, INPUT_DIR)
except Exception as e:
    st.error(f"Errore nel caricamento: {e}")
    st.stop()

st.sidebar.markdown("---")

# Selezione piano anatomico
view_plane = st.sidebar.radio("View Plane", ["Axial", "Coronal", "Sagittal"])
axis_map = {"Axial": 2, "Coronal": 1, "Sagittal": 0}
axis = axis_map[view_plane]

# Slider per slice
# key esplicita: serve al bottone "Jump to worst slice" (idea #4) per poter
# scrivere il valore in session_state. Streamlit vieta di riassegnare
# session_state[key] DOPO che il widget con quel key e' gia' stato
# istanziato in questo stesso run: per questo il bottone (piu' in basso nello
# script) scrive in un key "ponte" e forza un rerun, e qui — prima che lo
# slider venga creato — il valore ponte viene travasato nel key vero.
if st.session_state.get("pending_jump_slice") is not None:
    st.session_state["slice_idx_slider"] = st.session_state.pop("pending_jump_slice")

max_slice = data['shape'][axis] - 1
slice_idx = st.sidebar.slider("Slice Index", 0, max_slice, max_slice // 2, key="slice_idx_slider")

st.sidebar.markdown("---")

# Overlay controls
st.sidebar.subheader("Overlay Settings")
show_gt = st.sidebar.checkbox("Show Reference", value=True)
show_pred = st.sidebar.checkbox("Show Prediction", value=True)

gt_alpha = st.sidebar.slider("Reference opacity", 0.0, 1.0, 0.5, 0.05)
pred_alpha = st.sidebar.slider("Pred opacity", 0.0, 1.0, 0.5, 0.05)

# Error mask FP/FN come aree piene, indipendenti dai contorni
show_error_mask = st.sidebar.checkbox(
    "Show Error Mask (FP / FN)", value=True,
    help="Riempie i falsi positivi (predetto, non in GT) in arancio e i falsi "
         "negativi (in GT, non predetto) in azzurro, come aree piene."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Uncertainty Maps")

show_uncertainty = st.sidebar.checkbox("Show Uncertainty", value=False)

uncertainty_type = None
uncertainty_alpha = 0.5
uncertainty_data = None
boundary_distance_mm = 5.0
min_display_threshold = 0.01
binarize_uncertainty = False
compare_mode = False
thresholds_dict = {}

if show_uncertainty:
    # Carica SOLO se checkbox attivo
    uncertainty_maps, uncertainty_min_vals, uncertainty_max_vals = load_uncertainty_maps(selected_case, UNCERTAINTY_DIR, data['shape'])

    if uncertainty_maps:
        available_unc_types = list(uncertainty_maps.keys())

        # Categoria (gruppo) prima, poi la misura dentro la
        # categoria — al posto della lista flat di ~21 radio. Solo i gruppi
        # con almeno una misura presente su disco per questo case vengono
        # mostrati.
        group_options = [
            g for g, types in UNCERTAINTY_GROUPS.items()
            if any(t in available_unc_types for t in types)
        ]
        selected_group = st.sidebar.selectbox("Uncertainty category", group_options)
        group_items = [t for t in UNCERTAINTY_GROUPS[selected_group] if t in available_unc_types]

        uncertainty_type = st.sidebar.radio(
            "Measure",
            group_items,
            format_func=display_name
        )
        uncertainty_alpha = st.sidebar.slider(
            "Uncertainty opacity",
            0.0, 1.0, 0.6, 0.05
        )

        # Select the chosen uncertainty data
        uncertainty_data = uncertainty_maps.get(uncertainty_type)

        # Load thresholds CSV and look up optimal t* for this case/type
        thresholds_dict = load_thresholds(THRESHOLD_DIR)
        min_display_threshold = get_tstar(thresholds_dict, selected_case, uncertainty_type)

        # Show threshold as read-only metric (no slider)
        st.sidebar.markdown("##### Display Threshold (τ★)")
        st.sidebar.metric(
            label="Optimal threshold (τ★)",
            value=f"{min_display_threshold:.4f}",
            help="Loaded from resultsVR.csv — optimal entropy cutoff for this case and uncertainty type."
        )

        # Heatmap continua (comportamento originale) vs vista
        # binarizzata a tau* — un colore piatto invece del gradiente viridis,
        # per confrontarla visivamente con la error mask qui sopra.
        uncertainty_display_mode = st.sidebar.radio(
            "Uncertainty display",
            ["Heatmap (continuous)", "Binarized @ τ★"],
            help="Binarized mostra un colore piatto solo dove l'incertezza "
                 "supera τ★ (la soglia calibrata), invece del gradiente "
                 "viridis continuo."
        )
        binarize_uncertainty = (uncertainty_display_mode == "Binarized @ τ★")

        # Confronto affiancato entropy / DEU / EBCU, indipendente
        # dalla misura singola selezionata sopra. Disponibile solo se almeno
        # due delle tre mappe esistono su disco per questo case.
        compare_available = [t for t in COMPARE_MODE_TYPES if t in available_unc_types]
        if len(compare_available) >= 2:
            compare_mode = st.sidebar.checkbox(
                "Compare mode: Entropy · DEU · EBCU side-by-side",
                value=False,
                help="Mostra entropy, DEU_entropy ed ebcu_entropy affiancate, "
                     "ognuna con la propria soglia τ★, al posto dell'immagine singola."
            )

    else:
        st.sidebar.warning(f"No uncertainty maps found for {selected_case}")

st.sidebar.markdown("---")

# Windowing CT
ct_level = 40
ct_width = 370


st.sidebar.subheader("TotalSegmentator Organs")
show_organs = st.sidebar.checkbox("Show Organs", value=False)


# Definisci colori per organi
ORGAN_COLORS = {
    'liver': 'Greens',
    'spleen': 'Purples',
    'kidney_left': 'Oranges',
    'kidney_right': 'Oranges',
    'pancreas': 'YlOrBr',
    'stomach': 'PuRd',
    'urinary_bladder': 'Blues',
    'vertebrae_L4': 'Greys',
    'vertebrae_T1': 'Greys',
    'femur_left': 'bone',
    'femur_right': 'bone',
    'humerus_left': 'bone',
    'humerus_right': 'bone',
}

# Ottieni lista organi disponibili (veloce, solo nomi)
available_organs = get_available_organs(selected_case, TOTALSEG_DIR)

# Checkbox per ogni organo disponibile
selected_organs = {}
organ_alphas = {}

if show_organs:
    available_organs = get_available_organs(selected_case, TOTALSEG_DIR)

    if available_organs:
        for organ_name in available_organs:
            show = st.sidebar.checkbox(
                f"Show {organ_name.replace('_', ' ').title()}",
                value=False,
                key=f"organ_{organ_name}"
            )
            if show:  # Carica SOLO se checkbox attivato
                selected_organs[organ_name] = True
                organ_alphas[organ_name] = st.sidebar.slider(
                    f"{organ_name} opacity",
                    0.0, 1.0, 0.3, 0.05,
                    key=f"alpha_{organ_name}"
                )


# ============================================================================
# PROCESSING & DISPLAY
# ============================================================================

# Estrai slice
ct_slice = extract_slice(data['ct'], axis, slice_idx)
gt_slice = extract_slice(data['gt'], axis, slice_idx)
pred_slice = extract_slice(data['pred'], axis, slice_idx)

# Calcola FP / FN sulla slice corrente (idea #1)
fp_slice = None
fn_slice = None
if show_error_mask:
    gt_bin_slice = gt_slice >= 0.5
    pred_bin_slice = pred_slice >= 0.5
    fp_slice = pred_bin_slice & ~gt_bin_slice
    fn_slice = gt_bin_slice & ~pred_bin_slice

# Estrai slice organi (SOLO quelli selezionati)
organ_slices = {}
if selected_organs:
    for organ_name in selected_organs.keys():
        # Carica l'organo on-demand
        organ_data = load_single_organ(selected_case, organ_name, TOTALSEG_DIR)
        if organ_data is not None:
            organ_slice = extract_slice(organ_data, axis, slice_idx)
            organ_slices[organ_name] = organ_slice

# Applica windowing
ct_slice = apply_ct_windowing(ct_slice, ct_level, ct_width)


# Determina voxel sizes per il piano corrente
voxel_sizes = data['voxel_sizes']
if view_plane == "Axial":
    vs_row, vs_col = voxel_sizes[0], voxel_sizes[1]
elif view_plane == "Coronal":
    vs_row, vs_col = voxel_sizes[0], voxel_sizes[2]
else:  # Sagittal
    vs_row, vs_col = voxel_sizes[1], voxel_sizes[2]


row_slice, col_slice = slice(None), slice(None)

# Estrai slice uncertainty
uncertainty_slice = None
if show_uncertainty and uncertainty_data is not None:
    uncertainty_slice = extract_slice(uncertainty_data, axis, slice_idx)

# Colocalization: quota di voxel d'errore (FP unito FN, su questa slice)
# coperti dall'incertezza binarizzata a tau*. Indipendente dalla modalita di
# display scelta (Heatmap/Binarized) — usa comunque tau* internamente,
# perche' e' una metrica, non quello che viene disegnato.
colocalization = None
if show_error_mask and show_uncertainty and uncertainty_slice is not None and fp_slice is not None and fn_slice is not None:
    error_mask_slice = fp_slice | fn_slice
    if np.any(error_mask_slice):
        unc_bin_for_coloc = uncertainty_slice >= min_display_threshold
        colocalization = float(np.sum(error_mask_slice & unc_bin_for_coloc)) / float(np.sum(error_mask_slice))

# Calcola extent in mm
extent = [0, ct_slice.shape[1] * vs_col, 0, ct_slice.shape[0] * vs_row]

# ============================================================================
# IMMAGINE CENTRALE
# ============================================================================
if compare_mode:
    # Entropy / DEU / EBCU affiancate, ognuna con la propria tau*.
    st.markdown("##### Compare mode — Entropy · DEU · EBCU")
    compare_types = [t for t in COMPARE_MODE_TYPES if t in uncertainty_maps]
    compare_cols = st.columns(len(compare_types))

    for col, ctype in zip(compare_cols, compare_types):
        with col:
            unc_slice_c = extract_slice(uncertainty_maps[ctype], axis, slice_idx)
            tstar_c = get_tstar(thresholds_dict, selected_case, ctype)

            fig_c, ax_c = create_overlay_visualization(
                ct_slice, gt_slice, pred_slice,
                show_gt, show_pred,
                gt_alpha, pred_alpha,
                extent,
                organ_slices=organ_slices if selected_organs else None,
                selected_organs=selected_organs if selected_organs else None,
                organ_alphas=organ_alphas if selected_organs else None,
                uncertainty_slice=unc_slice_c,
                uncertainty_min=uncertainty_min_vals.get(ctype),
                uncertainty_max=uncertainty_max_vals.get(ctype),
                show_uncertainty=True,
                uncertainty_alpha=uncertainty_alpha,
                min_unc_threshold=tstar_c,
                show_error_mask=show_error_mask,
                fp_slice=fp_slice,
                fn_slice=fn_slice,
                binarize_uncertainty=binarize_uncertainty
            )
            ax_c.set_title(f"{display_name(ctype)}  (τ★={tstar_c:.3f})", fontsize=11)
            st.pyplot(fig_c)
            plt.close(fig_c)
else:
    # Immagine "un po' piu piccola": confinata in una colonna piu stretta
    # invece di occupare tutta la larghezza del main, che a schermo intero
    # la faceva risultare enorme.
    col_margin_l, col_img, col_margin_r = st.columns([1, 4, 1])
    with col_img:
        fig, ax = create_overlay_visualization(
            ct_slice, gt_slice, pred_slice,
            show_gt, show_pred,
            gt_alpha, pred_alpha,
            extent,
            organ_slices=organ_slices if selected_organs else None,
            selected_organs=selected_organs if selected_organs else None,
            organ_alphas=organ_alphas if selected_organs else None,
            uncertainty_slice=uncertainty_slice,
            uncertainty_min=uncertainty_min_vals.get(uncertainty_type) if uncertainty_type else None,
            uncertainty_max=uncertainty_max_vals.get(uncertainty_type) if uncertainty_type else None,
            show_uncertainty=show_uncertainty,
            uncertainty_alpha=uncertainty_alpha,
            min_unc_threshold=min_display_threshold,
            show_error_mask=show_error_mask,
            fp_slice=fp_slice,
            fn_slice=fn_slice,
            binarize_uncertainty=binarize_uncertainty
        )
        ax.set_title(
            f"{selected_case} | {view_plane} | Slice {slice_idx}/{max_slice}",
            fontsize=14, pad=15
        )

        st.pyplot(fig)
        # Download button (must be before closing the figure)
        png_bytes = figure_to_png_bytes(fig, dpi=200)
        st.download_button(
            label="Download PNG (CT + overlays)",
            data=png_bytes,
            file_name=f"{selected_case}_{view_plane}_slice_{slice_idx}_{uncertainty_type}.png",
            mime="image/png"
        )
        plt.close(fig)

# ============================================================================
# SPARKLINE PER-SLICE (idea #4)
# ============================================================================
st.markdown("---")

dice_per_slice, has_gt_per_slice = compute_dice_per_slice(selected_case, axis, OUTPUT_DIR, GT_DIR, INPUT_DIR)

# Slice peggiore SOLO tra quelle con GT non vuoto: altrimenti vince sempre
# una slice di sfondo puro, dove Dice=0 e' banale, non un fallimento.
if np.any(has_gt_per_slice):
    worst_idx = int(np.argmin(np.where(has_gt_per_slice, dice_per_slice, np.inf)))
else:
    worst_idx = int(np.argmin(dice_per_slice))
worst_dice = float(dice_per_slice[worst_idx])

spark_label_col, spark_btn_col = st.columns([4, 1])
with spark_label_col:
    st.caption(f"Dice per slice — {view_plane.lower()} · worst: slice {worst_idx} (Dice {worst_dice:.2f})")
with spark_btn_col:
    if st.button("⤓ Jump to worst slice", help=f"Slice {worst_idx}, Dice {worst_dice:.2f}"):
        st.session_state["pending_jump_slice"] = worst_idx
        st.rerun()

fig_spark, ax_spark = plt.subplots(figsize=(10, 1.3))
ax_spark.plot(dice_per_slice, color="#2dd4bf", linewidth=1.2)
ax_spark.axvline(slice_idx, color="#facc15", linestyle="--", linewidth=1.2)
ax_spark.scatter([worst_idx], [worst_dice], color="#ef4444", zorder=5, s=25)
ax_spark.set_xlim(0, max(len(dice_per_slice) - 1, 1))
ax_spark.set_ylim(0, 1.05)
ax_spark.set_yticks([0, 0.5, 1.0])
ax_spark.tick_params(labelsize=8)
ax_spark.set_xlabel("slice index", fontsize=8)
for spine in ["top", "right"]:
    ax_spark.spines[spine].set_visible(False)
fig_spark.tight_layout()
st.pyplot(fig_spark)
plt.close(fig_spark)

# ============================================================================
# INFO PANEL
# ============================================================================
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Case Metrics")

# Dice 2D (slice corrente) e Dice 3D (idea #10, sull'intero case)
dice_2d = compute_dice_2d(pred_slice, gt_slice)
dice_3d, gt_vol_ml, pred_vol_ml = compute_case_metrics(selected_case, OUTPUT_DIR, GT_DIR, INPUT_DIR)

col_dice2d, col_dice3d = st.sidebar.columns(2)
col_dice2d.metric("Dice (2D slice)", f"{dice_2d:.4f}")
col_dice3d.metric("Dice (3D case)", f"{dice_3d:.4f}")

# Colocalization: quanto l'incertezza calibrata copre gli errori veri
if colocalization is not None:
    st.sidebar.metric(
        "Colocalization",
        f"{colocalization * 100:.0f}%",
        help="Quota di voxel FP/FN (su questa slice) coperti da incertezza ≥ τ★."
    )

# Info voxel + volumi (idea #10)
n_gt = np.sum(gt_slice > 0.5)
n_pred = np.sum(pred_slice > 0.5)
st.sidebar.text(f"GT voxels: {int(n_gt)} (slice) · {gt_vol_ml:.1f} mL (case)")
st.sidebar.text(f"Pred voxels: {int(n_pred)} (slice) · {pred_vol_ml:.1f} mL (case)")

# Info FP / FN sulla slice (idea #1)
if show_error_mask:
    n_fp = int(np.sum(fp_slice)) if fp_slice is not None else 0
    n_fn = int(np.sum(fn_slice)) if fn_slice is not None else 0
    st.sidebar.text(f"FP voxels: {n_fp}")
    st.sidebar.text(f"FN voxels: {n_fn}")

# Info shape
st.sidebar.markdown("---")
st.sidebar.subheader("📐 Volume Info")
st.sidebar.text(f"Shape: {data['shape']}")
st.sidebar.text(f"Voxel size: {[f'{v:.2f}' for v in voxel_sizes]} mm")
