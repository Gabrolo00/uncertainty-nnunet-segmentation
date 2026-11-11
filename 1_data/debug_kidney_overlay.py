"""
Controllo visivo delle label del rene: PNG con tre viste della CT e la label sovrapposta,
dal formato TotalSegmentator oppure da una cartella nnU-Net (immagini + label).

Uso: python debug_kidney_overlay.py --case_ids s0011 s0013 --out_dir <dir> [--images_dir <dir> --labels_dir <dir>]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset_root", default="/mnt/sda2/DATASETS/TOTALSEGMENTATOR",
                    help="Modalità label: cartella raw TotalSegmentator (dataset_root/{case}/ct.nii.gz + "
                         "segmentations/kidney_{left,right}.nii.gz). "
                         "Modalità predizione (richiede --images_dir e --labels_dir): cartella con le "
                         "predizioni flat dataset_root/{case}.nii.gz, es. fold_N/validation/ di nnU-Net.")
    p.add_argument("--images_dir", default=None,
                    help="Modalità predizione: cartella nnU-Net imagesTr/imagesTs con {case}_0000.nii.gz (CT).")
    p.add_argument("--labels_dir", default=None,
                    help="Modalità predizione: cartella nnU-Net labelsTr/labelsTs con {case}.nii.gz (ground truth).")
    p.add_argument("--case_ids", required=True,
                    help="Lista comma-separated di case id, es. 's0001,s0002'")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--wl", type=float, default=40, help="Window level CT (HU)")
    p.add_argument("--ww", type=float, default=400, help="Window width CT (HU)")
    return p.parse_args()


def load_case_ras(case_dir):
    """Carica ct + maschere separate, riorientate a RAS canonico così gli assi
    array hanno un significato anatomico fisso: axis0=R-L, axis1=A-P, axis2=I-S."""
    ct_img = nib.as_closest_canonical(nib.load(os.path.join(case_dir, "ct.nii.gz")))
    kl_img = nib.as_closest_canonical(
        nib.load(os.path.join(case_dir, "segmentations", "kidney_left.nii.gz")))
    kr_img = nib.as_closest_canonical(
        nib.load(os.path.join(case_dir, "segmentations", "kidney_right.nii.gz")))

    ct = np.asarray(ct_img.dataobj, dtype=np.float32)
    kl = np.asarray(kl_img.dataobj) > 0
    kr = np.asarray(kr_img.dataobj) > 0
    return ct, kl, kr


def load_case_ras_prediction(case_id, dataset_root, images_dir, labels_dir):
    """Come load_case_ras ma per confrontare la predizione del modello (mappa
    binaria kidney unica, non più left/right separati) con la ground truth,
    sovrapposte alla CT vera letta da imagesTr/imagesTs."""
    ct_img = nib.as_closest_canonical(
        nib.load(os.path.join(images_dir, f"{case_id}_0000.nii.gz")))
    gt_img = nib.as_closest_canonical(
        nib.load(os.path.join(labels_dir, f"{case_id}.nii.gz")))
    pred_img = nib.as_closest_canonical(
        nib.load(os.path.join(dataset_root, f"{case_id}.nii.gz")))

    ct = np.asarray(ct_img.dataobj, dtype=np.float32)
    gt = np.asarray(gt_img.dataobj) > 0
    pred = np.asarray(pred_img.dataobj) > 0
    return ct, gt, pred


def best_index(mask_combined, axis):
    other_axes = tuple(a for a in range(3) if a != axis)
    counts = mask_combined.sum(axis=other_axes)
    if counts.max() == 0:
        return mask_combined.shape[axis] // 2
    return int(np.argmax(counts))


def window_ct(plane, wl, ww):
    lo, hi = wl - ww / 2, wl + ww / 2
    return np.clip((plane - lo) / (hi - lo), 0, 1)


def make_rgb_overlay(ct_plane, kl_plane, kr_plane, alpha=0.45):
    rgb = np.stack([ct_plane] * 3, axis=-1)
    out = rgb.copy()
    out[kl_plane] = (1 - alpha) * rgb[kl_plane] + alpha * np.array([1.0, 0.0, 0.0])
    out[kr_plane] = (1 - alpha) * rgb[kr_plane] + alpha * np.array([0.0, 0.45, 1.0])
    return out


def make_rgb_overlay_tpfnfp(ct_plane, gt_plane, pred_plane, alpha=0.45):
    """verde=TP (accordo), rosso=FN (persa dal modello), blu=FP (falso allarme)."""
    tp = gt_plane & pred_plane
    fn = gt_plane & ~pred_plane
    fp = pred_plane & ~gt_plane
    rgb = np.stack([ct_plane] * 3, axis=-1)
    out = rgb.copy()
    out[tp] = (1 - alpha) * rgb[tp] + alpha * np.array([0.0, 1.0, 0.0])
    out[fn] = (1 - alpha) * rgb[fn] + alpha * np.array([1.0, 0.0, 0.0])
    out[fp] = (1 - alpha) * rgb[fp] + alpha * np.array([0.0, 0.45, 1.0])
    return out


# Per ciascun axis "fissato" (quello lungo cui si taglia), definisce:
# - nome vista anatomica
# - etichette (sinistra, destra, basso, alto) del plot risultante
VIEW_SPECS = {
    0: dict(name="sagittale", left="P", right="A", bottom="I", top="S"),
    1: dict(name="coronale", left="L", right="R", bottom="I", top="S"),
    2: dict(name="assiale", left="L", right="R", bottom="P", top="A"),
}


def plot_case(case_id, ct, kl, kr, out_path, wl, ww):
    combined = kl | kr
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

    for ax_plot, axis in zip(axes, (0, 1, 2)):
        idx = best_index(combined, axis)
        sl = [slice(None)] * 3
        sl[axis] = idx
        sl = tuple(sl)

        ct_plane = window_ct(ct[sl], wl, ww)
        kl_plane = kl[sl]
        kr_plane = kr[sl]

        # axis rimossi in ordine crescente: le due dimensioni residue restano
        # nell'ordine originale (es. axis=1 rimosso -> residue (0,2) = (R-L, I-S))
        # trasponiamo per avere la dimensione "verticale anatomica" sulle righe
        ct_plane = ct_plane.T
        kl_plane = kl_plane.T
        kr_plane = kr_plane.T

        overlay = make_rgb_overlay(ct_plane, kl_plane, kr_plane)
        spec = VIEW_SPECS[axis]

        ax_plot.imshow(overlay, origin="lower")
        ax_plot.set_title(f"{spec['name']} (slice {idx})")
        ax_plot.axis("off")
        h, w = overlay.shape[0], overlay.shape[1]
        ax_plot.text(0.02 * w, 0.5 * h, spec["left"], color="yellow", fontsize=14, fontweight="bold")
        ax_plot.text(0.96 * w, 0.5 * h, spec["right"], color="yellow", fontsize=14, fontweight="bold")
        ax_plot.text(0.5 * w, 0.03 * h, spec["bottom"], color="yellow", fontsize=14, fontweight="bold")
        ax_plot.text(0.5 * w, 0.94 * h, spec["top"], color="yellow", fontsize=14, fontweight="bold")

    n_left, n_right = int(kl.sum()), int(kr.sum())
    fig.suptitle(f"{case_id} — rosso=kidney_left ({n_left} vox), blu=kidney_right ({n_right} vox)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_case_prediction(case_id, ct, gt, pred, out_path, wl, ww):
    combined = gt | pred
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

    for ax_plot, axis in zip(axes, (0, 1, 2)):
        idx = best_index(combined, axis)
        sl = [slice(None)] * 3
        sl[axis] = idx
        sl = tuple(sl)

        ct_plane = window_ct(ct[sl], wl, ww)
        gt_plane = gt[sl]
        pred_plane = pred[sl]

        ct_plane = ct_plane.T
        gt_plane = gt_plane.T
        pred_plane = pred_plane.T

        overlay = make_rgb_overlay_tpfnfp(ct_plane, gt_plane, pred_plane)
        spec = VIEW_SPECS[axis]

        ax_plot.imshow(overlay, origin="lower")
        ax_plot.set_title(f"{spec['name']} (slice {idx})")
        ax_plot.axis("off")
        h, w = overlay.shape[0], overlay.shape[1]
        ax_plot.text(0.02 * w, 0.5 * h, spec["left"], color="yellow", fontsize=14, fontweight="bold")
        ax_plot.text(0.96 * w, 0.5 * h, spec["right"], color="yellow", fontsize=14, fontweight="bold")
        ax_plot.text(0.5 * w, 0.03 * h, spec["bottom"], color="yellow", fontsize=14, fontweight="bold")
        ax_plot.text(0.5 * w, 0.94 * h, spec["top"], color="yellow", fontsize=14, fontweight="bold")

    tp, fn, fp = int((gt & pred).sum()), int((gt & ~pred).sum()), int((pred & ~gt).sum())
    dice = 2 * tp / (2 * tp + fn + fp) if (2 * tp + fn + fp) > 0 else float("nan")
    fig.suptitle(f"{case_id} — verde=TP ({tp}), rosso=FN ({fn}), blu=FP ({fp}), Dice={dice:.4f}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    case_ids = [c.strip() for c in args.case_ids.split(",") if c.strip()]
    prediction_mode = args.images_dir is not None and args.labels_dir is not None

    for case_id in case_ids:
        out_path = os.path.join(args.out_dir, f"{case_id}_qa.png")
        if prediction_mode:
            ct, gt, pred = load_case_ras_prediction(case_id, args.dataset_root, args.images_dir, args.labels_dir)
            plot_case_prediction(case_id, ct, gt, pred, out_path, args.wl, args.ww)
            tp, fn, fp = int((gt & pred).sum()), int((gt & ~pred).sum()), int((pred & ~gt).sum())
            print(f"{case_id}: salvato {out_path} (TP={tp}, FN={fn}, FP={fp})")
        else:
            case_dir = os.path.join(args.dataset_root, case_id)
            ct, kl, kr = load_case_ras(case_dir)
            plot_case(case_id, ct, kl, kr, out_path, args.wl, args.ww)
            print(f"{case_id}: salvato {out_path} (voxel sx={int(kl.sum())}, dx={int(kr.sum())})")


if __name__ == "__main__":
    main()
